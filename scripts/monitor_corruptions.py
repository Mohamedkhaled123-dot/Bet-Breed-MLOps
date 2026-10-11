import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from evidently import Report
from evidently.presets import DataDriftPreset
from PIL import Image
from sklearn.decomposition import PCA

from bet_breed_mlops.airflow_trigger import trigger_retraining_dag
from bet_breed_mlops.confidence import MANIFEST_PATH, load_temperature
from bet_breed_mlops.corruptions import CORRUPTIONS
from bet_breed_mlops.data import (
    PetBreedDataset,
    eval_transform,
)
from bet_breed_mlops.model import PetBreedClassifier
from bet_breed_mlops.monitoring import (
    calibrate_drift_threshold,
    confidence_psi,
    pixel_drift_score,
    rbf_mmd_squared,
)
from bet_breed_mlops.optimization import select_stratified_indices
from bet_breed_mlops.prometheus_metrics import save_monitoring_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "resnet18.pt"
CORRUPTED_ROOT = PROJECT_ROOT / "data" / "processed" / "corrupted_test"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCORECARD_PATH = REPORTS_DIR / "drift_detector_scorecard.json"
DRIFT_REPORT_PATH = REPORTS_DIR / "evidently" / "latest_batch_drift.html"
SAMPLES_PER_CLASS = 4
BATCH_SIZE = 32
THRESHOLD_QUANTILE = 0.95
DETECTORS = ("embedding_mmd", "pixel_statistics", "confidence_psi")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int | None = None
    relative_test_path: Path | None = None


@dataclass
class ExtractedBatch:
    embeddings: np.ndarray
    pixel_statistics: np.ndarray
    confidences: np.ndarray
    batch_latency_seconds: list[float]


def load_model(device: torch.device) -> PetBreedClassifier:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = PetBreedClassifier(
        num_classes=37,
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def sample_split(
    split: str,
    samples_per_class: int | None = SAMPLES_PER_CLASS,
) -> list[ImageRecord]:
    dataset = PetBreedDataset(MANIFEST_PATH, split, transform=None)
    labels = [record["class_index"] for record in dataset.records]
    indices = (
        list(range(len(labels)))
        if samples_per_class is None
        else select_stratified_indices(
            labels,
            samples_per_class=samples_per_class,
            seed=20261008,
        )
    )
    indices = np.random.default_rng(20261008).permutation(indices).tolist()
    records = []
    test_root = PROJECT_ROOT / "data" / "raw" / "split" / "test"
    for index in indices:
        record = dataset.records[index]
        relative_path = Path(record["path"])
        absolute_path = PROJECT_ROOT / relative_path
        test_relative = (
            absolute_path.relative_to(test_root)
            if split == "test"
            else None
        )
        records.append(
            ImageRecord(
                path=absolute_path,
                label=int(record["class_index"]),
                relative_test_path=test_relative,
            )
        )
    return records


def pixel_statistics(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    grayscale = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    horizontal_edges = np.abs(np.diff(grayscale, axis=1)).mean()
    vertical_edges = np.abs(np.diff(grayscale, axis=0)).mean()
    return np.array(
        [
            float(grayscale.mean()),
            float(grayscale.std()),
            float((horizontal_edges + vertical_edges) / 2.0),
        ],
        dtype=np.float64,
    )


def extract_batch(
    model: PetBreedClassifier,
    records: list[ImageRecord],
    device: torch.device,
    temperature: float,
) -> ExtractedBatch:
    feature_batches = []
    capture = []

    def capture_embedding(module, inputs, output) -> None:
        capture.append(output.flatten(1).detach().cpu())

    hook = model.backbone.avgpool.register_forward_hook(capture_embedding)
    all_pixels = []
    all_confidences = []
    batch_latencies = []
    try:
        model.eval()
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start : start + BATCH_SIZE]
            tensors = []
            pixels = []
            for record in batch_records:
                with Image.open(record.path) as opened_image:
                    image = opened_image.convert("RGB")
                    pixels.append(pixel_statistics(image))
                    tensors.append(eval_transform(image))
            images = torch.stack(tensors).to(device, non_blocking=True)
            capture.clear()
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(images)
            torch.cuda.synchronize(device)
            batch_latencies.append(time.perf_counter() - started)
            if len(capture) != 1:
                raise RuntimeError("Expected one pooled embedding capture per batch.")
            feature_batches.append(capture[0])
            confidence = torch.softmax(logits / temperature, dim=1).max(dim=1).values
            all_confidences.extend(confidence.float().cpu().tolist())
            all_pixels.extend(pixels)
    finally:
        hook.remove()

    if not feature_batches:
        raise ValueError("Cannot monitor an empty image batch.")
    return ExtractedBatch(
        embeddings=torch.cat(feature_batches).numpy().astype(np.float64),
        pixel_statistics=np.stack(all_pixels),
        confidences=np.asarray(all_confidences, dtype=np.float64),
        batch_latency_seconds=batch_latencies,
    )


def make_batches(extracted: ExtractedBatch) -> list[ExtractedBatch]:
    batch_count = int(np.ceil(len(extracted.confidences) / BATCH_SIZE))
    batches = []
    for index in range(batch_count):
        start = index * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(extracted.confidences))
        if end - start < 2 and batches:
            previous = batches.pop()
            start = (index - 1) * BATCH_SIZE
            previous_end = end
            batches.append(
                ExtractedBatch(
                    embeddings=extracted.embeddings[start:previous_end],
                    pixel_statistics=extracted.pixel_statistics[start:previous_end],
                    confidences=extracted.confidences[start:previous_end],
                    batch_latency_seconds=previous.batch_latency_seconds
                    + extracted.batch_latency_seconds[index : index + 1],
                )
            )
            continue
        batches.append(
            ExtractedBatch(
                embeddings=extracted.embeddings[start:end],
                pixel_statistics=extracted.pixel_statistics[start:end],
                confidences=extracted.confidences[start:end],
                batch_latency_seconds=extracted.batch_latency_seconds[
                    index : index + 1
                ],
            )
        )
    return batches


def score_batches(
    reference: ExtractedBatch,
    current: ExtractedBatch,
) -> dict[str, float]:
    reference_scale = reference.embeddings.std(axis=0)
    scale = np.maximum(reference_scale, 1e-6)
    normalized_reference = (
        reference.embeddings - reference.embeddings.mean(axis=0)
    ) / scale
    normalized_current = (
        current.embeddings - reference.embeddings.mean(axis=0)
    ) / scale
    return {
        "embedding_mmd": rbf_mmd_squared(
            normalized_reference,
            normalized_current,
        ),
        "pixel_statistics": pixel_drift_score(
            reference.pixel_statistics,
            current.pixel_statistics,
        ),
        "confidence_psi": confidence_psi(
            reference.confidences,
            current.confidences,
        ),
    }


def score_group(
    reference: ExtractedBatch,
    extracted: ExtractedBatch,
) -> list[dict[str, float]]:
    return [
        score_batches(reference, batch)
        for batch in make_batches(extracted)
    ]


def corruption_records(
    clean_test_records: list[ImageRecord],
    corruption: str,
    severity: int,
) -> list[ImageRecord]:
    scenario_root = CORRUPTED_ROOT / corruption / f"severity_{severity}"
    records = []
    for clean_record in clean_test_records:
        if clean_record.relative_test_path is None or clean_record.label is None:
            raise ValueError("Corruption evaluation requires labeled clean test records.")
        path = scenario_root / clean_record.relative_test_path
        if corruption == "jpeg":
            path = path.with_suffix(".jpg")
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            ImageRecord(
                path=path,
                label=clean_record.label,
                relative_test_path=clean_record.relative_test_path,
            )
        )
    return records


def threshold_map(
    reference: ExtractedBatch,
    clean_calibration: ExtractedBatch,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    calibration_scores = score_group(reference, clean_calibration)
    thresholds = {
        detector: calibrate_drift_threshold(
            np.asarray([score[detector] for score in calibration_scores]),
            quantile=THRESHOLD_QUANTILE,
        )
        for detector in DETECTORS
    }
    return thresholds, calibration_scores


def summarize_detections(
    scores: list[dict[str, float]],
    thresholds: dict[str, float],
) -> dict:
    return {
        detector: {
            "flagged_batches": sum(
                score[detector] > thresholds[detector] for score in scores
            ),
            "batch_count": len(scores),
            "detected": any(
                score[detector] > thresholds[detector] for score in scores
            ),
            "max_score": max(score[detector] for score in scores),
        }
        for detector in DETECTORS
    }


def mean_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        detector: float(np.mean([score[detector] for score in scores]))
        for detector in DETECTORS
    }


def batch_alerts(
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, bool]:
    return {
        detector: scores[detector] > thresholds[detector]
        for detector in DETECTORS
    }


def evidential_dataframes(
    reference: ExtractedBatch,
    current: ExtractedBatch,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    component_count = min(8, reference.embeddings.shape[0] - 1)
    projection = PCA(n_components=component_count, random_state=0)
    reference_projection = projection.fit_transform(reference.embeddings)
    current_projection = projection.transform(current.embeddings)
    embedding_columns = [
        f"embedding_pc_{index + 1:02d}"
        for index in range(component_count)
    ]
    columns = embedding_columns + [
        "pixel_brightness",
        "pixel_contrast",
        "pixel_sharpness",
        "confidence",
    ]

    def frame(data: ExtractedBatch, projected: np.ndarray) -> pd.DataFrame:
        combined = np.column_stack(
            (
                projected,
                data.pixel_statistics,
                data.confidences,
            )
        )
        return pd.DataFrame(combined, columns=columns)

    return frame(reference, reference_projection), frame(current, current_projection), columns


def write_evidently_report(
    reference: ExtractedBatch,
    current: ExtractedBatch,
) -> None:
    reference_frame, current_frame, columns = evidential_dataframes(
        reference,
        current,
    )
    report = Report(metrics=[DataDriftPreset(columns=columns)])
    snapshot = report.run(
        reference_data=reference_frame,
        current_data=current_frame,
        name="Latest pet breed inference batch drift",
    )
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_REPORT_PATH))


def write_scorecard(
    reference: ExtractedBatch,
    calibration_scores: list[dict[str, float]],
    thresholds: dict[str, float],
    clean_evaluation_scores: list[dict[str, float]],
    clean_evaluation_image_count: int,
    scenarios: list[dict],
    latest_batch: ExtractedBatch,
) -> None:
    clean_summary = summarize_detections(clean_evaluation_scores, thresholds)
    clean_batch_mean_scores = mean_scores(clean_evaluation_scores)
    clean_batch_alerts = batch_alerts(clean_batch_mean_scores, thresholds)
    caught = {
        detector: sum(
            scenario["detectors"][detector]["detected"]
            for scenario in scenarios
        )
        for detector in DETECTORS
    }
    clean_false_alarms = {
        detector: {
            "flagged_batches": clean_summary[detector]["flagged_batches"],
            "batch_count": clean_summary[detector]["batch_count"],
            "false_alarm_rate": (
                clean_summary[detector]["flagged_batches"]
                / clean_summary[detector]["batch_count"]
            ),
        }
        for detector in DETECTORS
    }
    report = {
        "reference_examples": int(reference.confidences.size),
        "reference_samples_per_class": SAMPLES_PER_CLASS,
        "clean_threshold_calibration_batches": len(calibration_scores),
        "clean_test_evaluation_batches": len(clean_evaluation_scores),
        "corruption_examples_per_scenario": (
            scenarios[0]["image_count"] if scenarios else 0
        ),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "thresholds": thresholds,
        "clean_false_alarms": clean_false_alarms,
        "clean_evaluation_batch": {
            "image_count": clean_evaluation_image_count,
            "mean_scores": clean_batch_mean_scores,
            "drift_alerts": clean_batch_alerts,
        },
        "corruption_scenarios": scenarios,
        "caught_scenarios": caught,
        "batch_caught_scenarios": {
            detector: sum(
                scenario["batch_alerts"][detector]
                for scenario in scenarios
            )
            for detector in DETECTORS
        },
        "total_corruption_scenarios": len(scenarios),
        "scorecard_unit": (
            "Sensitivity and clean false-alarm rates are reported per "
            f"{BATCH_SIZE}-image inference microbatch; whole-batch alerts "
            "aggregate the mean score across all microbatches."
        ),
        "evidently_report": str(DRIFT_REPORT_PATH.relative_to(PROJECT_ROOT)),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_monitoring_state(
        {
            "embedding_mmd": float(
                scenarios[-1]["batch_mean_scores"]["embedding_mmd"]
            ),
            "pixel_drift_score": float(
                scenarios[-1]["batch_mean_scores"]["pixel_statistics"]
            ),
            "confidence_psi": float(
                scenarios[-1]["batch_mean_scores"]["confidence_psi"]
            ),
            "drift_alert": any(
                scenarios[-1]["batch_alerts"][detector]
                for detector in DETECTORS
            ),
            "batch_inference_p95_seconds": float(
                np.percentile(latest_batch.batch_latency_seconds, 95)
            ),
            "timestamp": time.time(),
        }
    )

    corruption_names = sorted({scenario["corruption"] for scenario in scenarios})
    locations = np.arange(len(corruption_names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 6))
    for offset, detector in enumerate(DETECTORS):
        rates = []
        for corruption in corruption_names:
            matching = [
                scenario
                for scenario in scenarios
                if scenario["corruption"] == corruption
            ]
            rates.append(
                sum(
                    scenario["detectors"][detector]["detected"]
                    for scenario in matching
                )
                / len(matching)
            )
        axis.bar(
            locations + (offset - 1) * width,
            rates,
            width,
            label=detector,
        )
    axis.set(
        title="Corruption detector sensitivity by corruption type",
        ylabel="Fraction of severities detected",
        ylim=(0, 1),
        xticks=locations,
        xticklabels=corruption_names,
    )
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "drift_detector_scorecard.png", dpi=160)
    plt.close(figure)


def evaluate_corruption_suite(device: torch.device) -> dict:
    model = load_model(device)
    temperature = load_temperature()
    reference = extract_batch(
        model,
        sample_split("train"),
        device,
        temperature,
    )
    clean_calibration = extract_batch(
        model,
        sample_split("val", samples_per_class=None),
        device,
        temperature,
    )
    clean_test = sample_split("test", samples_per_class=None)
    clean_evaluation = extract_batch(
        model,
        clean_test,
        device,
        temperature,
    )
    thresholds, calibration_scores = threshold_map(reference, clean_calibration)
    clean_evaluation_scores = score_group(reference, clean_evaluation)

    scenarios = []
    latest_batch = None
    for corruption in sorted(CORRUPTIONS):
        for severity in range(1, 4):
            records = corruption_records(clean_test, corruption, severity)
            latest_batch = extract_batch(
                model,
                records,
                device,
                temperature,
            )
            scores = score_group(reference, latest_batch)
            scenarios.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "image_count": len(records),
                    "batch_count": len(scores),
                    "detectors": summarize_detections(scores, thresholds),
                    "batch_mean_scores": mean_scores(scores),
                    "batch_alerts": batch_alerts(
                        mean_scores(scores),
                        thresholds,
                    ),
                    "latest_scores": scores[-1],
                }
            )
    if latest_batch is None:
        raise RuntimeError("No corruption scenarios were evaluated.")
    write_evidently_report(reference, latest_batch)
    write_scorecard(
        reference,
        calibration_scores,
        thresholds,
        clean_evaluation_scores,
        len(clean_test),
        scenarios,
        latest_batch,
    )
    return json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))


def monitor_current_batch(
    image_paths: list[Path],
    batch_id: str,
    device: torch.device,
    *,
    trigger_airflow: bool,
) -> dict:
    if len(image_paths) < BATCH_SIZE:
        raise ValueError(
            f"Drift monitoring requires at least {BATCH_SIZE} images per batch."
        )
    model = load_model(device)
    temperature = load_temperature()
    reference = extract_batch(
        model,
        sample_split("train"),
        device,
        temperature,
    )
    calibration = extract_batch(
        model,
        sample_split("val", samples_per_class=None),
        device,
        temperature,
    )
    thresholds, _ = threshold_map(reference, calibration)
    records = [ImageRecord(path=path) for path in image_paths]
    current = extract_batch(model, records, device, temperature)
    microbatch_scores = score_group(reference, current)
    scores = [
        {
            detector: float(
                np.mean([score[detector] for score in microbatch_scores])
            )
            for detector in DETECTORS
        }
    ]
    detector_summary = summarize_detections(scores, thresholds)
    latest = scores[-1]
    alert = any(detector_summary[detector]["detected"] for detector in DETECTORS)
    state = {
        "embedding_mmd": latest["embedding_mmd"],
        "pixel_drift_score": latest["pixel_statistics"],
        "confidence_psi": latest["confidence_psi"],
        "drift_alert": alert,
        "batch_inference_p95_seconds": float(
            np.percentile(current.batch_latency_seconds, 95)
        ),
        "timestamp": time.time(),
    }
    save_monitoring_state(state)
    write_evidently_report(reference, current)
    results = {
        "batch_id": batch_id,
        "thresholds": thresholds,
        "latest_scores": latest,
        "detectors": detector_summary,
        "alert": alert,
    }
    if alert and trigger_airflow:
        results["airflow_dag_run"] = trigger_retraining_dag(
            batch_id,
            results,
        )
    return results


def load_batch_paths(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = payload.get("image_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Batch path file must contain a non-empty image_paths list.")
    resolved = []
    for item in paths:
        image_path = Path(item)
        resolved.append(
            image_path if image_path.is_absolute() else PROJECT_ROOT / image_path
        )
    missing = [str(image_path) for image_path in resolved if not image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Batch image paths are missing: {missing[:5]}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-paths",
        type=Path,
        help="JSON file with image_paths for one live batch (runs alert trigger).",
    )
    parser.add_argument("--batch-id", default="batch-score")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Drift feature monitoring requires CUDA.")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    if args.batch_paths is None:
        scorecard = evaluate_corruption_suite(device)
        print(
            f"Scored {scorecard['total_corruption_scenarios']} corruption "
            f"scenarios; scorecard={SCORECARD_PATH}"
        )
    else:
        results = monitor_current_batch(
            load_batch_paths(args.batch_paths),
            args.batch_id,
            device,
            trigger_airflow=True,
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
