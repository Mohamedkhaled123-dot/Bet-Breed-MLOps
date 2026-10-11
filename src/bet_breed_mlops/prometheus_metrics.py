import json
import math
import threading
from collections import deque
from pathlib import Path

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_STATE_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "latest_batch.json"
)
METRICS_REGISTRY = CollectorRegistry()

PREDICTION_REQUESTS = Counter(
    "pet_breed_prediction_requests_total",
    "Number of image prediction requests.",
    registry=METRICS_REGISTRY,
)
PREDICTION_LATENCY = Histogram(
    "pet_breed_prediction_latency_seconds",
    "Image prediction request latency.",
    registry=METRICS_REGISTRY,
)
PREDICTION_P95 = Gauge(
    "pet_breed_prediction_p95_seconds",
    "Rolling p95 image prediction latency.",
    registry=METRICS_REGISTRY,
)
BATCH_P95 = Gauge(
    "pet_breed_batch_inference_p95_seconds",
    "Measured p95 inference latency for the latest monitored batch.",
    registry=METRICS_REGISTRY,
)
EMBEDDING_MMD = Gauge(
    "pet_breed_embedding_mmd",
    "Latest embedding MMD drift score.",
    registry=METRICS_REGISTRY,
)
PIXEL_DRIFT = Gauge(
    "pet_breed_pixel_drift_score",
    "Latest brightness/contrast/sharpness drift score.",
    registry=METRICS_REGISTRY,
)
CONFIDENCE_PSI = Gauge(
    "pet_breed_confidence_drift_psi",
    "Latest confidence-distribution PSI score, tracked separately from input drift.",
    registry=METRICS_REGISTRY,
)
DRIFT_ALERT = Gauge(
    "pet_breed_drift_alert",
    "Whether the latest monitored batch crossed an alert threshold.",
    registry=METRICS_REGISTRY,
)
LATEST_BATCH_TIMESTAMP = Gauge(
    "pet_breed_latest_batch_timestamp_seconds",
    "Unix timestamp of the latest monitored batch.",
    registry=METRICS_REGISTRY,
)

_latency_lock = threading.Lock()
_latency_samples: deque[float] = deque(maxlen=1000)


def _validate_monitoring_state(state: dict) -> None:
    required = {
        "embedding_mmd",
        "pixel_drift_score",
        "confidence_psi",
        "drift_alert",
        "batch_inference_p95_seconds",
        "timestamp",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"Monitoring state is missing fields: {sorted(missing)}")
    for field in required - {"drift_alert"}:
        value = state[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Monitoring field {field} must be numeric.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Monitoring field {field} must be finite and non-negative.")
    if not isinstance(state["drift_alert"], bool):
        raise ValueError("drift_alert must be a boolean.")


def observe_prediction(seconds: float) -> None:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Prediction latency must be a finite non-negative value.")
    PREDICTION_REQUESTS.inc()
    PREDICTION_LATENCY.observe(seconds)
    with _latency_lock:
        _latency_samples.append(seconds)
        ordered = sorted(_latency_samples)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        PREDICTION_P95.set(ordered[index])


def save_monitoring_state(state: dict) -> None:
    _validate_monitoring_state(state)

    MONITORING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MONITORING_STATE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(MONITORING_STATE_PATH)


def _load_monitoring_state() -> dict | None:
    if not MONITORING_STATE_PATH.exists():
        return None
    state = json.loads(MONITORING_STATE_PATH.read_text(encoding="utf-8"))
    _validate_monitoring_state(state)
    return state


def render_prometheus_metrics() -> tuple[bytes, str]:
    state = _load_monitoring_state()
    if state is not None:
        EMBEDDING_MMD.set(state["embedding_mmd"])
        PIXEL_DRIFT.set(state["pixel_drift_score"])
        CONFIDENCE_PSI.set(state["confidence_psi"])
        DRIFT_ALERT.set(1 if state["drift_alert"] else 0)
        LATEST_BATCH_TIMESTAMP.set(state["timestamp"])
        BATCH_P95.set(state["batch_inference_p95_seconds"])
    return generate_latest(METRICS_REGISTRY), CONTENT_TYPE_LATEST
