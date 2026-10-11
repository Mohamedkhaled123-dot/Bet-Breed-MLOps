import json

import numpy as np
import pytest
import torch

from bet_breed_mlops.airflow_trigger import trigger_retraining_dag
from bet_breed_mlops.monitoring import (
    calibrate_drift_threshold,
    confidence_psi,
    pixel_drift_score,
    rbf_mmd_squared,
)
from bet_breed_mlops.prometheus_metrics import (
    render_prometheus_metrics,
    save_monitoring_state,
)
from scripts.monitor_corruptions import monitor_current_batch


def test_mmd_is_small_for_identical_samples_and_larger_for_shift() -> None:
    reference = np.arange(60, dtype=np.float64).reshape(20, 3)
    shifted = reference + 30

    assert abs(rbf_mmd_squared(reference, reference)) < 1e-9
    assert rbf_mmd_squared(reference, shifted) > 0.3


def test_pixel_drift_uses_brightness_contrast_and_sharpness() -> None:
    reference = np.ones((8, 3), dtype=np.float64)
    current = reference.copy()
    current[:, 0] += 2.0

    assert pixel_drift_score(reference, current) > 1_000_000


def test_confidence_psi_is_zero_for_same_distribution() -> None:
    values = np.linspace(0.1, 0.99, 100)

    assert confidence_psi(values, values) == pytest.approx(0.0)
    assert confidence_psi(values, 1.0 - values) > 0


def test_clean_threshold_rejects_invalid_scores() -> None:
    assert calibrate_drift_threshold(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.9)
    with pytest.raises(ValueError, match="finite"):
        calibrate_drift_threshold(np.array([np.nan]))


def test_prometheus_metrics_export_latest_drift_state(
    tmp_path,
    monkeypatch,
) -> None:
    from bet_breed_mlops import prometheus_metrics

    monkeypatch.setattr(
        prometheus_metrics,
        "MONITORING_STATE_PATH",
        tmp_path / "latest_batch.json",
    )
    save_monitoring_state(
        {
            "embedding_mmd": 0.12,
            "pixel_drift_score": 2.5,
            "confidence_psi": 0.08,
            "drift_alert": True,
            "batch_inference_p95_seconds": 0.004,
            "timestamp": 1_800_000_000.0,
        }
    )
    body, content_type = render_prometheus_metrics()

    assert "text/plain" in content_type
    assert b"pet_breed_embedding_mmd 0.12" in body
    assert b"pet_breed_confidence_drift_psi 0.08" in body
    assert b"pet_breed_drift_alert 1.0" in body
    assert b"pet_breed_batch_inference_p95_seconds 0.004" in body


def test_airflow_trigger_requires_api_url() -> None:
    with pytest.raises(RuntimeError, match="AIRFLOW_API_URL"):
        trigger_retraining_dag(
            "batch-001",
            {"embedding_mmd": 0.3},
            airflow_url="",
        )


def test_airflow_trigger_posts_dag_run_with_drift_context(monkeypatch) -> None:
    from bet_breed_mlops import airflow_trigger

    captured = {}

    class FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"dag_run_id":"accepted"}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(airflow_trigger, "urlopen", fake_urlopen)
    result = trigger_retraining_dag(
        "validation batch/001",
        {"embedding_mmd": 0.12},
        airflow_url="http://airflow.test/",
        username="test-user",
        password="test-password",
    )

    assert result == {"dag_run_id": "accepted"}
    assert captured["url"] == (
        "http://airflow.test/api/v1/dags/"
        "pet_breed_drift_retraining/dagRuns"
    )
    assert captured["payload"]["conf"] == {
        "batch_id": "validation batch/001",
        "drift_results": {"embedding_mmd": 0.12},
    }
    assert captured["payload"]["dag_run_id"].startswith("drift__validation-batch-001__")
    assert captured["timeout"] == 10


def test_airflow_trigger_rejects_partial_credentials() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        trigger_retraining_dag(
            "batch-001",
            {},
            airflow_url="http://airflow.test",
            username="test-user",
        )


def test_live_drift_monitor_rejects_too_small_batches() -> None:
    with pytest.raises(ValueError, match="at least 32 images"):
        monitor_current_batch(
            [],
            "small-batch",
            device=torch.device("cpu"),
            trigger_airflow=False,
        )
