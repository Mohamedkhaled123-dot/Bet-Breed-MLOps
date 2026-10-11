import base64
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

AIRFLOW_DAG_ID = "pet_breed_drift_retraining"


def trigger_retraining_dag(
    batch_id: str,
    drift_results: dict,
    *,
    airflow_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict:
    base_url = (airflow_url or os.getenv("AIRFLOW_API_URL", "")).rstrip("/")
    if not base_url:
        raise RuntimeError(
            "Drift alert cannot trigger retraining: AIRFLOW_API_URL is not configured."
        )
    username = username if username is not None else os.getenv("AIRFLOW_USERNAME")
    password = password if password is not None else os.getenv("AIRFLOW_PASSWORD")
    if (username is None) != (password is None):
        raise ValueError("AIRFLOW_USERNAME and AIRFLOW_PASSWORD must be set together.")

    normalized_batch_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", batch_id).strip("-")
    if not normalized_batch_id:
        raise ValueError("batch_id must contain at least one safe identifier character.")
    dag_run_id = f"drift__{normalized_batch_id[:80]}__{time.time_ns()}"
    payload = {
        "dag_run_id": dag_run_id,
        "conf": {
            "batch_id": batch_id,
            "drift_results": drift_results,
        },
    }
    headers = {"Content-Type": "application/json"}
    if username is not None and password is not None:
        token = base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    request = Request(
        (
            f"{base_url}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
        ),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status not in (200, 201):
                raise RuntimeError(
                    f"Airflow returned unexpected HTTP status {response.status}."
                )
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Airflow rejected drift DAG trigger with HTTP {error.code}: "
            f"{response_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach Airflow API at {base_url}: {error.reason}"
        ) from error
    return result
