import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "retrain_from_drift.py"


def run_retraining_action(action: str, batch_id: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(RETRAIN_SCRIPT),
            action,
            "--batch-id",
            batch_id,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"Retraining action {action!r} produced no JSON result. "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )


@dag(
    dag_id="pet_breed_drift_retraining",
    description=(
        "Fine-tune on a balanced mixed corruption distribution, quality-gate "
        "against Production, then promote or reject."
    ),
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["pet-breed", "drift", "retraining"],
)
def pet_breed_drift_retraining():
    @task
    def validate_drift_alert() -> dict:
        context = get_current_context()
        dag_run = context["dag_run"]
        configuration = dag_run.conf or {}
        if not configuration.get("batch_id"):
            raise ValueError("Drift DAG run requires a batch_id.")
        drift_results = configuration.get("drift_results")
        if not isinstance(drift_results, dict):
            raise ValueError("Drift DAG run requires structured drift_results.")
        if not drift_results.get("alert"):
            raise ValueError("Retraining DAG was triggered without a drift alert.")
        return {
            "batch_id": str(configuration["batch_id"]),
            "drift_results": drift_results,
        }

    @task
    def fine_tune_on_corruptions(context: dict) -> dict:
        return run_retraining_action("train", context["batch_id"])

    @task
    def enforce_production_quality_gate(
        context: dict,
        candidate: dict,
    ) -> dict:
        result = run_retraining_action("evaluate", context["batch_id"])
        if result["batch_id"] != candidate["batch_id"]:
            raise RuntimeError("Evaluation returned a different retraining batch id.")
        return result

    @task
    def promote_or_reject_candidate(
        context: dict,
        evaluation: dict,
    ) -> dict:
        if evaluation["batch_id"] != context["batch_id"]:
            raise RuntimeError("Quality gate returned a different retraining batch id.")
        return run_retraining_action("promote", context["batch_id"])

    alert = validate_drift_alert()
    candidate = fine_tune_on_corruptions(alert)
    evaluation = enforce_production_quality_gate(alert, candidate)
    promote_or_reject_candidate(alert, evaluation)


pet_breed_drift_retraining()
