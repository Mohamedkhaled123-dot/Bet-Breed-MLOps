import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nginx_canary_routes_five_percent_to_candidate() -> None:
    config = (PROJECT_ROOT / "nginx" / "nginx.conf").read_text(encoding="utf-8")
    weights = {
        int(weight): int(port)
        for port, weight in re.findall(
            r"127\.0\.0\.1:(\d+)\s+weight=(\d+)",
            config,
        )
    }

    assert weights == {95: 8000, 5: 8001}
    assert "proxy_pass http://pet_breed_canary;" in config


def test_grafana_dashboard_has_latency_and_drift_panels() -> None:
    dashboard = json.loads(
        (
            PROJECT_ROOT
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "pet-breed.json"
        ).read_text(encoding="utf-8")
    )
    titles = {panel["title"] for panel in dashboard["panels"]}

    assert len(dashboard["panels"]) >= 3
    assert "Prediction latency p95 (ms)" in titles
    assert "Embedding drift (RBF MMD)" in titles
    assert "Confidence-distribution drift (PSI)" in titles
