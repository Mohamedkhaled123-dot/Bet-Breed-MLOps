FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./

COPY src ./src

RUN pip install --no-cache-dir uv

RUN uv sync --frozen --no-dev

COPY artifacts/models ./artifacts/models
COPY artifacts/calibration ./artifacts/calibration
COPY data/metadata ./data/metadata

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "bet_breed_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]