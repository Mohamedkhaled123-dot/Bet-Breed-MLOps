FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-production.txt ./

RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch torchvision \
    && python -m pip install --no-cache-dir -r requirements-production.txt

COPY src ./src
COPY artifacts/models/resnet18.onnx ./artifacts/models/resnet18.onnx
COPY artifacts/models/resnet18.onnx.data ./artifacts/models/resnet18.onnx.data
COPY artifacts/calibration ./artifacts/calibration
COPY data/metadata ./data/metadata

EXPOSE 8000

CMD ["uvicorn", "bet_breed_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
