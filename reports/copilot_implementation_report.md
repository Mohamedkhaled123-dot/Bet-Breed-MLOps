# Handbook implementation report

## Phase 0 — API, health, uploads, and BentoML

**Status: complete.**

- The FastAPI health endpoint returns `status`, `model_version`, and the loaded class count.
- Prediction keeps the existing TensorRT FP16 production engine and evaluation transform. The response includes breed/species, calibrated top-3 scores, confidence, threshold, abstention decision, class ID, and model version. Low-confidence predictions abstain with `breed` and `species` set to `null`.
- Uploads are limited to 10 MiB, fully decoded before inference, and invalid image payloads return 422. Four-channel PNGs are converted to RGB.
- The ONNX artifact can be registered in the BentoML model store with a content hash and a batchable ONNX `run` signature. The Bento service calls this Runner; `scripts/register_bento_model.py` refreshes or reuses the registered artifact.
- The Bento service was started on `127.0.0.1:3001`; `/readyz` returned 200 and `/predict` returned 200 for a real test image. Its measured response for `Abyssinian_2.jpg` was Abyssinian, cat, confidence 0.9327, with three returned breeds. This is a smoke test, not an accuracy estimate.
- A direct TensorRT call and FastAPI request both ran on the NVIDIA RTX 5050 Laptop GPU with PyTorch 2.14.0+cu130 / CUDA 13.0 and the existing TensorRT 11.3.0.99 engine. A synthetic low-confidence sample returned `decision="uncertain"` and abstained rather than returning a breed.
- Validation: `pytest tests/test_api.py tests/test_bento_service.py -q` — **7 passed**; Python compilation and Pylance diagnostics passed. The earlier focused run had one incorrect test expectation about BentoML's normalized batch dimension; it was corrected to `(0, 0)` and the rerun passed.
- Environment note: installed BentoML is 1.4.39. Its ONNX model adapter, `Runner`, legacy `Service`, and `bentoml.io` emit deprecation warnings. They are used here because the handbook explicitly requires the Runner pattern; migration to the modern BentoML service API remains a follow-up compatibility concern.

## Phase 1 — Calibration and abstention

**Status: complete.**

- Recalibrated the committed `artifacts/checkpoints/resnet18.pt` on CUDA using the 736-image validation split and saved the temperature parameter.
- Measured temperature: **0.859533**. Validation ECE improved from **0.102520** to **0.063133**.
- Created [calibration.png](./calibration.png), a 15-bin reliability diagram showing validation reliability before and after temperature scaling. Machine-readable measurements are in [calibration_metrics.json](./calibration_metrics.json).
- Evaluated ten candidate thresholds. The selection rule is maximum coverage subject to validation selective accuracy ≥ **0.98**. Threshold **0.80** is the best listed candidate: **498/736 answered (67.66% coverage)** with **98.80% selective accuracy**. At 0.75, accuracy was 97.91%, so it did not meet the floor. The selected threshold is now the shared inference default; low-confidence predictions abstain.
- The temperature and threshold use the same validation split; these are validation estimates, not an independent test-set claim. The report states the selection rule and records the full threshold curve in `reports/confidence_evaluation.json`.
- Validation: calibration, confidence, API, BentoML, and manifest tests — **23 passed**; Pylance diagnostics and `git diff --check` passed. Direct TensorRT inference confirmed the 0.80 threshold and returned a three-breed result on a validation image.
- Updated `.gitignore` to include generated report deliverables rather than silently ignoring `reports/`.

## Phase 2 — MLflow experiments and production registration

**Status: complete.**

- Extended the classifier and training backbone choices to include ResNet-50. Added a CUDA-only experiment runner that performs five real two-epoch, frozen-backbone/head-training runs with ImageNet initialization and distinct logged seeds, parameters, checkpoint artifacts, training loss, elapsed time, and measured validation metrics.
- The six logged runs comprise the five experiments plus a measured re-evaluation of the existing production checkpoint. All use the 736-image validation split:

| Run | Top-1 | Macro-F1 | ECE | Coverage | Selective accuracy | Quality gate |
|---|---:|---:|---:|---:|---:|---|
| ResNet-18, lr 0.001 | 0.8410 | 0.8324 | 0.1867 | 0.5095 | 0.9867 | Fail |
| MobileNetV3-Small, lr 0.001 | 0.8261 | 0.8261 | 0.1834 | 0.4878 | 0.9944 | Fail |
| ResNet-50, lr 0.001 | 0.8832 | 0.8784 | 0.3490 | 0.3886 | 1.0000 | Fail |
| ResNet-18, lr 0.0003 | 0.7432 | 0.7332 | 0.4818 | 0.1454 | 0.9813 | Fail |
| MobileNetV3-Small, lr 0.0003 | 0.7582 | 0.7519 | 0.4824 | 0.1698 | 0.9760 | Fail |
| Existing ResNet-18 production checkpoint | **0.8967** | **0.8975** | **0.1025** | **0.6766** | **0.9880** | **Pass** |

- The quality gate is candidate top-1 ≥ measured production top-1 − 0.01 (floor **0.8867**). None of the newly trained two-epoch heads met the floor, so the incumbent remains selected and the original `artifacts/checkpoints/resnet18.pt` remains the production model. ResNet-50 is the measured teacher candidate (0.8832 top-1) for the later distillation phase; this short comparison is not represented as a converged teacher.
- Registered `PetBreedClassifier` and promoted model version **3** to both the `Production` stage and `Production` alias. Its artifact uses MLflow's safe PyTorch PT2 format with tensor signature `(-1, 3, 224, 224) -> (-1, 37)`. Loaded the alias and ran a two-image inference smoke test; output shape was **(2, 37)**. An intermediate pickle-format version 2 remains archived and is not referenced by the Production alias; it was not deleted.
- Generated [mlflow_comparison.png](./mlflow_comparison.png) and [mlflow_comparison.json](./mlflow_comparison.json) from the real MLflow run results. ECE is reported both before and after per-run temperature fitting in the JSON. Coverage/selective accuracy and calibration use the same validation split, so they are validation estimates with calibration-selection optimism risk.
- Validation: Phase 2 test module — **6 passed**; Pylance diagnostics and `git diff --check` passed. ResNet-50 weights were retrieved from torchvision's official model-weight endpoint because they were not cached locally.
- MLflow emitted a deprecation warning for registry stages (the Production alias is also set) and warned that inferred requirements omit PyTorch's `+cu130` local version suffix. Model registration and loading succeeded; dependency-environment portability remains to be validated separately.

## Phase 3 — Optimization matrix

**Status: complete.**

- Ran all optimization comparisons on the same **736-image validation split**. Latency is single-image p95 after 20 warmups and 100 timed iterations; GPU samples synchronize CUDA. CPU rows use CPU inference, so CPU/GPU latency is hardware-specific rather than a direct device-only comparison.

| Technique | Top-1 | P95 | Model size | Hardware |
|---|---:|---:|---:|---|
| ResNet-18 production baseline | 89.67% | 2.636 ms | 42.78 MB | RTX 5050 Laptop GPU |
| Structured filter pruning, selected 2% | 88.99% | 2.429 ms | 42.79 MB | RTX 5050 Laptop GPU |
| Static INT8 PTQ, 148 stratified train images | 89.95% | 24.108 ms | 10.85 MB | Intel CPU |
| INT8 QAT, 2 CUDA epochs | 90.08% | 11.205 ms | 10.87 MB | Intel CPU |
| Supervised MobileNetV3-Small student baseline | 84.38% | 12.732 ms | 6.07 MB | RTX 5050 Laptop GPU |
| ResNet-50 → MobileNetV3-Small distillation | 86.82% | 5.480 ms | 6.07 MB | RTX 5050 Laptop GPU |
| TensorRT FP16 | 89.67% | **0.690 ms** | 21.67 MB | RTX 5050 Laptop GPU |

- Structured pruning was swept at 2%, 5%, 10%, 15%, and 30% per Conv2d layer. The chosen 1.98% filter sparsity is the strongest candidate within the 1.0 percentage-point top-1 drop limit (0.68 pp below baseline). Higher rates reduced top-1 to 84.92%, 66.71%, 51.36%, and 3.53%; the lightweight channel masks do not compact layer dimensions, so model size did not shrink. The small measured p95 change is not claimed as a dependable speedup.
- PTQ used static QDQ quantization with a deterministic four-images-per-class training calibration set. It measured 0.27 pp higher top-1 and about 4× smaller ONNX size, but CPU p95 was slower than the GPU baseline. QAT ran on CUDA, converted to a CPU INT8 graph, measured 0.41 pp higher top-1 and lower ECE (0.0734 vs 0.1025); its CPU latency must not be presented as a like-for-like GPU comparison.
- Distillation starts from the measured supervised MobileNetV3-Small checkpoint and the two-epoch ResNet-50 experiment checkpoint. With temperature 3, soft-target weight 0.3, and three CUDA epochs, student top-1 improved from **84.38% to 86.82%** and p95 from 12.732 ms to 5.480 ms on the same GPU, at essentially unchanged 6.07 MB size. It remains 2.85 pp below the ResNet-18 production accuracy; the short teacher training is a limitation.
- TensorRT FP16 matched ResNet-18 validation top-1 at 89.67% and had about **3.8× lower measured p95** than the PyTorch GPU baseline. PyTorch/ONNX FP16/TensorRT parity passed on a validation image: all predicted class 0; max absolute logit difference was 0.00973 between PyTorch and TensorRT. The runtime engine validation also passed with finite FP16 logits of shape `(1, 37)`.
- Persisted the measured table and full pruning sweep in [optimization_benchmark.json](./optimization_benchmark.json). Added DVC pointers for `model_int8.onnx`, optimization outputs, the ResNet-50 teacher, the unchanged ResNet-18 production checkpoint, its MobileNet student baseline, ONNX source artifacts, and the TensorRT engine; pushed those targeted artifacts to the configured local `D:\DVC_Remote`. DVC reports the cache and remote in sync. No Docker image was pulled or built.
- Validation: optimization and MLflow/model tests — **10 passed**; TensorRT engine and parity scripts passed; Pylance diagnostics and `git diff --check` passed. PyTorch 2.14 emits deprecation warnings for FX QAT and TorchScript serialization; these APIs remain functional in the pinned environment and are noted for migration.

## Phase 4 — Drift monitoring, metrics, and retraining wiring

**Status: implementation and full-data scorecard complete; external service execution remains unavailable.**

- Implemented CUDA feature extraction from the unchanged production checkpoint, RBF MMD over pooled ResNet-18 embeddings, brightness/contrast/sharpness drift, and calibrated-confidence PSI. Thresholds use the 95th percentile of clean validation microbatches. Reference embeddings use four stratified train images per class (148 total).
- Evaluated all **66,042 generated test corruptions**: 18 combinations of six corruption types and three severities, with all 3,669 test images in every scenario. Clean threshold calibration used 736 validation images (23 batches of 32); the independent clean test set used 3,669 images (115 batches).
- Measured thresholds: embedding MMD **0.01803**, pixel-statistics score **0.39451**, confidence PSI **2.19881**. Per-32-image clean false-alarm rates were MMD **2/115 (1.74%)**, pixel statistics **14/115 (12.17%)**, and confidence PSI **9/115 (7.83%)**. These microbatch rates are reported rather than hidden; pixel/PSI rates remain high enough to warrant monitoring before production alerting.
- Whole-batch scoring aggregates microbatch scores before alerting. The independent 3,669-image clean test batch triggered **no detector alert**. Across the 18 corrupted full batches, aggregate MMD caught **16/18**, pixel statistics **15/18**, and confidence PSI **3/18**; at least one detector caught **18/18**. Individual microbatch sensitivity was 18/18 for each detector. Calibration and test false-alarm estimates are data-dependent and should be recalibrated on production-like batches.
- Generated [drift_detector_scorecard.json](./drift_detector_scorecard.json), [drift_detector_scorecard.png](./drift_detector_scorecard.png), and the full [Evidently HTML report](./evidently/latest_batch_drift.html). Batch-scoring output is directed to `data/scoring/output/`; each 1,000-image scoring run invokes the drift monitor and writes the latest state used by `/metrics`.
- Added `/metrics` with request count/latency, rolling prediction p95, latest batch p95, drift gauges, confidence PSI, and alert state. API tests verify the Prometheus exposition and inference timing path.
- Added the Airflow REST trigger client and `pet_breed_drift_retraining` DAG. The DAG performs CUDA fine-tuning on a deterministic mixed corruption distribution, evaluates against the current Production checkpoint, and promotes through MLflow only when the production-minus-0.01 top-1 gate passes. **No actual Airflow DAG run is claimed:** localhost:8080 is not accepting connections, and Airflow is not installed in this Windows environment. A real drift alert fails explicitly if `AIRFLOW_API_URL` is missing or the API rejects the run.
- Added Prometheus alert/scrape configuration and a four-panel Grafana dashboard definition. The local Prometheus (9090) and Grafana (3000) ports are unavailable, so no live dashboard or screenshot is claimed. Added a 95%-stable/5%-candidate Nginx config and rollout stages; Nginx is not installed here, so `nginx -t` could not be run.
- Added GitHub Actions lint/test/quality-gate/build-and-push workflow. The workflow requires the `DVC_REMOTE_URL` repository secret because the configured local DVC remote (`D:\DVC_Remote`) is not accessible to hosted runners. No GitHub Actions run or image build was started.
- Production quality gate measured **0.896739 top-1** against the same validation baseline (floor **0.886739**) and passed. CPU ONNX Runtime p95 was **6.788 ms**; OpenVINO CPU p95 was **10.137 ms**, each after 20 warmups and 100 timed single-image inferences. These runtime measurements are on the Intel CPU and are not comparable to the GPU measurements in Phase 3.
- PyTorch/ONNX parity passed on 200 test images with maximum absolute logit difference **1.1444e-5 CPU-to-CPU** and **3.1471e-5 CUDA-to-CUDA**, both under 1e-4. CUDA parity required disabling TF32 in PyTorch and ONNX Runtime; the cross-device GPU-PyTorch/CPU-ONNX comparison is recorded separately and is not used as the parity gate.
- Validation: full `pytest -q` — **56 passed**; focused Ruff checks, Python compilation, YAML/dashboard JSON parsing, `uv lock --check`, and `git diff --check` passed. No Docker images were pulled and no Docker build was run.

## Remaining phases

Continue with live serving and Locust measurements, the 1,000-image batch-scoring run, and final deployment checks. At the first Triton GPU/runtime/Windows/Docker compatibility blocker, stop and report the exact error, attempts, likely cause, and safest recommendation for user decision.
