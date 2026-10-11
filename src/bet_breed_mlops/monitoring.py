import numpy as np


def rbf_mmd_squared(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if (
        reference.ndim != 2
        or current.ndim != 2
        or reference.shape[1] != current.shape[1]
    ):
        raise ValueError("MMD inputs must be 2D arrays with matching features.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("MMD inputs must contain only finite values.")
    if reference.shape[0] < 2 or current.shape[0] < 2:
        raise ValueError("MMD requires at least two examples in each sample.")

    reference_norms = np.sum(reference * reference, axis=1, keepdims=True)
    reference_distances = np.maximum(
        reference_norms + reference_norms.T - 2 * reference @ reference.T,
        0.0,
    )
    nonzero_distances = reference_distances[
        np.triu_indices(reference.shape[0], k=1)
    ]
    positive_distances = nonzero_distances[nonzero_distances > 0]
    if positive_distances.size == 0:
        raise ValueError("Reference embeddings have no nonzero pairwise distances.")
    gamma = 1.0 / (2.0 * float(np.median(positive_distances)))

    current_norms = np.sum(current * current, axis=1, keepdims=True)
    current_distances = np.maximum(
        current_norms + current_norms.T - 2 * current @ current.T,
        0.0,
    )
    cross_distances = np.maximum(
        reference_norms + current_norms.T - 2 * reference @ current.T,
        0.0,
    )
    reference_kernel = np.exp(-gamma * reference_distances)
    current_kernel = np.exp(-gamma * current_distances)
    cross_kernel = np.exp(-gamma * cross_distances)
    return float(
        reference_kernel.mean()
        + current_kernel.mean()
        - 2.0 * cross_kernel.mean()
    )


def pixel_drift_score(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if (
        reference.ndim != 2
        or current.ndim != 2
        or reference.shape[1] != 3
        or current.shape[1] != 3
    ):
        raise ValueError("Pixel statistics must be [examples, brightness/contrast/sharpness].")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Pixel statistics must contain only finite values.")
    if reference.shape[0] < 2 or current.shape[0] == 0:
        raise ValueError("Pixel drift needs at least two reference and one current image.")

    reference_mean = reference.mean(axis=0)
    reference_std = reference.std(axis=0, ddof=1)
    scale = np.maximum(reference_std, 1e-6)
    standardized_shift = np.abs(current.mean(axis=0) - reference_mean) / scale
    return float(standardized_shift.max())


def confidence_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    if reference.size == 0 or current.size == 0 or bins <= 1:
        raise ValueError("PSI requires non-empty confidence arrays and at least two bins.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Confidence values must be finite.")
    if (
        np.any(reference < 0)
        or np.any(reference > 1)
        or np.any(current < 0)
        or np.any(current > 1)
    ):
        raise ValueError("Confidence values must be within [0, 1].")

    edges = np.linspace(0.0, 1.0, bins + 1)
    reference_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    current_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * bins
    )
    current_share = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * bins
    )
    return float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )


def calibrate_drift_threshold(
    clean_scores: np.ndarray,
    quantile: float = 0.95,
) -> float:
    clean_scores = np.asarray(clean_scores, dtype=np.float64).reshape(-1)
    if clean_scores.size == 0:
        raise ValueError("At least one clean calibration score is required.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("Threshold quantile must be between zero and one.")
    if not np.isfinite(clean_scores).all():
        raise ValueError("Clean calibration scores must be finite.")
    return float(np.quantile(clean_scores, quantile))
