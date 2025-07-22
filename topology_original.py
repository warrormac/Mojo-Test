import numpy as np
from scipy.spatial import distance
import time

# Safely import CuPy for GPU operations
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy (GPU) is available.")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy (GPU) not found. Falling back to CPU.")

# --- Hardcoded constants ---
NO_SAMPLING_THRESHOLD = 500
SAMPLING_PERCENTAGE = 0.35
MAX_SAMPLE_SIZE_CAP = 1000
# ---

def get_dynamic_sample_size(num_chunks: int) -> int:
    if num_chunks <= NO_SAMPLING_THRESHOLD:
        print(f"INFO: (Python) Document has {num_chunks} chunks. Running on all chunks.")
        return num_chunks
    else:
        calculated_sample_size = int(num_chunks * SAMPLING_PERCENTAGE)
        final_sample_size = min(calculated_sample_size, MAX_SAMPLE_SIZE_CAP)
        print(f"INFO: (Python) Document has {num_chunks} chunks. Running on a sample of {final_sample_size}.")
        return final_sample_size

# --- CPU IMPLEMENTATION ---
def test_fiber_bundle_hypothesis_cpu(embeddings: np.ndarray, r_min=0.01, r_max=20.0, n_r=50, window_size=10, change_threshold=1.5):
    num_embeddings = embeddings.shape[0]
    dist_matrix = distance.cdist(embeddings, embeddings, 'euclidean')
    r_values = np.linspace(r_min, r_max, n_r)
    log_r = np.log(r_values)
    rejections = []
    for i in range(num_embeddings):
        distances_from_i = dist_matrix[i]
        nx_r = np.sum(distances_from_i[:, np.newaxis] <= r_values, axis=0)
        log_nx_r = np.log(nx_r + 1e-10)
        slopes = np.gradient(log_nx_r, log_r)
        is_rejected = False
        for j in range(window_size, len(slopes) - window_size):
            before = slopes[j - window_size : j]
            after = slopes[j : j + window_size]
            if np.mean(after) > np.mean(before) and (np.mean(after) - np.mean(before)) > change_threshold:
                is_rejected = True
                break
        rejections.append(is_rejected)
    return rejections

# --- GPU IMPLEMENTATION (from your stable version) ---
if CUPY_AVAILABLE:
    def pairwise_distances_gpu(embeddings_gpu: cp.ndarray, batch_size=200) -> cp.ndarray:
        n = embeddings_gpu.shape[0]
        dist_matrix = cp.empty((n, n), dtype=cp.float32)
        for i in range(0, n, batch_size):
            end_index = min(i + batch_size, n)
            batch = embeddings_gpu[i:end_index]
            dist_slice = cp.linalg.norm(batch[:, None, :] - embeddings_gpu[None, :, :], axis=-1)
            dist_matrix[i:end_index, :] = dist_slice
        return dist_matrix

    def analyze_slopes_on_gpu(slopes_gpu: cp.ndarray, window_size: int = 10, change_threshold: float = 1.5) -> cp.ndarray:
        padded_slopes = cp.pad(slopes_gpu, ((0, 0), (window_size - 1, 0)), 'edge')
        cumsum_slopes = padded_slopes.cumsum(axis=1)
        rolling_sums = cumsum_slopes[:, window_size:] - cumsum_slopes[:, :-window_size]
        rolling_means = rolling_sums / window_size
        after_windows = rolling_means[:, window_size:]
        before_windows = rolling_means[:, :-window_size]
        slope_increase = after_windows - before_windows
        max_increase_per_chunk = slope_increase.max(axis=1)
        return max_increase_per_chunk > change_threshold

    def test_fiber_bundle_hypothesis_gpu(embeddings_gpu: cp.ndarray, r_min=0.01, r_max=20.0, n_r=50, window_size=10, change_threshold=1.5):
        r_values_gpu = cp.linspace(r_min, r_max, n_r)
        log_r_gpu = cp.log(r_values_gpu)
        dist_matrix = pairwise_distances_gpu(embeddings_gpu)
        nx_r_matrix = cp.sum(dist_matrix[..., None] <= r_values_gpu, axis=1)
        log_nx_r_matrix = cp.log(nx_r_matrix + 1e-10)
        slopes_matrix = cp.gradient(log_nx_r_matrix, log_r_gpu, axis=-1)
        rejections_gpu = analyze_slopes_on_gpu(slopes_matrix, window_size, change_threshold)
        return cp.asnumpy(rejections_gpu).tolist()
