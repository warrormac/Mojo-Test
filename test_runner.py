import numpy as np
import time
from topology_original import (
    get_dynamic_sample_size, 
    test_fiber_bundle_hypothesis_cpu, 
    CUPY_AVAILABLE
)

# --- GPU Activator ---
# Set this to True to use the GPU if available, or False to force CPU execution.
USE_GPU = True
# ---------------------

# Conditionally import GPU functions if available
if CUPY_AVAILABLE:
    import cupy as cp
    from topology_original import test_fiber_bundle_hypothesis_gpu

if __name__ == "__main__":
    print("Loading embeddings from embeddings.npy...")
    try:
        embeddings = np.load("embeddings.npy").astype(np.float32)
    except FileNotFoundError:
        print("ERROR: embeddings.npy not found.")
        exit()

    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

    sample_size = get_dynamic_sample_size(embeddings.shape[0])
    indices_to_test = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sampled_embeddings_cpu = embeddings[indices_to_test]

    print("--------------------------------------------------")

    # --- Decide whether to run on GPU or CPU based on the flag ---
    if USE_GPU and CUPY_AVAILABLE:
        print("🚀 Starting Python analysis on GPU...")
        # Move data to the GPU
        sampled_embeddings_gpu = cp.asarray(sampled_embeddings_cpu)
        
        start_time = time.perf_counter()
        result = test_fiber_bundle_hypothesis_gpu(sampled_embeddings_gpu)
        end_time = time.perf_counter()
        
        print("Python GPU analysis complete.")
        
    else:
        if USE_GPU and not CUPY_AVAILABLE:
            print("⚠️ GPU was requested, but CuPy is not available. Falling back to CPU.")
        print("🧠 Starting Python analysis on CPU...")
        
        start_time = time.perf_counter()
        result = test_fiber_bundle_hypothesis_cpu(sampled_embeddings_cpu)
        end_time = time.perf_counter()
        
        print("Python CPU analysis complete.")

    print("--------------------------------------------------")

    rejection_count = sum(1 for rejected in result if rejected)
    
    # Label the output clearly
    backend = "GPU" if (USE_GPU and CUPY_AVAILABLE) else "CPU"
    print(f"Python ({backend}) Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Total rejections in sample: {rejection_count}")
