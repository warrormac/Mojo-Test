from topology import MatrixFloat32, test_fiber_bundle_hypothesis, get_dynamic_sample_size, create_sampled_embeddings
from python import Python, PythonObject

fn simple_rand(seed: Int) -> Float32:
    var hash = (seed * 9301 + 49297) % 233280
    return Float32(hash) / 233280.0

fn rand_float(min: Float32, max: Float32, seed: Int) -> (Float32, Int):
    var new_seed = (seed * 9301 + 49297) % 233280
    var value = min + (max - min) * Float32(new_seed) / 233280.0
    return (value, new_seed)

fn main() raises:
    # --- Use Python interop to load the .npy file ---
    print("Loading embeddings from embeddings.npy...")
    
    var np: PythonObject = Python.import_module("numpy")
    var embeddings_np: PythonObject
    try:
        embeddings_np = np.load("embeddings.npy").astype(np.float32)
    except:
        print("ERROR: embeddings.npy not found. Please make sure it's in the mojo_test directory.")
        return

    var num_embeddings = embeddings_np.shape[0]
    var dim = embeddings_np.shape[1]
    print("Loaded", Int(num_embeddings), "embeddings with dimension", Int(dim), ".")

    # --- Copy data from Python (NumPy) to Mojo (MatrixFloat32) ---
    var embeddings = MatrixFloat32(Int(num_embeddings), Int(dim))
    for i in range(Int(num_embeddings)):
        for j in range(Int(dim)):
            embeddings[i, j] = Float32(embeddings_np[i, j])


    # --- Run the test with dynamic sampling ---
    var sample_size = get_dynamic_sample_size(embeddings.rows)
    var sampled_embeddings = create_sampled_embeddings(embeddings, sample_size)

    print("--------------------------------------------------")
    print("Starting Mojo analysis on the PDF embeddings...")

    # --- Time the execution (Python Interop) ---
    var time = Python.import_module("time")
    var start_time = time.time()

    var result = test_fiber_bundle_hypothesis(sampled_embeddings)

    var end_time = time.time()

    print("Mojo analysis complete.")
    print("--------------------------------------------------")

    var rejection_count = 0
    for i in range(len(result)):
        if result[i]:
            rejection_count += 1

    # --- Print the results ---
    var elapsed_ms = (end_time - start_time) * 1000.0
    print("Mojo Time:", elapsed_ms, "ms")
    print("Total rejections in sample:", rejection_count)