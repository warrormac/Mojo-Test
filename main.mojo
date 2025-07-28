from topology import MatrixFloat32, test_fiber_bundle_hypothesis, get_dynamic_sample_size, create_sampled_embeddings
from topology_gpu import pairwise_distances_gpu
from python import Python, PythonObject


fn simple_rand(seed: Int) -> Float32:
    var hash = (seed * 9301 + 49297) % 233280
    return Float32(hash) / 233280.0

fn rand_float(min: Float32, max: Float32, seed: Int) -> (Float32, Int):
    var new_seed = (seed * 9301 + 49297) % 233280
    var value = min + (max - min) * Float32(new_seed) / 233280.0
    return (value, new_seed)

fn main() raises:
    print("Loading embeddings from embeddings.npy...")

    var np: PythonObject = Python.import_module("numpy")
    var embeddings_np: PythonObject
    try:
        embeddings_np = np.load("embeddings.npy").astype(np.float32)
    except:
        print("ERROR: embeddings.npy not found.")
        return

    var num_embeddings = embeddings_np.shape[0]
    var dim = embeddings_np.shape[1]
    print("Loaded", Int(num_embeddings), "embeddings with dimension", Int(dim))

    # --- Convert NumPy -> List[List[Float32]] for GPU ---
    var embeddings = List[List[Float32]]()
    for i in range(Int(num_embeddings)):
        var row = List[Float32]()
        for j in range(Int(dim)):
            row.append(Float32(embeddings_np[i, j]))
        embeddings.append(row)

    # --- Time it ---
    var time = Python.import_module("time")
    var start_time = time.time()

    var dist_matrix = pairwise_distances_gpu(embeddings)

    var end_time = time.time()
    print("--------------------------------------------------")
    print("Mojo GPU pairwise distances complete.")
    print("Mojo Time:", (end_time - start_time) * 1000.0, "ms")
    print("--------------------------------------------------")