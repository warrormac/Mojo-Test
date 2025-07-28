
from topology_gpu import pairwise_distances_gpu
from python import Python, PythonObject

fn main() raises:
    print("Loading embeddings from embeddings.npy...")

    var np: PythonObject = Python.import_module("numpy")
    var embeddings_np: PythonObject
    try:
        embeddings_np = np.load("embeddings.npy").astype(np.float32)
    except:
        print("ERROR: embeddings.npy not found.")
        return

    var num_embeddings: Int = Int(embeddings_np.shape[0])
    var dim: Int = Int(embeddings_np.shape[1])
    print("Loaded", num_embeddings, "embeddings with dimension", dim)

    # Convert numpy array to List[List[Float32]]
    var embeddings = List[List[Float32]]()
    for i in range(num_embeddings):
        var row = List[Float32]()
        for j in range(dim):
            row.append(Python.cast[Float32](embeddings_np.getitem(i).getitem(j)))
        embeddings.append(row)

    print("Launching GPU kernel for pairwise distances...")
    var time = Python.import_module("time")
    var start_time = time.time()

    var dist_matrix = pairwise_distances_gpu(embeddings)

    var end_time = time.time()
    var elapsed_ms = (end_time - start_time) * 1000.0

    print("GPU pairwise distance matrix computed.")
    print("Elapsed time:", elapsed_ms, "ms")
    print("Matrix size:", len(dist_matrix), "x", len(dist_matrix[0]))
