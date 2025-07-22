from collections.list import List
from math import sqrt, log

struct MatrixFloat32:
    var data: List[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self: Self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Float32]()
        self.data.resize(rows * cols, 0.0)

    fn __copyinit__(out self: Self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self.data = List[Float32](other.data)

    fn __getitem__(self: Self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]

    fn __setitem__(mut self, row: Int, col: Int, val: Float32):
        self.data[row * self.cols + col] = val


fn pairwise_distances(embeddings: MatrixFloat32) -> MatrixFloat32:
    var n = embeddings.rows
    var dist_matrix = MatrixFloat32(n, n)
    for i in range(n):
        for j in range(i, n):
            var dist_sq: Float32 = 0.0
            for k in range(embeddings.cols):
                var diff = embeddings[i, k] - embeddings[j, k]
                dist_sq += diff * diff
            var final_dist = sqrt(dist_sq)
            dist_matrix[i, j] = final_dist
            dist_matrix[j, i] = final_dist
            #if i == 0 and j < 5:
            #    print("DEBUG: Distance from 0 to", j, "=", final_dist)
    return dist_matrix

fn get_dynamic_sample_size(total_embeddings: Int) -> Int:
    var no_sampling_threshold = 500
    var sampling_percentage: Float32 = 0.35
    var max_sample_size_cap = 1000
    if total_embeddings <= no_sampling_threshold:
        print("INFO: (Mojo) Document has", total_embeddings, "chunks. Running on all chunks.")
        return total_embeddings
    else:
        var calculated_sample_size = Int(Float32(total_embeddings) * sampling_percentage)
        var final_sample_size = min(calculated_sample_size, max_sample_size_cap)
        print("INFO: (Mojo) Document has", total_embeddings, "chunks. Running on a sample of", final_sample_size, ".")
        return final_sample_size
    #return no_sampling_threshold + Int(sqrt(Float32(total_embeddings - no_sampling_threshold)))

fn simple_rand_int(min: Int, max: Int, seed: Int) -> Int:
    var hash = (seed * 9301 + 49297) % 233280
    return min + hash % (max - min + 1)

fn create_sampled_embeddings(original: MatrixFloat32, sample_size: Int) -> MatrixFloat32:
    var chosen_indices = List[Int]()
    chosen_indices.resize(sample_size, -1)

    var sampled_embeddings = MatrixFloat32(sample_size, original.cols)
    var i = 0
    while i < sample_size:
        var rand_index = simple_rand_int(0, original.rows - 1, i)
        var already_chosen = False
        for j in range(i):
            if chosen_indices[j] == rand_index:
                already_chosen = True
                break

        if not already_chosen:
            chosen_indices[i] = rand_index
            for col in range(original.cols):
                sampled_embeddings[i, col] = original[rand_index, col]
            i += 1

    return sampled_embeddings

fn analyze_slopes(slopes: MatrixFloat32, window_size: Int, threshold: Float32) -> List[Bool]:
    var num_embeddings = slopes.rows
    var num_slopes = slopes.cols
    var rejections = List[Bool]()
    rejections.resize(num_embeddings, False)

    for i in range(num_embeddings):
        var is_rejected = False
        for j in range(window_size, num_slopes - window_size):
            var before_mean: Float32 = 0.0
            var after_mean: Float32 = 0.0
            for k in range(window_size):
                before_mean += slopes[i, j - window_size + k]
                after_mean += slopes[i, j + k]
            before_mean /= Float32(window_size)
            after_mean /= Float32(window_size)

            if after_mean > before_mean and (after_mean - before_mean) > threshold:
                is_rejected = True
                break
        rejections[i] = is_rejected
    return rejections

fn test_fiber_bundle_hypothesis(
    embeddings: MatrixFloat32,
    r_min: Float32 = 0.01,
    r_max: Float32 = 20.0,
    n_r: Int = 50,
    window_size: Int = 10,
    change_threshold: Float32 = 1.5
) -> List[Bool]:
    var num_embeddings = embeddings.rows
    # DIAGNOSTIC PRINT: Confirm the size of the dataset being analyzed.
    #print("DEBUG: Analyzing", num_embeddings, "embeddings.")
    var dist_matrix = pairwise_distances(embeddings)

    var r_values = List[Float32]()
    r_values.resize(n_r, 0.0)
    for i in range(n_r):
        r_values[i] = r_min + (r_max - r_min) * Float32(i) / Float32(n_r - 1)

    var log_r = List[Float32]()
    log_r.resize(n_r, 0.0)
    for i in range(n_r):
        log_r[i] = log(r_values[i])

    var slopes_matrix = MatrixFloat32(num_embeddings, n_r)

    for i in range(num_embeddings):
        var nx_r = List[Int]()
        nx_r.resize(n_r, 0)
        for k in range(n_r):
            var count = 0
            for j in range(num_embeddings):
                if dist_matrix[i, j] <= r_values[k]:
                    count += 1
            nx_r[k] = count

        var log_nx_r = List[Float32]()
        log_nx_r.resize(n_r, 0.0)
        for k in range(n_r):
            log_nx_r[k] = log(Float32(nx_r[k]) + 1e-10)

        for j in range(1, n_r - 1):
            var slope = (log_nx_r[j + 1] - log_nx_r[j - 1]) / (log_r[j + 1] - log_r[j - 1])
            slopes_matrix[i, j] = slope

        # DIAGNOSTIC PRINT: Check the slopes for the first embedding.
        #if i == 0:
        #    print("DEBUG: Slopes for embedding 0 (first 5):", slopes_matrix[0,0], slopes_matrix[0,1], slopes_matrix[0,2], slopes_matrix[0,3], slopes_matrix[0,4])


        if n_r > 1:
            slopes_matrix[i, 0] = slopes_matrix[i, 1]
            slopes_matrix[i, n_r - 1] = slopes_matrix[i, n_r - 2]

    return analyze_slopes(slopes_matrix, window_size, change_threshold)
