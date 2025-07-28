from mojo import kernel, index  # for defining GPU kernels



@kernel
fn pairwise_dist_kernel(
    input_ptr: Pointer[Float32],
    output_ptr: Pointer[Float32],
    n: Int,
    d: Int
):
    i = index(0)
    j = index(1)
    if i >= n or j >= n:
        return

    var dist: Float32 = 0
    for k in range(0, d):
        var a = input_ptr.at(i * d + k)
        var b = output_ptr.at(j * d + k)
        var diff = a - b
        dist += diff * diff

    output_ptr.at(i * n + j).set(dist)



fn pairwise_distances_gpu(embeddings: List[List[Float32]]) -> List[List[Float32]]:
    var n: Int = len(embeddings)
    var d: Int = len(embeddings[0])

    var flat = List[Float32](capacity=n * d)
    for i in range(n):
        for j in range(d):
            flat.append(embeddings[i][j])

    var output_flat = List[Float32](capacity=n * n)
    for _ in range(n * n):
        output_flat.append(0.0)

    var input_ptr = flat.unsafe_pointer_cast()
    var output_ptr = output_flat.unsafe_pointer_cast()
    
    pairwise_dist_kernel.launch(
        grid=(n, n),
        block=(1, 1),
        args=(input_ptr, output_ptr, n, d)
    )

    var output = List[List[Float32]]()
    output.reserve(n)

    for i in range(n):
        var row = List[Float32]()
        row.reserve(n)
        for j in range(n):
            row.append(output_flat[i * n + j])
        output.append(row)

    return output
