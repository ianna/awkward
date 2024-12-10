// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, invocation_index, err_code) = args
// 
//     if block[0] > 0:
//         grid_size = math.ceil(lenparents / block[0])
//     else:
//         grid_size = 1
// 
//     # Temporary arrays for block-level results
//     block_results = cupy.full(grid_size, -1, dtype=toptr.dtype)
//     block_parents = cupy.full(grid_size, -1, dtype=parents.dtype)
// 
//     print("parents:", parents)
//     identity = get_identity(toptr.dtype, "max")
// 
//     # Launch the first kernel
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_argmax_a",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (
//         toptr, fromptr, parents, lenparents, outlength, 
//         invocation_index, err_code))
// 
//     # Launch the second kernel (with shared memory usage)
//     shared_mem_size = 2 * block[0] * toptr.itemsize
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_argmax_b",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (
//         toptr, fromptr, parents, lenparents, outlength, 
//         identity, block_results, block_parents, 
//         invocation_index, err_code), shared_mem=shared_mem_size)
// 
//     # Debugging: Print intermediate results
//     print("block_results:", block_results)
//     print("block_parents:", block_parents)
//     print("toptr (after kernel b):", toptr)
//     print("grid_size:", grid_size)
// 
// 
// # Mark the kernels in the output dictionary
// out["awkward_reduce_argmax_a", {dtype_specializations}] = None
// out["awkward_reduce_argmax_b", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    // Early exit if there are no parents to process
    if (lenparents == 0) return;

    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      toptr[thread_id] = -1; // Initialize indices to -1 (invalid index)
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_argmax_b(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T identity,
    T* block_results,
    U* block_parents,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    // Early exit if there are no parents to process
    if (lenparents == 0) return;
    
    extern __shared__ char shared_memory[];
    T* shared_temp = reinterpret_cast<T*>(shared_memory);
    T* shared_indices = reinterpret_cast<T*>(shared_memory + blockDim.x * sizeof(T));

    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize local variables
    T local_max = identity;
    U local_index = -1;
    U local_parent = -1;

    if (thread_id < lenparents) {
      local_max = fromptr[thread_id];
      local_index = thread_id;  // Track the index of the max value
      local_parent = parents[thread_id];  // Track the parent of the current element
      shared_temp[threadIdx.x] = local_max;
      shared_indices[threadIdx.x] = local_index;
    } else {
      shared_temp[threadIdx.x] = identity;
      shared_indices[threadIdx.x] = -1;  // Invalid index when out of bounds
    }
    __syncthreads();

     // Perform block-level reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x >= stride && thread_id - stride >= 0 &&
            parents[thread_id] == parents[thread_id - stride]) {
            // Compare max values and update
            if (shared_temp[threadIdx.x] < shared_temp[threadIdx.x - stride]) {
                shared_temp[threadIdx.x] = shared_temp[threadIdx.x - stride];
                shared_indices[threadIdx.x] = shared_indices[threadIdx.x - stride];
            }
        }
        __syncthreads();
    }

    // Store the results of the block reduction
    if (threadIdx.x == blockDim.x - 1 || 
        thread_id == lenparents - 1 || 
        parents[thread_id] != parents[thread_id + 1]) {
        
        int64_t parent = parents[thread_id];
        atomicMax(&toptr[parent], shared_indices[threadIdx.x]);

        // Only the last thread updates block results and parents
        if (threadIdx.x == blockDim.x - 1 || 
            thread_id == lenparents - 1 || 
            parents[thread_id] != parents[thread_id + 1]) {
            block_results[blockIdx.x] = shared_temp[threadIdx.x];
            block_parents[blockIdx.x] = parent;  // Ensure block_parents is updated
        }
    }
  }
}
