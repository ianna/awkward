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
//         invocation_index, err_code), shared_mem=shared_mem_size)
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
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    // Early exit if there are no parents to process
    if (lenparents == 0) return;

    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit if thread is out of bounds
    if (thread_id >= lenparents) return;

    extern __shared__ char shared_memory[];
    T* shared_temp = reinterpret_cast<T*>(shared_memory);
    T* shared_indices = reinterpret_cast<T*>(shared_memory + blockDim.x * sizeof(T));

    // Initialize shared memory with the current value and index
    shared_temp[threadIdx.x] = fromptr[thread_id];
    shared_indices[threadIdx.x] = thread_id;
    __syncthreads();

    // Perform block-level reduction for argmax
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
      int neighbor_idx = threadIdx.x - stride;
      if (neighbor_idx >= 0 && thread_id >= stride &&
          parents[thread_id] == parents[thread_id - stride]) {
        if (shared_temp[threadIdx.x] < shared_temp[neighbor_idx]) {  // Argmax comparison
          shared_temp[threadIdx.x] = shared_temp[neighbor_idx];
          shared_indices[threadIdx.x] = shared_indices[neighbor_idx];
        }
      }
      __syncthreads();
    }

    // Store block-level results
    if (threadIdx.x == blockDim.x - 1 || 
        thread_id == lenparents - 1 || 
        (thread_id + 1 < lenparents && parents[thread_id] != parents[thread_id + 1])) {
      int64_t parent = parents[thread_id];

      // Validate parent index before updating `toptr`
      if (parent >= 0 && parent < outlength) {
        int current_index = shared_indices[threadIdx.x];
        C current_value = fromptr[current_index];

        // Use atomicCAS to compare values and update `toptr` conditionally
        bool updated = false;
        while (!updated) {
          int existing_index = toptr[parent];
          if (existing_index == -1) {
            // If `toptr[parent]` is uninitialized, assign the current index
            if (atomicCAS(&toptr[parent], -1, current_index) == -1) {
              updated = true;  // Successfully assigned
            }
          } else {
            // Compare the current value with the existing value
            C existing_value = fromptr[existing_index];
            if (current_value > existing_value) {  // Argmax comparison
              // Attempt to replace the index with the current one
              if (atomicCAS(&toptr[parent], existing_index, current_index) == existing_index) {
                updated = true;  // Successfully replaced
              }
            } else {
              // No update needed
              updated = true;
            }
          }
        }
      } else {
        // Debugging: Print invalid parent indices
        printf("Invalid parent index: %lld (thread_id: %lld)\n", parent, thread_id);
      }
    }
  }
}
