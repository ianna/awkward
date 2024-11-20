// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code) = args
// 
//     # Ensure block size is valid
//     if block[0] <= 0:
//         raise ValueError("Block size must be greater than 0")
// 
//     # Compute grid size
//     grid_size = math.ceil(lenparents / block[0]) 
// 
//     # Temporary arrays for block-level results
//     block_results = cupy.full(grid_size, identity, dtype=toptr.dtype)
//     block_parents = cupy.full(grid_size, -1, dtype=parents.dtype)
// 
//     print("parents:", parents)
// 
//     # Launch the first kernel
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_max_a",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (
//         toptr, fromptr, parents, lenparents, outlength, 
//         toptr.dtype.type(identity), invocation_index, err_code))
// 
//     # Launch the second kernel (with shared memory usage)
//     shared_mem_size = block[0] * (toptr.itemsize + parents.itemsize)  # Shared memory size
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_max_b",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (
//         toptr, fromptr, parents, lenparents, outlength, 
//         toptr.dtype.type(identity), block_results, block_parents, 
//         invocation_index, err_code), shared_mem=shared_mem_size)
// 
//     # Debugging: Print intermediate results
//     print("block_results:", block_results)
//     print("block_parents:", block_parents)
//     print("toptr (after kernel b):", toptr)
//     print("grid_size:", grid_size)
// 
//     # Launch the third kernel (global reduction across blocks)
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_max_c",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((1,), (grid_size,), (toptr, block_results, block_parents, grid_size, invocation_index, err_code))
// 
// # Mark the kernels in the output dictionary
// out["awkward_reduce_max_a", {dtype_specializations}] = None
// out["awkward_reduce_max_b", {dtype_specializations}] = None
// out["awkward_reduce_max_c", {dtype_specializations}] = None
// END PYTHON

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_max_a(
    T* toptr,
    const C* fromptr,
    const U* parents,
    int64_t lenparents,
    int64_t outlength,
    T identity,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < outlength) {
      toptr[thread_id] = identity;
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_max_b(
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
    extern __shared__ char shared_memory[];
    T* shared_temp = reinterpret_cast<T*>(shared_memory);

    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize local variables
    T local_max = identity;
    U local_parent = -1;

    if (thread_id < lenparents) {
      local_max = fromptr[thread_id];
      local_parent = parents[thread_id];
      shared_temp[threadIdx.x] = local_max;
    } else {
      shared_temp[threadIdx.x] = identity;
    }
    __syncthreads();

    // Perform block-level reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
      if (threadIdx.x >= stride && 
          thread_id - stride >= 0 && 
          parents[thread_id] == parents[thread_id - stride]) {
        shared_temp[threadIdx.x] = max(shared_temp[threadIdx.x], shared_temp[threadIdx.x - stride]);
      }
      __syncthreads();
    }

    // Store the results of the block reduction
    if (threadIdx.x == blockDim.x - 1 || thread_id == lenparents - 1 || 
        parents[thread_id] != parents[thread_id + 1]) {
      int64_t parent = parents[thread_id];
      atomicMax(&toptr[parent], shared_temp[threadIdx.x]);

      // Only the last thread updates block results and parents
      if (threadIdx.x == blockDim.x - 1 || 
          thread_id == lenparents - 1 || 
          parents[thread_id] != parents[thread_id + 1]) {
        block_results[blockIdx.x] = shared_temp[threadIdx.x];
        block_parents[blockIdx.x] = parent; // Ensure block_parents is updated
      }
    }
  }
}

template <typename T, typename C, typename U>
__global__ void
awkward_reduce_max_c(
    T* toptr,
    const T* block_results,
    const U* block_parents,
    int64_t num_blocks,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < num_blocks) {
      U parent = block_parents[thread_id];
      T value = block_results[thread_id];

      if (parent != -1) {
        atomicMax(&toptr[parent], value);
      }
    }
  }
}
