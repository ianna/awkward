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
//     # Temporary array for intermediate results
//     temp = cupy.full(lenparents, identity, dtype=toptr.dtype)
// 
//     # Launch the first kernel
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_max_a",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
// 
//     # Launch the second kernel
//     cuda_kernel_templates.get_function(fetch_specialization([
//         "awkward_reduce_max_b",
//         cupy.dtype(toptr.dtype).type,
//         cupy.dtype(fromptr.dtype).type,
//         parents.dtype
//     ]))((grid_size,), block, (toptr, fromptr, parents, lenparents, outlength, toptr.dtype.type(identity), temp, invocation_index, err_code))
// 
// # Mark the kernels in the output dictionary
// out["awkward_reduce_max_a", {dtype_specializations}] = None
// out["awkward_reduce_max_b", {dtype_specializations}] = None
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
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread_id is within bounds
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
    T* temp,
    uint64_t invocation_index,
    uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = threadIdx.x;

    // Ensure thread_id is within bounds
    if (thread_id < lenparents) {
        temp[thread_id] = fromptr[thread_id];
    } else {
        temp[thread_id] = identity;
    }
    __syncthreads();

    // Reduction within each block
    for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
      if (idx >= stride && thread_id < lenparents &&
          parents[thread_id] == parents[thread_id - stride]) {
        T val = temp[thread_id - stride];
        temp[thread_id] = max(temp[thread_id], val);
      }
      __syncthreads();
    }

    // Write the block-level maximum to the output array
    int64_t parent = parents[thread_id];
    if (idx == blockDim.x - 1 || thread_id == lenparents - 1 ||
        parents[thread_id] != parents[thread_id + 1]) {
      atomicMax(&toptr[parent], temp[thread_id]);
    }
  }
}
