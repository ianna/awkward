// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

template <typename T, typename C>
__global__ void
awkward_RegularArray_getitem_next_range_spreadadvanced(T* toadvanced,
                                                       const C* fromadvanced,
                                                       int64_t length,
                                                       int64_t nextsize,
                                                       uint64_t invocation_index,
                                                       uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) % length;
    int64_t thready_id = (blockIdx.x * blockDim.x + threadIdx.x) % nextsize;

    toadvanced[(thread_id * nextsize) + thready_id] = fromadvanced[thread_id];
  }
}