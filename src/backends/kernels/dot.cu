extern "C" __global__ void dot(const float *a, const float *b, float *c,
                               size_t n) {
  __shared__ float cache[256];
  
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  
  float temp = 0.0f;
  while (idx < n) {
    temp += a[idx] * b[idx];
    idx += stride;
  }
  
  cache[threadIdx.x] = temp;
  __syncthreads();
  
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
  }
  
  if (threadIdx.x == 0) {
    atomicAdd(c, cache[0]);
  }
}
