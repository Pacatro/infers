extern "C" __global__ void mul(const float *a, const float *b, float *c,
                               size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}
