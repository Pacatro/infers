extern "C" __global__ void relu(const float *in, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = in[idx] > 0 ? in[idx] : 0;
  }
}
