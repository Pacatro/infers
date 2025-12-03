extern "C" __global__ void gemm(int m, int n, int k, float alpha,
                                const float *a, const float *b, float beta,
                                float *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;

    for (int p = 0; p < k; p++) {
      sum += a[row * k + p] * b[p * n + col];
    }

    float cOld = c[row * n + col];
    c[row * n + col] = alpha * sum + beta * cOld;
  }
}
