extern "C" __global__ void gemm(int m, int n, int k, float alpha,
                                const float *a, const float *b, float beta,
                                float *c) {
  __shared__ float As[16][16];
  __shared__ float Bs[16][16];
  
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  
  int row = by * 16 + ty;
  int col = bx * 16 + tx;
  
  float sum = 0.0f;
  
  for (int p = 0; p < (k + 15) / 16; ++p) {
    if (row < m && p * 16 + tx < k) {
      As[ty][tx] = a[row * k + p * 16 + tx];
    } else {
      As[ty][tx] = 0.0f;
    }
    
    if (p * 16 + ty < k && col < n) {
      Bs[ty][tx] = b[(p * 16 + ty) * n + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    for (int i = 0; i < 16; ++i) {
      sum += As[ty][i] * Bs[i][tx];
    }
    
    __syncthreads();
  }
  
  if (row < m && col < n) {
    float cOld = c[row * n + col];
    c[row * n + col] = alpha * sum + beta * cOld;
  }
}
