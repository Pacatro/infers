#define TILE_SIZE 16

extern "C" __global__ void gemm(int m, int n, int k, float alpha,
                                const float *a, int a_row_stride,
                                int a_col_stride, const float *b,
                                int b_row_stride, int b_col_stride, float beta,
                                float *c, int c_row_stride, int c_col_stride) {

  // TILE_SIZE + 1 = 17 will ensure column accesses by 16 threads fall into
  // different banks.
  __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  // The thread (ty, tx) is responsible for computing C[row, col]
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;

  // Determine how many blocks we need to iterate over k
  int num_blocks_k = (k + TILE_SIZE - 1) / TILE_SIZE;

  for (int p_block = 0; p_block < num_blocks_k; ++p_block) {
    int p_base = p_block * TILE_SIZE;

    // --- 1. Load Tile from A (Global to Shared) ---
    // A single thread (ty, tx) is now responsible for loading *multiple*
    // elements into As[ty][tx], As[ty][tx + TILE_SIZE], etc., OR multiple
    // threads load a single row/col. To maximize coalescing, threads access
    // contiguous memory in global A.

    // Strategy: Each thread loads one element from A and one from B.
    // A (row, p): Thread (ty, tx) loads A[row, p_base + tx] into As[ty][tx]
    // B (p, col): Thread (ty, tx) loads B[p_base + ty, col] into Bs[ty][tx]

    // Load As[ty][tx] and As[ty+16][tx] etc. from global A
    // We will make threads load along the 'inner' dimension (p) for A, and (p)
    // for B. To load TILE_SIZE*TILE_SIZE elements, all TILE_SIZE*TILE_SIZE
    // threads are used. We will use the thread index 'tx' for the column (p)
    // index, and 'ty' for the row index.

    int a_load_row = by * TILE_SIZE + ty;
    int a_load_col = p_base + tx;

    int b_load_row = p_base + ty;
    int b_load_col = bx * TILE_SIZE + tx;

    if (a_load_row < m && a_load_col < k) {
      int a_idx = a_load_row * a_row_stride + a_load_col * a_col_stride;
      As[ty][tx] = a[a_idx];
    } else {
      As[ty][tx] = 0.0f;
    }

    if (b_load_row < k && b_load_col < n) {
      int b_idx = b_load_row * b_row_stride + b_load_col * b_col_stride;
      Bs[ty][tx] = b[b_idx];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // --- 2. Compute the Partial Sum ---
    // The threads calculate the partial dot product for their C element
    // As[ty][i] is the element A[row, p_base + i]
    // Bs[i][tx] is the element B[p_base + i, col]

    for (int i = 0; i < TILE_SIZE; ++i) {
      // Note: We use As[ty][i] and Bs[i][tx] which is a transposed access
      // for Bs, but the layout of Bs in shared memory is already transposed
      // in terms of the block/thread indices used for loading above!
      // The loading: Bs[ty][tx] = B[p_base + ty, b_load_col]
      // We want to access B[p_base + i, col] -> which is Bs[i][tx]
      // The improved access uses:
      // As[ty][i] (thread row index, inner loop index)
      // Bs[i][tx] (inner loop index, thread col index)
      sum += As[ty][i] * Bs[i][tx];
    }

    __syncthreads(); // Wait for all threads to finish computing this block
  }

  // --- 3. Final Result Update ---
  if (row < m && col < n) {
    // Correct C index calculation is essential
    int c_idx = row * c_row_stride + col * c_col_stride;
    float c_old = c[c_idx];
    c[c_idx] = alpha * sum + beta * c_old;
  }
}
