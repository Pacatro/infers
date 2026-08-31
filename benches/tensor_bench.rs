use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use infers::Tensor;

const BENCH_MATMUL_GROUP: &str = "Matrix Multiplication (N x N)";
const BENCH_TRANSPOSE_GROUP: &str = "Transpose";
const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];
const SAMPLE_SIZE: usize = 10;

fn bench_matmul_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group(BENCH_MATMUL_GROUP);

    for &n in SIZES {
        let bench_id = format!("CPU_{}x{}", n, n);

        let t1 = Tensor::rand([n, n]).unwrap();
        let t2 = Tensor::rand([n, n]).unwrap();

        group.bench_function(&bench_id, |b| {
            b.iter(|| {
                let r = t1.gemm(&t2, None, None).unwrap();
                black_box(r);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_matmul_cuda(c: &mut Criterion) {
    use infers::backends::Cuda;
    let mut group = c.benchmark_group(BENCH_MATMUL_GROUP);

    for &n in SIZES {
        let bench_id = format!("CUDA_{}x{}", n, n);

        let t1 = Tensor::rand([n, n]).unwrap().to::<Cuda>().unwrap();
        let t2 = Tensor::rand([n, n]).unwrap().to::<Cuda>().unwrap();

        group.bench_function(&bench_id, |b| {
            b.iter(|| {
                let r = t1.gemm(&t2, None, None).unwrap();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    bench_matmul_cpu(c);
    #[cfg(feature = "cuda")]
    bench_matmul_cuda(c);
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group(BENCH_TRANSPOSE_GROUP);

    for (&n, &m) in SIZES.iter().zip(SIZES.iter().rev()) {
        let bench_id = format!("Transpose_{}x{}", n, m);

        let t = Tensor::rand([n, m]).unwrap();

        group.bench_function(&bench_id, |b| {
            b.iter(|| {
                let r = t.t().unwrap();
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = matmul_benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE);
    targets = bench_matmul
}

criterion_group! {
    name = transpose_benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE);
    targets = bench_transpose
}

criterion_main!(matmul_benches, transpose_benches);
