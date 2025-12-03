use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use infers::Tensor;

const BENCH_GROUP: &str = "Matrix Multiplication (N x N)";
const SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];
const SAMPLE_SIZE: usize = 10;

fn bench_cpu_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group(BENCH_GROUP);

    for &n in SIZES {
        let bench_id = format!("CPU_{}x{}", n, n);

        let t1 = Tensor::rand(&[n, n]);
        let t2 = Tensor::rand(&[n, n]);

        group.bench_function(&bench_id, |b| {
            b.iter(|| {
                let r = t1.matmul(&t2);
                black_box(r);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_sizes(c: &mut Criterion) {
    use infers::backends::Cuda;
    let mut group = c.benchmark_group(BENCH_GROUP);

    for &n in SIZES {
        let bench_id = format!("CUDA_{}x{}", n, n);

        let t1 = Tensor::rand(&[n, n]).to::<Cuda>().unwrap();
        let t2 = Tensor::rand(&[n, n]).to::<Cuda>().unwrap();

        group.bench_function(&bench_id, |b| {
            b.iter(|| {
                let r = t1.matmul(&t2);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_cpu_sizes(c);
    #[cfg(feature = "cuda")]
    bench_cuda_sizes(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE);
    targets = bench_all
}
criterion_main!(benches);
