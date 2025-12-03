use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infers::Tensor;
use std::hint::black_box;

fn bench_matmul_cpu(c: &mut Criterion) {
    let size = 900;
    let mut group = c.benchmark_group(format!("matmul {}x{}", size, size));

    group.bench_with_input(BenchmarkId::new("CPU", size), &size, |b, &size| {
        let t1 = Tensor::rand(&[size, size]);
        let t2 = Tensor::rand(&[size, size]);

        b.iter(|| {
            let result = black_box(&t1).matmul(black_box(&t2));
            black_box(result);
        });
    });

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_matmul_cuda(c: &mut Criterion) {
    use infers::backends::Cuda;

    let mut group = c.benchmark_group("matmul 900x900");
    let size = 900;

    group.bench_with_input(BenchmarkId::new("CUDA", size), &size, |b, &size| {
        let t1 = Tensor::rand(&[size, size]).to::<Cuda>().unwrap();
        let t2 = Tensor::rand(&[size, size]).to::<Cuda>().unwrap();

        b.iter(|| {
            let result = black_box(&t1).matmul(black_box(&t2));
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches_cpu, bench_matmul_cpu);

#[cfg(feature = "cuda")]
criterion_group!(benches_cuda, bench_matmul_cuda);

#[cfg(feature = "cuda")]
criterion_main!(benches_cpu, benches_cuda);

#[cfg(not(feature = "cuda"))]
criterion_main!(benches_cpu);
