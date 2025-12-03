use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{SMatrix, SVector, matrix, vector};
use tinympc_rs::{
    TinyMpc,
    cache::ArrayCache,
    project::{ProjectMulti, Sphere},
};

const HX: usize = 150;
const HU: usize = HX - 15;
const NX: usize = 9;
const NU: usize = 3;
const DT: f32 = 0.1;
const DD: f32 = 0.5 * DT * DT;
const LP: f32 = 0.95;

const A: SMatrix<f32, NX, NX> = matrix![
    1., 0., 0., DT, 0., 0., DD, 0., 0.;
    0., 1., 0., 0., DT, 0., 0., DD, 0.;
    0., 0., 1., 0., 0., DT, 0., 0., DD;
    0., 0., 0., 1., 0., 0., DT, 0., 0.;
    0., 0., 0., 0., 1., 0., 0., DT, 0.;
    0., 0., 0., 0., 0., 1., 0., 0., DT;
    0., 0., 0., 0., 0., 0., LP, 0., 0.;
    0., 0., 0., 0., 0., 0., 0., LP, 0.;
    0., 0., 0., 0., 0., 0., 0., 0., LP;
];

const B: SMatrix<f32, NX, NU> = matrix![
    0., 0., 0.;
    0., 0., 0.;
    0., 0., 0.;
    0., 0., 0.;
    0., 0., 0.;
    0., 0., 0.;
    (1. - LP), 0., 0.;
    0., (1. - LP), 0.;
    0., 0., (1. - LP);
];

const Q: SVector<f32, NX> = vector![5., 5., 5., 0., 0., 0., 0., 0., 0.];
const R: SVector<f32, NU> = vector![1., 1., 1.];
const RHO: f32 = 4.0;

fn mpc_benchmark(c: &mut Criterion) {
    const NUM_CACHES: usize = 5;
    type Cache = ArrayCache<f32, NX, NU, NUM_CACHES>;
    let cache = Cache::new(
        RHO,
        10.0,
        1.8,
        HX,
        &A,
        &B,
        &SMatrix::from_diagonal(&Q),
        &SMatrix::from_diagonal(&R),
        &SMatrix::zeros(),
    )
    .unwrap();

    type Mpc = TinyMpc<f32, Cache, NX, NU, HX, HU>;
    let mut mpc = Mpc::new(A, B, cache);
    mpc.config.max_iter = 6;
    mpc.config.do_check = 3;

    let x_now = SVector::<f32, NX>::zeros();
    let mut xref = SMatrix::<f32, NX, HX>::zeros();

    for i in 0..HX {
        let mut xref_col = SVector::<f32, NX>::zeros();
        xref_col[0] = ((i) as f32 / 25.0).sin() * 10.;
        xref_col[1] = ((i) as f32 / 25.0).cos() * 10.;
        xref.set_column(i, &xref_col);
    }

    let x_projector = Sphere {
        center: vector![
            None,
            None,
            None,
            Some(0.0),
            Some(0.0),
            Some(0.0),
            None,
            None,
            None
        ],
        radius: 5.0,
    };
    let u_projector = Sphere {
        center: vector![Some(0.0), Some(0.0), Some(0.0)],
        radius: 10.0,
    };

    let mut x_con = [x_projector.constraint()];
    let mut u_con = [u_projector.constraint()];

    c.bench_function("tinympc_kinematic3d_solve", |b| {
        b.iter(|| {
            let _solution = mpc
                .initial_condition(x_now)
                .u_constraints(&mut u_con)
                .x_constraints(&mut x_con)
                .x_reference(&xref)
                .solve();
        });
    });
}

criterion_group!(benches, mpc_benchmark);
criterion_main!(benches);
