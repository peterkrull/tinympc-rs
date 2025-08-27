use std::time::Duration;

use nalgebra::{SMatrix, SVector, matrix, vector};
use tinympc_rs::{
    Error, TinyMpc,
    cache::ArrayCache,
    project::{Box, Project, ProjectExt as _},
};

type Float = f64;

const HX: usize = 20;
const HU: usize = HX - 5;

const NX: usize = 3;
const NU: usize = 1;

const DT: Float = 0.2;
const DD: Float = 0.5 * DT * DT;

const LP: Float = 0.5;

const A: SMatrix<Float, NX, NX> = matrix![
    1., DT, DD;
    0., 1., DT;
    0., 0., LP;
];

const B: SMatrix<Float, NX, NU> = vector![0., 0., (1. - LP)];

const Q: SVector<Float, NX> = vector! {5., 0., 0.};
const R: SVector<Float, NU> = vector! {1.,};
const RHO: Float = 2.0;

fn main() -> Result<(), Error> {
    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    const NUM_CACHES: usize = 5;
    type Cache = ArrayCache<Float, NX, NU, NUM_CACHES>;
    type Mpc = TinyMpc<Float, Cache, NX, NU, HX, HU>;

    let mut mpc = Mpc::new(A, B, Q, R, RHO)?;
    mpc.config.max_iter = 400;
    mpc.config.do_check = 1;
    mpc.config.prim_tol = 0.05;
    mpc.config.dual_tol = 0.02;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut x_now = SVector::zeros();
    let mut x_ref = SMatrix::<Float, NX, HX>::zeros();

    // Velocity limiter
    let x_projector_box = Box {
        upper: vector![None, Some(0.3), None],
        lower: vector![None, Some(-0.3), None],
    };

    // Actuation limiter
    let u_projector_box = Box {
        upper: vector![Some(0.15)],
        lower: vector![Some(-0.15)],
    };

    let mut x_con = [x_projector_box.constraint()];
    let mut u_con = [u_projector_box.constraint()];

    let mut total_iters = 0;
    for k in 0..=300 {
        for i in 0..HX {
            let mut x_ref_col = SVector::zeros();

            if ((i + k) / 100) % 2 == 0 {
                x_ref_col[0] = 1.;
            } else {
                x_ref_col[0] = -1.;
            }

            x_ref.set_column(i, &x_ref_col);
        }

        let time = std::time::Instant::now();

        let solution = mpc
            .initial_condition(x_now)
            .u_constraints(u_con.as_mut())
            .x_constraints(x_con.as_mut())
            .x_reference(x_ref.as_view())
            .solve();

        total_iters += solution.iterations;

        println!(
            "{}: Got solution: {:?} in {} ms in {} iters ({:?})",
            k,
            solution.u_now().as_slice(),
            time.elapsed().as_micros() as f32 / 1e3,
            solution.iterations,
            solution.reason,
        );

        // Apply input to system
        let mut u_now = solution.u_now();
        u_projector_box.project(u_now.as_view_mut());
        x_now = A * x_now + B * u_now;

        // ------ RERUN VISUALIZATION -------

        rec.set_time("timeline", Duration::from_millis((50 * k) as u64));

        let x_mat = solution.x_prediction();
        let u_mat = solution.u_prediction();

        let x_iter = x_mat.column_iter().enumerate();
        let line_strips: [rerun::LineStrip2D; 3] = core::array::from_fn(|index| {
            rerun::LineStrip2D::from_iter(
                x_iter
                    .clone()
                    .map(|(i, vec)| [(i + k) as f32 / 100., vec[index] as f32]),
            )
        });

        let strips = rerun::LineStrips2D::new(line_strips);
        rec.log("x_pred", &strips).unwrap();

        let x_iter = x_ref.column_iter().enumerate();
        let line_strips: [rerun::LineStrip2D; 3] = core::array::from_fn(|index| {
            rerun::LineStrip2D::from_iter(
                x_iter
                    .clone()
                    .map(|(i, vec)| [(i + k) as f32 / 100., vec[index] as f32]),
            )
        });

        let strips = rerun::LineStrips2D::new(line_strips);
        rec.log("x_ref", &strips).unwrap();

        let line_strip = rerun::LineStrip2D::from_iter(
            u_mat
                .iter()
                .enumerate()
                .map(|(i, vec)| [(i + k) as f32 / 100., *vec as f32]),
        );

        let strips = rerun::LineStrips2D::new([line_strip]);
        rec.log("u_strips", &strips).unwrap();
    }

    println!("Total iterations: {total_iters}");

    Ok(())
}
