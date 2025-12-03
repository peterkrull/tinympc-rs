use std::time::Duration;

use nalgebra::{SMatrix, SVector, matrix, vector};
use tinympc_rs::{
    Constant, Error, Expand, TinyMpc, cache::ArrayCache, project::{Box, ProjectExt as _, ProjectSingle}
};

type Float = f64;

const HX: usize = 150;
const HU: usize = HX - 10;

const NX: usize = 3;
const NU: usize = 1;

const DT: Float = 0.05;
const DD: Float = 0.5 * DT * DT;

const LP: Float = 0.90;

const A: SMatrix<Float, NX, NX> = matrix![
    1., DT, DD;
    0., 1., DT;
    0., 0., LP;
];

const B: SMatrix<Float, NX, NU> = matrix![0.; 0.; (1. - LP)];

const Q: SMatrix<Float, NX, NX> = matrix![
    10., 0., 0.;
    0., 0.1, 0.;
    0., 0., 0.1;
];

const R: SMatrix<Float, NU, NU> = matrix![0.1];

const RHO: Float = 8.0;

fn main() -> Result<(), Error> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .without_timestamps()
        .init()
        .ok();

    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    eprintln!("Run this to view profiling data:  puffin_viewer {server_addr}");
    profiling::puffin::set_scopes_on(true);

    std::thread::sleep(std::time::Duration::from_millis(1000));

    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    const NUM_CACHES: usize = 5;
    type Cache = ArrayCache<Float, NX, NU, NUM_CACHES>;
    type Mpc = TinyMpc<Float, Cache, NX, NU, HX, HU>;

    let cache = Cache::new(RHO, 10.0, 2.0, 1000, &A, &B, &Q, &R, &SMatrix::zeros()).unwrap();

    let mut mpc = Mpc::new(A, B, cache);
    mpc.config.max_iter = 5;
    mpc.config.do_check = 1;
    mpc.config.prim_tol = 1e-3;
    mpc.config.dual_tol = 1e-3;
    mpc.config.relaxation = 1.6;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut x_now = SVector::zeros();
    let mut x_ref = SMatrix::<Float, NX, HX>::zeros();

    // Velocity limiter
    let x_projector_box = Expand::new([1], Box {
        upper: vector![0.8],
        lower: vector![-0.8],
    });

    // Actuation limiter
    let u_projector_box = Expand::new([0], Box {
        upper: vector![0.5],
        lower: vector![-0.5],
    });

    let mut x_con = [Constant::new(&x_projector_box).constraint_owned()];
    let mut u_con = [Constant::new(&u_projector_box).constraint_owned()];

    rec.log_static(
        "prim_res",
        &rerun::SeriesLines::new()
            .with_colors([[255, 100, 100]])
            .with_names(["primal residual"])
            .with_widths([1.0]),
    )
    .unwrap();

    rec.log_static(
        "dual_res",
        &rerun::SeriesLines::new()
            .with_colors([[80, 220, 80]])
            .with_names(["dual residual"])
            .with_widths([1.0]),
    )
    .unwrap();

    let mut total_iters = 0;
    for k in 0..=500 {
        for i in 0..HX {
            let mut x_ref_col = SVector::zeros();

            if ((i + k) / 170) % 2 == 0 {
                x_ref_col[0] = 1.;
            } else {
                x_ref_col[0] = -1.;
            }

            x_ref.set_column(i, &x_ref_col);
        }

        let time = std::time::Instant::now();

        profiling::scope!("solver loop: ", format!("iteration {k}"));
        let solution = mpc
            .initial_condition(x_now)
            .u_constraints(&mut u_con)
            .x_constraints(&mut x_con)
            .x_reference(&x_ref)
            .solve();
        profiling::finish_frame!();

        total_iters += solution.iterations;

        println!(
            "{}: Got solution: {:?} in {} ms in {} iters ({:?})",
            k,
            solution.u_now().as_slice(),
            time.elapsed().as_micros() as f32 / 1e3,
            solution.iterations,
            solution.reason,
        );

        // Apply (constrained) input to system
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

        let res = rerun::Scalars::single(solution.prim_residual);
        rec.log("prim_res", &res).unwrap();

        let res = rerun::Scalars::single(solution.dual_residual);
        rec.log("dual_res", &res).unwrap();
    }

    println!("Total iterations: {total_iters}");

    Ok(())
}
