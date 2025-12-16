use std::time::Duration;

use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut, matrix, vector};
use rerun::Color;
use tinympc_rs::{
    Error, Solver,
    policy::ArrayPolicy,
    project::{Affine, Box, ProjectMulti, ProjectMultiExt, ProjectSingleExt, Sphere},
};

const HX: usize = 150;
const HU: usize = HX - 20;

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

fn sys(mut xnext: SVectorViewMut<f32, NX>, x: SVectorView<f32, NX>, u: SVectorView<f32, NU>) {
    xnext[0] = x[0] + x[3] * DT + x[6] * DD;
    xnext[1] = x[1] + x[4] * DT + x[7] * DD;
    xnext[2] = x[2] + x[5] * DT + x[8] * DD;
    xnext[3] = x[3] + x[6] * DT;
    xnext[4] = x[4] + x[7] * DT;
    xnext[5] = x[5] + x[8] * DT;
    xnext[6] = x[6] * LP + (1.0 - LP) * u[0];
    xnext[7] = x[7] * LP + (1.0 - LP) * u[1];
    xnext[8] = x[8] * LP + (1.0 - LP) * u[2];
}

const Q: SVector<f32, NX> = vector! {3., 3., 3., 0., 0., 0., 0., 0., 0.};
const R: SVector<f32, NU> = vector! {1., 1., 1.,};
const RHO: f32 = 6.0;

fn main() -> Result<(), Error> {
    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .without_timestamps()
        .init()
        .ok();

    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    eprintln!("Run this to view profiling data:  puffin_viewer {server_addr}");
    // profiling::puffin::set_scopes_on(true);

    const NUM_CACHES: usize = 9;
    type Cache = ArrayPolicy<f32, NX, NU, NUM_CACHES>;
    let cache = Cache::new(
        RHO,
        10.0,
        2.0,
        HX,
        &A,
        &B,
        &SMatrix::from_diagonal(&Q),
        &SMatrix::from_diagonal(&R),
        &SMatrix::zeros(),
    )
    .unwrap();

    type Mpc = Solver<f32, Cache, NX, NU, HX, HU>;
    let mut mpc = Mpc::new(A, B, cache).with_sys(sys);
    mpc.config.max_iter = 5;
    mpc.config.do_check = 3;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut x_now = SVector::zeros();
    let mut xref = SMatrix::<f32, NX, HX>::zeros();

    let x_project_sphere = Sphere {
        center: SVector::zeros(),
        radius: 5.0,
    }
    .dim_lift([3, 4, 5]);

    let x_projector_affine = Affine::new()
        .normal([-1.0, -1.0])
        .distance(18.0)
        .dim_lift([0, 1]);

    // Two (or more!) projectors can be bundled together, saving a pair of slack/dual variables.
    // This should only be done if one constraint cannot immediately invalidate another constrant.
    let x_projector_bundle = (x_project_sphere, x_projector_affine).time_fixed();

    // We can also iteratively project a bundle if they could push each other out of feasible region
    // let x_projector_bundle = [&x_projector_bundle; 10];

    let u_projector_sphere = Sphere {
        center: SVector::zeros(),
        radius: 10.0,
    };

    let u_projector_box = Box {
        upper: vector![2.0, 2.0, 2.0],
        lower: vector![-2.0, -2.0, -2.0],
    };

    let u_projector_bundle = (u_projector_sphere, u_projector_box).time_fixed();

    let mut x_con = [x_projector_bundle.constraint()];
    let mut u_con = [u_projector_bundle.constraint()];

    let mut true_pos = vec![vector![0.0, 0.0, 0.0]];

    let mut total_cost = 0.0;

    // TODO figure out visualization of planes
    // let normalized = vector![-1.0, -1.0, 0., 0., 0., 0., 0., 0., 0.].normalize().xyz();
    // let normal = [normalized[0], normalized[1], normalized[2]];
    // let plane = Plane3D::new(normal, 18.0);
    // rec.log("position_constraint", &plane).unwrap();

    let mut total_iters = 0;
    let mut k = 0;
    while k < 5000 {
        k += 1;

        for i in 0..HX {
            let mut xref_col = SVector::zeros();
            xref_col[0] = ((i + k) as f32 / 25.0).sin() * 10.;
            xref_col[1] = ((i + k) as f32 / 25.0).cos() * 10.;
            xref_col[2] = ((i + k) as f32 / 1000.0).cos() * 50.;

            match ((i + k) / 500) % 4 {
                0 => {
                    xref_col[0] += 15.;
                    xref_col[1] += 15.;
                }
                1 => {
                    xref_col[0] -= 15.;
                    xref_col[1] += 15.;
                }
                2 => {
                    xref_col[0] -= 15.;
                    xref_col[1] -= 15.;
                }
                _ => {
                    xref_col[0] += 15.;
                    xref_col[1] -= 15.;
                }
            }

            xref.set_column(i, &xref_col);
        }

        let time = std::time::Instant::now();

        let solution = {
            profiling::scope!("solver loop: ", format!("iteration {k}"));
            mpc.initial_condition(x_now)
                .x_reference(&xref)
                .x_constraints(&mut x_con)
                .u_constraints(&mut u_con)
                .solve()
        };
        profiling::finish_frame!();

        total_iters += solution.iterations;

        let mut iteration_cost = 0.0;
        for col in solution.x_prediction_full().column_iter() {
            iteration_cost += (col.transpose() * SMatrix::from_diagonal(&Q) * col)[0];
        }

        for col in solution.u_prediction_full().column_iter() {
            iteration_cost += (col.transpose() * SMatrix::from_diagonal(&R) * col)[0];
        }

        total_cost += iteration_cost;

        println!(
            "Got solution: {:?} in {} ms) in {} iters ({:?}), cost: {}",
            solution.u_now().as_slice(),
            time.elapsed().as_micros() as f32 / 1e3,
            solution.iterations,
            solution.reason,
            iteration_cost,
        );

        // Apply input to system
        let u_now = solution.u_now();
        x_now = A * x_now + B * u_now;

        // ------ RERUN VISUALIZATION -------

        rec.set_time("timeline", Duration::from_millis((50 * k) as u64));

        let u_mat = solution.u_prediction_full();
        let u_iter = u_mat.column_iter().enumerate();

        let u_x = rerun::LineStrip2D::from_iter(
            u_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec.x as f32]),
        );
        let u_y = rerun::LineStrip2D::from_iter(
            u_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec.y as f32]),
        );
        let u_z = rerun::LineStrip2D::from_iter(
            u_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec.z as f32]),
        );

        let strips = rerun::LineStrips2D::new([u_x, u_y, u_z]);
        rec.log("u_strips", &strips).unwrap();

        let mut x_mat = solution.x_prediction_full();
        let x_iter = x_mat.column_iter().enumerate();
        let x_x = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[0] as f32]),
        );
        let x_y = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[1] as f32]),
        );
        let x_z = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[2] as f32]),
        );
        let strips = rerun::LineStrips2D::new([x_x, x_y, x_z]);
        rec.log("x_pos_strips", &strips).unwrap();

        let x_iter = x_mat.column_iter().enumerate();
        let x_x = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[3] as f32]),
        );
        let x_y = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[4] as f32]),
        );
        let x_z = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[5] as f32]),
        );
        let strips = rerun::LineStrips2D::new([x_x, x_y, x_z]);
        rec.log("x_vel_strips", &strips).unwrap();

        x_projector_bundle.project_multi(&mut x_mat);

        let x_iter = x_mat.column_iter().enumerate();
        let x_x = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[3] as f32]),
        );
        let x_y = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[4] as f32]),
        );
        let x_z = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[5] as f32]),
        );
        let strips = rerun::LineStrips2D::new([x_x, x_y, x_z]);
        rec.log("x_vel_projected_strips", &strips).unwrap();

        let x_iter = xref.column_iter().enumerate();
        let x_x = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[0] as f32]),
        );
        let x_y = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[1] as f32]),
        );
        let x_z = rerun::LineStrip2D::from_iter(
            x_iter
                .clone()
                .map(|(i, vec)| [(i + k) as f32 / 10., vec[2] as f32]),
        );
        let strips = rerun::LineStrips2D::new([x_x, x_y, x_z]);
        rec.log("xref_strips", &strips).unwrap();

        let x = x_now;

        let vec = solution.x_prediction(0);
        rec.log(
            "position",
            &rerun::Points3D::new([[vec[0] as f32, vec[1] as f32, vec[2] as f32]])
                .with_radii([0.2]),
        )
        .unwrap();

        let vec = xref.column(0);
        rec.log(
            "desired position",
            &rerun::Points3D::new([[vec[0] as f32, vec[1] as f32, vec[2] as f32]])
                .with_radii([0.2]),
        )
        .unwrap();

        true_pos.push(vector![x[0], x[1], x[2]]);

        let true_pos_strips = rerun::LineStrip3D::from_iter(
            true_pos
                .iter()
                .map(|vec| [vec[0] as f32, vec[1] as f32, vec[2] as f32]),
        );

        let strips = rerun::LineStrips3D::new([true_pos_strips]);
        rec.log("x_position", &strips).unwrap();

        let pos_ref_strips = rerun::LineStrip3D::from_iter(
            xref.column_iter()
                .map(|vec| [vec[0] as f32, vec[1] as f32, vec[2] as f32]),
        );

        let strips = rerun::LineStrips3D::new([pos_ref_strips]);
        rec.log("x_position_ref", &strips).unwrap();

        let pos_pred_strips = rerun::LineStrip3D::from_iter(
            solution.x_prediction_full()
                .column_iter()
                .map(|vec| [vec[0] as f32, vec[1] as f32, vec[2] as f32]),
        );

        let strips = rerun::LineStrips3D::new([pos_pred_strips]).with_colors([Color::WHITE]);
        rec.log("x_position_pred", &strips).unwrap();
    }

    println!("Total iterations: {total_iters}, cost: {total_cost}");

    Ok(())
}
