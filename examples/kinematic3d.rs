use std::time::Duration;

use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut, matrix, vector};
use rerun::Color;
use tinympc_rs::{
    Error, TinyMpc,
    cache::LookupCache,
    project::{Affine, Box, Project, ProjectExt as _, Sphere},
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

    const NUM_CACHES: usize = 5;
    type Cache = LookupCache<f32, NX, NU, NUM_CACHES>;
    type Mpc = TinyMpc<f32, Cache, NX, NU, HX, HU>;

    let mut mpc = Mpc::new(A, B, Q, R, RHO)?.with_sys(sys);
    mpc.config.max_iter = 5;
    mpc.config.do_check = 2;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut x_now = SVector::zeros();
    let mut xref = SMatrix::<f32, NX, HX>::zeros();

    #[rustfmt::skip]
    let x_project_sphere = Sphere {
        center: vector![None, None, None, Some(0.0), Some(0.0), Some(0.0), None, None, None],
        radius: 5.0,
    };

    let x_projector_affine = Affine {
        normal: vector![-1.0, -1.0, 0., 0., 0., 0., 0., 0., 0.],
        distance: 18.,
    };

    // Two (or more!) projectors can be bundled together, saving a pair of slack/dual variables.
    // This should only be done if one constraint cannot immediately invalidate another constrant.
    let x_projector_bundle = (x_project_sphere, x_projector_affine);

    // We can also iteratively project a bundle if they could push each other out of feasible region
    // let x_projector_bundle = [&x_projector_bundle; 10];

    let u_projector_sphere = Sphere {
        center: vector![Some(0.0), Some(0.0), Some(0.0)],
        radius: 10.0,
    };

    let u_projector_box = Box {
        upper: vector![Some(2.0), Some(2.0), Some(2.0)],
        lower: vector![Some(-2.0), Some(-2.0), Some(-2.0)],
    };

    let u_projector_bundle = (u_projector_sphere, u_projector_box);

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

        let (reason, mut u_now) = mpc
            .initial_condition(x_now)
            .u_constraints(u_con.as_mut())
            .x_constraints(x_con.as_mut())
            .x_reference(xref.as_view())
            .solve();

        total_iters += mpc.get_num_iters();

        let xmatrix = mpc.get_x_matrix() + xref;

        let mut iteration_cost = 0.0;
        for col in xmatrix.column_iter() {
            iteration_cost += (col.transpose() * SMatrix::from_diagonal(&Q) * col)[0];
        }

        for col in mpc.get_u_matrix().column_iter() {
            iteration_cost += (col.transpose() * SMatrix::from_diagonal(&R) * col)[0];
        }

        total_cost += iteration_cost;

        println!(
            "Got solution: {:?} in {} ms) in {} iters ({reason:?}), cost: {}",
            u_now.as_slice(),
            time.elapsed().as_micros() as f32 / 1e3,
            mpc.get_num_iters(),
            iteration_cost,
        );

        // Apply input to system
        u_projector_sphere.project(u_now.as_view_mut());
        x_now = A * x_now + B * u_now;

        // ------ RERUN VISUALIZATION -------

        rec.set_time("timeline", Duration::from_millis((50 * k) as u64));

        let u_mat = mpc.get_u_matrix().clone();
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

        let mut x_mat = mpc.get_x_matrix().clone() + xref;
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

        x_projector_bundle.project(x_mat.as_view_mut());

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

        let vec = mpc.get_x_matrix().column(0) + xref.column(0);
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
            (mpc.get_x_matrix() + xref)
                .column_iter()
                .map(|vec| [vec[0] as f32, vec[1] as f32, vec[2] as f32]),
        );

        let strips = rerun::LineStrips3D::new([pos_pred_strips]).with_colors([Color::WHITE]);
        rec.log("x_position_pred", &strips).unwrap();
    }

    println!("Total iterations: {total_iters}, cost: {total_cost}");

    Ok(())
}
