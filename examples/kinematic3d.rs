use std::time::Duration;

use nalgebra::{SMatrix, SVector, SVectorView, matrix, vector};
use rerun::Color;
use tinympc_rs::{
    constraint::{Box, Project as _, ProjectExt as _, Sphere}, Error, TinyMpc
};

const HX: usize = 100;
const HU: usize = HX - 15;

const NX: usize = 9;
const NU: usize = 3;

const DT: f32 = 0.1;
const DD: f32 = 0.5 * DT * DT;

const LP: f32 = 0.5;

pub static A: SMatrix<f32, NX, NX> = matrix![
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

pub static B: SMatrix<f32, NX, NU> = matrix![
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

fn sys(x: SVectorView<f32, NX>, u: SVectorView<f32, NU>) -> SVector<f32, NX> {
    [
        x[0] + x[3] * DT + x[6] * DD,
        x[1] + x[4] * DT + x[7] * DD,
        x[2] + x[5] * DT + x[8] * DD,
        x[3] + x[6] * DT,
        x[4] + x[7] * DT,
        x[5] + x[8] * DT,
        x[6] * LP + (1.0 - LP) * u[0],
        x[7] * LP + (1.0 - LP) * u[1],
        x[8] * LP + (1.0 - LP) * u[2],
    ]
    .into()
}

pub static Q: SVector<f32, NX> = vector! {9., 9., 9., 0., 0., 0., 0., 0., 0.};
pub static R: SVector<f32, NU> = vector! {3., 3., 3.};
pub static RHO: f32 = 1.0;

fn main() -> Result<(), Error> {
    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    type Cache = tinympc_rs::rho_cache::LookupCache<f32, NX, NU, 5>;
    // type Cache = tinympc_rs::rho_cache::SingleCache<f32, NX, NU>;
    type Mpc = TinyMpc<f32, Cache, NX, NU, HX, HU>;

    let mut mpc = Mpc::new(A, B, Q, R, RHO)?.with_sys(sys);
    mpc.config.max_iter = 5;
    mpc.config.do_check = 2;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut xnow = SVector::zeros();
    let mut xref = SMatrix::<f32, NX, HX>::zeros();

    #[rustfmt::skip]
    let x_project_sphere = Sphere {
        center: vector![None, None, None, Some(0.0), Some(0.0), Some(0.0), None, None, None],
        radius: 4.0,
    };

    #[rustfmt::skip]
    let x_project_box = Box {
        upper: vector![Some(50.0), Some(50.0), Some(50.0), None, None, None, None, None, None],
        lower: vector![Some(-50.0), Some(-50.0), Some(-50.0), None, None, None, None, None, None],
    };

    // Two (or more!) projectors can be used together, saving a pair of slack/dual variables.
    // This should only be done if one constraint cannot immediately invalidate another constrant.
    let x_projector = (x_project_sphere, x_project_box);
    let mut xcon = x_projector.dyn_constraint();

    let ucon_sphere = Sphere {
        center: vector![Some(0.0), Some(0.0), Some(0.0)],
        radius: 10.0,
    };

    let ucon_sphere_dyn = &mut ucon_sphere.dyn_constraint();

    let mut xcon = [&mut xcon];
    let mut ucon = [ucon_sphere_dyn];

    let mut true_pos = vec![vector![0.0, 0.0, 0.0]];

    let mut total_iters = 0;
    let mut k = 0;
    while k < 2000 {
        k += 1;

        for i in 0..HX {
            let mut xref_col = SVector::zeros();
            xref_col[0] = ((i + k) as f32 / 5.0 / (1.0 + (i + k) as f32 / 1000.)).sin() * 4.;
            xref_col[1] = ((i + k) as f32 / 5.0 / (1.0 + (i + k) as f32 / 1000.)).cos() * 4.;
            xref_col[2] = (i + k) as f32 / 100.0;

            if i + k > 400 && i + k < 1200 {
                xref_col[0] *= 2.0;
            }

            if i + k > 800 && i + k < 1500 {
                xref_col[1] += -50.0;
            }

            xref.set_column(i, &xref_col);
        }

        let time = std::time::Instant::now();
        let (reason, mut unow) = mpc.solve(
            xnow,
            Some(xref.as_view()),
            None,
            Some(&mut xcon),
            Some(&mut ucon),
        );
        println!(
            "Got solution: {:?} in {} ms)",
            unow.as_slice(),
            time.elapsed().as_micros() as f32 / 1e3
        );

        total_iters += mpc.get_num_iters();

        // std::thread::sleep(std::time::Duration::from_millis(16));

        match reason {
            tinympc_rs::TerminationReason::Converged => {
                println!("Converged in {} iters", mpc.get_num_iters())
            }
            tinympc_rs::TerminationReason::MaxIters => {
                println!("Reached max ({}) iters", mpc.get_num_iters())
            }
        }

        // Apply input to system
        ucon_sphere.project(unow.as_view_mut());
        xnow = A * xnow + B * unow;

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

        let mut x_mat = mpc.get_x_matrix().clone();
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

        x_projector.project(x_mat.as_view_mut());

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

        let x = xnow;

        let vec = mpc.get_x_matrix().column(0);
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
            mpc.get_x_matrix()
                .column_iter()
                .map(|vec| [vec[0] as f32, vec[1] as f32, vec[2] as f32]),
        );

        let strips = rerun::LineStrips3D::new([pos_pred_strips]).with_colors([Color::WHITE]);
        rec.log("x_position_pred", &strips).unwrap();
    }

    println!("Total iterations: {total_iters}");

    Ok(())
}
