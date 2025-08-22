use nalgebra::{matrix, vector, SMatrix, SVector, SVectorView};
use rerun::Color;
use tinympc_rs::{constraint::{Box, Project as _, ProjectExt as _, Sphere}, Error};

const HX: usize = 140;
const HU: usize = HX - 5;

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
    ].into()
}

pub static Q: SVector<f32, NX> = vector! {200., 200., 200., 0., 0., 0., 0., 0., 0.};
pub static R: SVector<f32, NU> = vector! {5.0, 5.0, 5.0};
pub static RHO: f32 = 5.0;

fn main() -> Result<(), Error> {

    let rec = rerun::RecordingStreamBuilder::new("tinympc-constraints")
        .spawn()
        .unwrap();

    let mut mpc = tinympc_rs::TinyMpc::<NX, NU, HX, HU, f32>::new(A, B, Q, R, RHO)?;
    mpc.config.max_iter = 25;
    mpc.config.do_check = 5;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut xnow = SVector::zeros();
    let mut xref = SMatrix::<f32, NX, HX>::zeros();

    let xcon_sphere = Sphere {
        center: vector![None, None, None, Some(0.0), Some(0.0), Some(0.0), None, None, None],
        radius: 2.0
    };

    let mut xcon = [&mut xcon_sphere.into_dyn_constraint()];

    let ucon_sphere = Sphere {
        center: vector![Some(0.0), Some(0.0), Some(0.0)],
        radius: 2.5,
    };

    let mut ucon = [&mut ucon_sphere.into_dyn_constraint()];

    let mut true_pos = vec![vector![0.0, 0.0, 0.0]];

    let mut k = 0;
    loop {
        k += 1;
        let time = std::time::Instant::now();
        let (reason, mut unow) = mpc.solve(xnow, Some(xref.as_view()), None, Some(&mut xcon), Some(&mut ucon));
        println!("Got solution: {:?} in {} ms)", unow.as_slice(), time.elapsed().as_micros() as f32 / 1e3);

        std::thread::sleep(std::time::Duration::from_millis(16));

        for i in 0..HX {
            let mut xref_col = SVector::zeros();
            xref_col[0] = ((i + k) as f32 / 10.0 / (1.0 + (i + k) as f32 / 1500.)).sin() * 4.;
            xref_col[1] = ((i + k) as f32 / 10.0 / (1.0 + (i + k) as f32 / 1500.)).cos() * 4.;
            xref_col[2] = (i + k) as f32 / 100.0 ;

            if k + i > 400 && i + k < 1200 {
                xref_col[0] += -20.0;
            }

            if k + i > 600 && i + k < 1500 {
                xref_col[1] += -15.0;
            }


            xref.set_column(i, &xref_col);
        }

        match reason {
            tinympc_rs::TerminationReason::Converged => println!("Converged in {} iters", mpc.get_num_iters()),
            tinympc_rs::TerminationReason::MaxIters => println!("Reached max ({}) iters", mpc.get_num_iters()),
        }

        // Apply input to system
        ucon_sphere.project(unow.as_view_mut());
        xnow = A * xnow + B * unow;
        println!("State vector: {:?}", xnow.as_slice());









        // ------ RERUN VISUALIZATION -------




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

        xcon_sphere.project(x_mat.as_view_mut());

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


        let vec = mpc.get_x_at(0);
        rec.log(
            "ed_point",
            &rerun::Points3D::new([[vec[0] as f32, vec[1] as f32, vec[2] as f32]])
                .with_radii([0.1]),
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
            xref
                .column_iter()
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
}
