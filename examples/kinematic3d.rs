use nalgebra::{matrix, vector, SMatrix, SVector};
use tinympc_rs::{constraint::{BoxFixed, Project as _, ProjectExt as _}, Error};

const HX: usize = 40;
const HU: usize = 30;

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

fn sys(x: SVector<f32, NX>, u: SVector<f32, NU>) -> SVector<f32, NX> {
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

pub static Q: SVector<f32, NX> = vector! {10., 10., 10., 0., 0., 0., 0., 0., 0.};
pub static R: SVector<f32, NU> = vector! {1.0, 1.0, 1.0};
pub static RHO: f32 = 1.0;

fn main() -> Result<(), Error> {

    let mut mpc = tinympc_rs::TinyMpc::<NX, NU, HX, HU, f32>::new(A, B, Q, R, RHO)?.with_sys(sys);
    mpc.config.max_iter = 200;
    mpc.config.do_check = 1;

    println!("Size of MPC object: {} bytes", core::mem::size_of_val(&mpc));

    let mut xnow = SVector::zeros();
    let mut xref = SMatrix::zeros();

    let xcon_box = BoxFixed::new()
        .with_lower([None, None, None, Some(-0.28), Some(-0.28), Some(-0.28), None, None, None])
        .with_upper([None, None, None, Some(0.28), Some(0.28), Some(0.28), None, None, None]);
    let mut xcon = [&mut xcon_box.into_constraint()];

    let ucon_box = BoxFixed::new()
        .with_lower([Some(-2.6), Some(-2.6), Some(-2.6)])
        .with_upper([Some(2.6), Some(2.6), Some(2.6)]);
    let mut ucon = [&mut ucon_box.into_constraint()];

    let mut k = 0;
    loop {
        k += 1;
        let time = std::time::Instant::now();
        let (reason, mut unow) = mpc.solve(xnow, Some(&xref), None, Some(&mut xcon), Some(&mut ucon));
        println!("Got solution: {:?} in {} ms)", unow.as_slice(), time.elapsed().as_micros() as f32 / 1e3);

        std::thread::sleep(std::time::Duration::from_millis(100));

        for i in 0..HX {
            let mut xref_col = SVector::zeros();
            xref_col[0] = ((i + k) as f32 / 200.0).sin() * 3.;
            xref_col[1] = ((i + k) as f32 / 200.0).cos() * 3.;

            // We do this check to ensure the reference is
            // consistently moving throughout the horizon
            if i < HX - 1 && k > 1 {
                assert_eq!(xref_col, xref.column(i+1))
            }

            xref.set_column(i, &xref_col);
        }

        match reason {
            tinympc_rs::TerminationReason::Converged => println!("Converged in {} iters", mpc.get_num_iters()),
            tinympc_rs::TerminationReason::MaxIters => println!("Reached max ({}) iters", mpc.get_num_iters()),
        }

        // Apply input to system
        println!("State vector: {:?}", xnow.as_slice());
        ucon_box.project(&mut unow);
        xnow = A * xnow + B * unow;
    }
}
