#![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

    Initial work for this will consist of porting TinyMPC to Rust with no_std compatability.

*/

use nalgebra::{SMatrix, SVector, Scalar, SimdValue, RealField};

#[derive(Debug)]
pub struct TinySolver<const Nx: usize, const Nu: usize, const Nz: usize, const Nh: usize, F> {
    pub settings: TinySettings<F>,          // Problem settings
    pub cache: TinyCache<Nx,Nu,Nz,F>,       // Problem cache
    pub work: TinyWorkspace<Nx,Nu,Nz,Nh,F>, // Solver workspace
}

#[derive(Debug)]
pub struct TinySettings<F> {
    rho: F,
    prim_tol: F,
    dual_tol: F,
    max_iter: usize,
    check_every: usize,
}

/// Contains all pre-computed values
#[derive(Debug)]
pub struct TinyCache<const Nx: usize, const Nu: usize, const Nz: usize, F> {

    /// Infinite-time horizon LQR gain
    Kinf: SMatrix<F,Nu,Nx>,

    /// Infinite-time horizon LQR Hessian
    Pinf: SMatrix<F,Nx,Nx>,

    /// Precomputed `inv(R + B^T * Pinf * B)`
    RpBPB_i: SMatrix<F,Nu,Nu>,

    /// Precomputed `inv(A - B * Kinf)^T`
    AmBKt: SMatrix<F,Nx,Nx>,
}

#[derive(Debug)]
pub struct TinyWorkspace<const Nx: usize, const Nu: usize, const Nz: usize, const Nh: usize, F> {

    // Linear state space model
    A: SMatrix<F,Nx,Nx>,
    B: SMatrix<F,Nx,Nu>,
    _C: SMatrix<F,Nx,Nz>,

    // State and inputs
    x: SMatrix<F,Nx,Nh>,
    u: SMatrix<F,Nu,Nh>,

    // State and input constraints
    x_bound: Option<(SMatrix<F,Nx,Nh>,SMatrix<F,Nx,Nh>)>,
    u_bound: Option<(SMatrix<F,Nu,Nh>,SMatrix<F,Nu,Nh>)>,

    // Linear cost matrices
    q: SMatrix<F,Nx,Nh>,
    r: SMatrix<F,Nu,Nh>,

    // Linear cost vectors
    Q: SVector<F,Nx>,
    _R: SVector<F,Nu>,

    // Riccati backward pass terms
    p: SMatrix<F,Nx,Nh>,
    d: SMatrix<F,Nu,Nh>,

    // Auxiliary variables
    v: SMatrix<F,Nx,Nh>,
    vnew: SMatrix<F,Nx,Nh>,
    z: SMatrix<F,Nu,Nh>,
    znew: SMatrix<F,Nu,Nh>,

    // Dual variables
    g: SMatrix<F,Nx,Nh>,
    y: SMatrix<F,Nu,Nh>,

    // Number of iterations for latest solve
    iter: usize,
}

impl <const Nx: usize, const Nu: usize, const Nz: usize, const Nh: usize, F> TinySolver<Nx,Nu,Nz,Nh,F>
where F: Scalar + SimdValue + RealField + Copy
{

    /// Creates a new [`TinySolver<Nx, Nu, Nz, Nh, F>`].
    ///
    /// ## Arguments
    /// - `Kinf`: Infinite-horizon LQR gain matrix
    /// - `Ping`: Infinite-horizon LQR Hessian matrix
    /// - `A`: State space propagation matrix
    /// - `B`: State space input matrix
    /// - `C`: State space controlled outputs matrix
    /// - `Q`: State penalty vector
    /// - `R`: Input penalty vector
    ///
    /// Important note about `C`
    #[must_use]
    pub fn new(
        Kinf:SMatrix<F, Nu, Nx>,
        Pinf:SMatrix<F, Nx, Nx>,
        A:SMatrix<F,Nx,Nx>,
        B:SMatrix<F,Nx,Nu>,
        C:SMatrix<F,Nx,Nz>,
        Q:SVector<F, Nx>,
        R:SVector<F, Nu>
    ) -> Self {
        // Many things need to be passed into here

        let mut q_mat = SMatrix::<F, Nx, Nh>::zeros();
        let mut r_mat = SMatrix::<F, Nu, Nh>::zeros();

        for i in 0..Nh {
            q_mat.set_column(i, &Q);
            r_mat.set_column(i, &R);
        }

        Self {
            settings: TinySettings {
                rho: F::one(),
                prim_tol: F::from_f32(1e-3).unwrap(),
                dual_tol: F::from_f32(1e-3).unwrap(),
                max_iter: 50,
                check_every: 4,
            },
            cache: TinyCache {
                Kinf,
                Pinf,
                RpBPB_i: (SMatrix::from_diagonal(&R) + B.transpose()*Pinf*B).try_inverse().unwrap(),
                AmBKt: (A-B*Kinf).transpose(),
            },
            work: TinyWorkspace {
                x: SMatrix::zeros(),
                u: SMatrix::zeros(),
                q: q_mat,
                r: r_mat,
                p: SMatrix::zeros(),
                d: SMatrix::zeros(),
                v: SMatrix::zeros(),
                vnew: SMatrix::zeros(),
                z: SMatrix::zeros(),
                znew: SMatrix::zeros(),
                g: SMatrix::zeros(),
                y: SMatrix::zeros(),
                Q,
                _R: R,
                A,
                B,
                _C: C,
                u_bound: None,
                x_bound: None,
                iter: 0,
            },
        }
    }

    pub fn reset_dual_variables(&mut self)  {
        self.work.y = SMatrix::zeros();
        self.work.g = SMatrix::zeros();
    }

    /// Get the system state `x` for the current time
    pub fn get_x_at(&self,i: usize) -> SVector<F,Nx> {
        self.work.x.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u_at(&self,i: usize) -> SVector<F,Nu> {
        self.work.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<F,Nu> {
        self.get_u_at(0)
    }
    
    /// # Solve for the optimal MPC solution
    /// 
    /// This function contains the iterative solver which approximates the minimizing control input `u`.
    /// 
    /// Pass the initial condition state `x` , which may be determined using eg. a Kalman filter or observer.
    /// 
    /// If the solver succeeds in in minimizing, the resurned `Result` will be the `Ok` variant. If it fails to
    /// converge within `max_iter` iterations, it will return an `Err` to indicate such. The solution may still
    /// be okay to use, and can in any case be accessed through the `.get_u()` method, but there is no guarantee that
    /// it satisfies the constraints or is well-behaved for an `Err`. For control purposes, some control is better
    /// than no control. It is recommended to handle the case where there is no convergence, and either reconsider
    /// the system model, consraints or the values of `max_iter`, `abs_prim_tol` and `abs_dual_tol`.

    pub fn tiny_solve(&mut self, x: SVector<F,Nx>, xref: &SMatrix<F,Nx,Nh>) -> Result<(),SolveError> {

        // Shift previous predictions for more accurate initial values
        self.shift_predicted_inputs();

        // Iteratively solve MPC problem
        for i in 1..=self.settings.max_iter {
            self.work.iter = i;

            // Solve linear system with Riccati and roll out to get new trajectory
            self.forward_pass(x);

            // Project slack variables into feasible domain
            self.update_slack();

            // Compute next iteration of dual variables
            self.update_dual();

            // Update linear control cost terms using reference trajectory, duals, and slack variables
            self.update_linear_cost(xref);

            // Check for early-stop condition 
            if self.check_termination() {
                return Ok(())
            }

            //Backward pass to update Ricatti variables
            self.backward_pass_grad();

            // Save previous slack variables
            self.work.v = self.work.vnew;
            self.work.z = self.work.znew;

        }

        Err(SolveError::MaxIters)
    }

    /// Shift all previously predicted inputs forward in time
    fn shift_predicted_inputs(&mut self) {
        for i in 0..Nh-1 {
            self.work.u.swap_columns(i, i+1);
        }
        self.work.u.set_column(Nh-1, &self.work.u.column(Nh-2).clone_owned());
    }

    fn forward_pass(&mut self, x: SVector<F,Nx>) {

        // Forward-pass with initial state
        self.work.u.set_column(0, &(-self.cache.Kinf * x - self.work.d.column(0)));
        self.work.x.set_column(0, &(self.work.A * x + self.work.B * self.work.u.column(0)));

        // Forward-pass for rest of horizon
        for i in 1..Nh {
            self.work.u.set_column(i, &(-self.cache.Kinf * self.work.x.column(i-1) - self.work.d.column(i)));
            self.work.x.set_column(i, &(self.work.A * self.work.x.column(i-1) + self.work.B * self.work.u.column(i)));

        }
    }

    fn update_slack(&mut self) {

        self.work.znew = self.work.y + self.work.u;
        self.work.vnew = self.work.g + self.work.x;

        // TODO - Support more complicated constraints than min-max on all states/inputs simultaniously

        // Box constraints on input
        if let Some((u_min,u_max)) = self.work.u_bound {
            self.work.znew.zip_zip_apply(&u_min,&u_max, |u,min, max|{ *u = (*u).clamp(min, max) });
        }

        // Box constraints on state
        if let Some((x_min,x_max)) = self.work.x_bound {
            self.work.vnew.zip_zip_apply(&x_min,&x_max, |x,min, max|{ *x = (*x).clamp(min, max) });
        }
    }

    fn update_dual(&mut self) {
        // Gadient ascent
        self.work.y = self.work.y + self.work.u - self.work.znew;
        self.work.g = self.work.g + self.work.x - self.work.vnew;
    }

    fn update_linear_cost(&mut self, xref: &SMatrix<F,Nx,Nh>) {
        self.work.r = - (self.work.znew - self.work.y).scale(self.settings.rho);

        xref.column_iter().enumerate().for_each(|(i,x)| {
            self.work.q.set_column(i, &x.component_mul(&self.work.Q))
        });

        self.work.q -= (self.work.vnew - self.work.g).scale(self.settings.rho);

        self.work.p.set_column(Nh - 1, &(-(xref.column(Nh - 1).transpose() * self.cache.Pinf).transpose() - (self.work.vnew.column(Nh - 1).scale(self.settings.rho) - self.work.g.column(Nh - 1))));
    }

    fn check_termination(&mut self) -> bool {

        // Only make this check every `check_termination` iterations
        if self.work.iter % self.settings.check_every == 0 {

            // Calculate residuals
            let prim_residual_state = (self.work.x - self.work.vnew).abs().max();
            let dual_residual_state = (self.work.v - self.work.vnew).abs().max() * self.settings.rho;
            let prim_residual_input = (self.work.u - self.work.znew).abs().max();
            let dual_residual_input = (self.work.z - self.work.znew).abs().max() * self.settings.rho;

            // If all residuals are below tolerance, we terminate
            prim_residual_state < self.settings.prim_tol &&
            prim_residual_input < self.settings.prim_tol &&
            dual_residual_state < self.settings.dual_tol &&
            dual_residual_input < self.settings.dual_tol

        } else { false }
    }

    fn backward_pass_grad(&mut self) {
        for i in (1..Nh).rev() {
            self.work.d.set_column(i - 1, &(self.cache.RpBPB_i * (self.work.B.transpose() * self.work.p.column(i) + self.work.r.column(i))));
            self.work.p.set_column(i - 1, &(self.work.q.column(i) + self.cache.AmBKt * self.work.p.column(i) - (self.cache.Kinf.transpose() * self.work.r.column(i))));
        }
    }
}

#[derive(Debug)]
pub enum SolveError {
    MaxIters
}