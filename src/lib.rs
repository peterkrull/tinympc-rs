#![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

    Initial work for this will consist of porting TinyMPC to Rust with no_std compatability.

*/

use nalgebra::{SMatrix, SVector, Scalar, SimdValue, RealField, convert};

#[derive(Debug)]
pub struct TinySolver<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    pub settings: TinySettings<F>,
    pub cache: TinyCache<Nx,Nu,F>,
    pub work: TinyWorkspace<Nx,Nu,Hx,Hu,F>,
}

#[derive(Debug)]
pub struct TinySettings<F> {
    pub rho: F,
    pub prim_tol: F,
    pub dual_tol: F,
    pub max_iter: usize,
    pub check_termination: usize,
}

/// Contains all pre-computed values
#[derive(Debug)]
pub struct TinyCache<const Nx: usize, const Nu: usize, F> {

    /// Infinite-time horizon LQR gain
    pub Klqr: SMatrix<F,Nu,Nx>,

    /// Infinite-time horizon LQR Hessian
    pub Plqr: SMatrix<F,Nx,Nx>,

    /// Precomputed `inv((R + I*rho) + B^T * Plqr * B)`
    pub RpBPBi: SMatrix<F,Nu,Nu>,

    /// Precomputed `(A - B * Klqr)^T`
    pub AmBKt: SMatrix<F,Nx,Nx>,
}

#[derive(Debug)]
pub struct TinyWorkspace<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {

    // Linear state space model
    A: SMatrix<F,Nx,Nx>,
    B: SMatrix<F,Nx,Nu>,

    // State and inputs
    x: SMatrix<F,Nx,Hx>,
    u: SMatrix<F,Nu,Hu>,

    // State and input constraints (min,max)
    x_bound: Option<(SMatrix<F,Nx,Hx>,SMatrix<F,Nx,Hx>)>,
    u_bound: Option<(SMatrix<F,Nu,Hu>,SMatrix<F,Nu,Hu>)>,

    // Linear cost matrices
    q: SMatrix<F,Nx,Hx>,
    r: SMatrix<F,Nu,Hu>,

    // Linear state and input cost vector
    Q: SVector<F,Nx>,
    R: SVector<F,Nu>,

    // Riccati backward pass terms
    p: SMatrix<F,Nx,Hx>,
    d: SMatrix<F,Nu,Hu>,

    // Auxiliary variables
    v: SMatrix<F,Nx,Hx>,
    vnew: SMatrix<F,Nx,Hx>,
    z: SMatrix<F,Nu,Hu>,
    znew: SMatrix<F,Nu,Hu>,

    // Dual variables
    g: SMatrix<F,Nx,Hx>,
    y: SMatrix<F,Nu,Hu>,

    // Number of iterations for latest solve
    iter: usize,
}

impl <const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> TinySolver<Nx,Nu,Hx,Hu,F>
where F: Scalar + SimdValue + RealField + Copy
{

    // TODO: Add constraint to ensure Hu <= Hx
    // This is not yet possible in stable Rust

    #[must_use]
    pub fn new(
        A: SMatrix<F,Nx,Nx>,
        B: SMatrix<F,Nx,Nu>,
        Q: SVector<F, Nx>,
        R: SVector<F, Nu>,
        rho : F
    ) -> Option<Self> {
        let Qaug = Q.add_scalar(rho);
        let Raug = R.add_scalar(rho);

        // Guard against incorrect horiozon lengths
        if Hx <= Hu {
            return None
        }

        let (K, P) = lqr(&A, &B, &Qaug, &Raug, 1000)?;

        Some(Self {
            settings: TinySettings {
                rho,
                prim_tol: convert(1e-3),
                dual_tol: convert(1e-3),
                max_iter: 100,
                check_termination: 10
            },
            cache: TinyCache {
                Klqr: K,
                Plqr: P,
                RpBPBi: (SMatrix::from_diagonal(&Raug) + B.transpose()*P*B).try_inverse().unwrap(),
                AmBKt: (A-B*K).transpose(),
            },
            work: TinyWorkspace {
                A,
                B,
                Q: Qaug,
                R: Raug,
                x: SMatrix::zeros(),
                u: SMatrix::zeros(),
                q: SMatrix::zeros(),
                r: SMatrix::zeros(),
                p: SMatrix::zeros(),
                d: SMatrix::zeros(),
                v: SMatrix::zeros(),
                vnew: SMatrix::zeros(),
                z: SMatrix::zeros(),
                znew: SMatrix::zeros(),
                g: SMatrix::zeros(),
                y: SMatrix::zeros(),
                u_bound: None,
                x_bound: None,
                iter: 0,
            }
        })
    }

    pub fn tiny_solve(
        &mut self,
        x: SVector<F,Nx>,
        xref: &SMatrix<F,Nx,Hx>,
        uref: &SMatrix<F,Nu,Hu>,
    ) -> (TerminationReason, SVector<F,Nu>) {
        let mut termination_reason = TerminationReason::MaxIters;

        // Iteratively solve MPC problem
        for i in 0..self.settings.max_iter {

            // Roll out to get new trajectory
            self.forward_pass(x);

            // Project slack variables into feasible domain
            self.update_slack();

            // Compute next iteration of dual variables
            self.update_dual();

            // Update linear control cost terms
            self.update_linear_cost(xref, uref);

            // Backward pass to update Ricatti variables
            self.backward_pass_grad();
            
            // Check for early-stop condition 
            if self.check_termination(i) {
                termination_reason = TerminationReason::Converged;
                self.work.iter = i + 1;
                break;
            }
        }

        if let TerminationReason::MaxIters = termination_reason { self.work.iter = self.settings.max_iter; }

        (termination_reason, self.get_u_at(0))
    }

    /// Use LQR feedback policy to roll out trajectory
    fn forward_pass(&mut self, x: SVector<F,Nx>) {
        // Forward-pass with initial state
        self.work.u.set_column(0, &(-self.cache.Klqr * x - self.work.d.column(0)));
        self.work.x.set_column(0, &(self.work.A * x + self.work.B * self.work.u.column(0)));

        // Roll out trajectory up to the control horizon Hu
        for i in 1..Hu {
            self.work.u.set_column(i, &(-self.cache.Klqr * self.work.x.column(i-1) - self.work.d.column(i)));
            self.work.x.set_column(i, &(self.work.A * self.work.x.column(i-1) + self.work.B * self.work.u.column(i)));
        }

        // For the rest of the prediction horizon (Hx), hold the last control input constant
        if Hu < Hx {
            let u_final = self.work.u.column(Hu - 1).clone_owned();
            for i in Hu..Hx {
                self.work.x.set_column(i, &(self.work.A * self.work.x.column(i-1) + self.work.B * u_final));
            }
        }
    }

    /// Project slack (auxiliary) variables into their feasible domain
    fn update_slack(&mut self) {
        self.work.znew = self.work.y + self.work.u;
        self.work.vnew = self.work.g + self.work.x;

        if let Some((u_min,u_max)) = &self.work.u_bound {
            self.work.znew.zip_zip_apply(u_min, u_max, |u, min, max| *u = u.clamp(min, max));
        }

        if let Some((x_min,x_max)) = &self.work.x_bound {
            self.work.vnew.zip_zip_apply(x_min, x_max, |x, min, max| *x = x.clamp(min, max));
        }
    }

    /// Update next iteration of dual variables
    fn update_dual(&mut self) {
        self.work.y += self.work.u - self.work.znew;
        self.work.g += self.work.x - self.work.vnew;
    }

    /// Update linear control cost terms
    fn update_linear_cost(
        &mut self,
        xref: &SMatrix<F,Nx,Hx>,
        uref: &SMatrix<F,Nu,Hu>,
    ) {
        // Input cost (up to Hu)
        uref.column_iter().enumerate().for_each(|(i,uref)| {
            self.work.r.set_column(i, &(-uref.component_mul(&self.work.R)))
        });
        self.work.r += (self.work.y - self.work.znew).scale(self.settings.rho);

        // State cost (up to Hx)
        xref.column_iter().enumerate().for_each(|(i,xref)| {
            self.work.q.set_column(i, &(-xref.component_mul(&self.work.Q)))
        });
        self.work.q += (self.work.g - self.work.vnew).scale(self.settings.rho);

        // Terminal condition at the end of the prediction horizon Hx
        let q_f = -self.cache.Plqr * xref.column(Hx - 1);
        let admm_term_f = (self.work.g.column(Hx - 1) - self.work.vnew.column(Hx - 1)).scale(self.settings.rho);
        self.work.p.set_column(Hx - 1, &(q_f + admm_term_f));
    }

    /// Update linear terms from Riccati backward pass
    fn backward_pass_grad(&mut self) {
        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx-1).rev() {
            let p_next = self.work.p.column(i + 1);

            // Control action is only optimized up to Hu
            if i < Hu {
                let r_curr = self.work.r.column(i);
                let d_val = self.cache.RpBPBi * (self.work.B.transpose() * p_next + r_curr);
                self.work.d.set_column(i, &d_val);
                let p_val = self.work.q.column(i) + self.cache.AmBKt * p_next - self.cache.Klqr.transpose() * r_curr;
                self.work.p.set_column(i, &p_val);
            } else {
                // Beyond Hu, there is no control input cost 'r' and no feedforward 'd'
                let p_val = self.work.q.column(i) + self.cache.AmBKt * p_next;
                self.work.p.set_column(i, &p_val);
            }
        }
    }

    /// Check for termination condition by evaluating residuals
    fn check_termination(&mut self, current_iter: usize) -> bool {
        let mut do_terminate = false;

        if current_iter > 0 && current_iter % self.settings.check_termination == 0 {
            let prim_residual_state = (self.work.x - self.work.vnew).abs().max();
            let dual_residual_state = (self.work.v - self.work.vnew).abs().max() * self.settings.rho;
            let prim_residual_input = (self.work.u - self.work.znew).abs().max();
            let dual_residual_input = (self.work.z - self.work.znew).abs().max() * self.settings.rho;

            do_terminate =
                prim_residual_state < self.settings.prim_tol &&
                prim_residual_input < self.settings.prim_tol &&
                dual_residual_state < self.settings.dual_tol &&
                dual_residual_input < self.settings.dual_tol;
        }

        self.work.v = self.work.vnew;
        self.work.z = self.work.znew;

        do_terminate
    }


    /// Set or un-set varying min-max bounds on inputs for entire horizon
    pub fn set_u_bounds(&mut self, u_bound: Option<(SMatrix<F,Nu,Hu>,SMatrix<F,Nu,Hu>)>) {
        self.work.u_bound = u_bound;
    }

    /// Set or un-set varying min-max bounds on states for entire horizon
    pub fn set_x_bounds(&mut self, x_bound: Option<(SMatrix<F,Nx,Hx>,SMatrix<F,Nx,Hx>)>) {
        self.work.x_bound = x_bound;
    }

    /// Set or un-set the constant min-max bounds on inputs for entire horizon
    pub fn set_const_u_bounds(&mut self, u_bound: Option<(SVector<F,Nu>,SVector<F,Nu>)>) {
        if let Some((vec_min,vec_max)) = u_bound {
            let mut min: SMatrix<F,Nu,Hu> = SMatrix::zeros();
            let mut max: SMatrix<F,Nu,Hu> = SMatrix::zeros();
            
            for i in 0..Hu {
                min.set_column(i, &vec_min);
                max.set_column(i, &vec_max);
            }
            self.work.u_bound = Some((min,max));
        } else {
            self.work.u_bound = None
        }
    }

    /// Set or un-set the constant min-max bounds on states for entire horizon
    pub fn set_const_x_bounds(&mut self, x_bound: Option<(SVector<F,Nx>,SVector<F,Nx>)>) {
        if let Some((vec_min,vec_max)) = x_bound {
            let mut min: SMatrix<F,Nx,Hx> = SMatrix::zeros();
            let mut max: SMatrix<F,Nx,Hx> = SMatrix::zeros();
            
            for i in 0..Hx {
                min.set_column(i, &vec_min);
                max.set_column(i, &vec_max);
            }
            self.work.x_bound = Some((min,max));
        } else {
            self.work.x_bound = None
        }
    }

    pub fn reset_dual_variables(&mut self)  {
        self.work.y = SMatrix::zeros();
        self.work.g = SMatrix::zeros();
    }

    pub fn get_num_iters(&self) -> usize {
        self.work.iter
    }

    /// Get the system state `x` for the time `i`
    pub fn get_x_at(&self,i: usize) -> SVector<F,Nx> {
        self.work.x.column(i).into()
    }

    /// Get the system input `u` for the time `i`
    pub fn get_u_at(&self,i: usize) -> SVector<F,Nu> {
        self.work.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<F,Nu> {
        self.get_u_at(0)
    }
    
    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<F,Nx,Hx> {
        &self.work.x
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<F,Nu,Hu> {
        &self.work.u
    }
    
    pub fn prediction_horizon_length(&self) -> usize { Hx }
    pub fn control_horizon_length(&self)  -> usize { Hu }
    pub fn num_states(&self) -> usize { Nx }
    pub fn num_inputs(&self) -> usize { Nu }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TerminationReason {
    Converged,
    MaxIters
}

pub fn lqr<T: RealField + Copy, const Nx: usize, const Nu: usize>(
    A: &SMatrix<T, Nx, Nx>,
    B: &SMatrix<T, Nx, Nu>,
    Qaug: &SVector<T, Nx>,
    Raug: &SVector<T, Nu>,
    iters: usize
) -> Option<(SMatrix<T, Nu, Nx>, SMatrix<T, Nx, Nx>)> {
    let mut K = SMatrix::zeros();
    let mut P = SMatrix::from_diagonal(&Qaug);

    for _ in 0..iters {
        K = (SMatrix::from_diagonal(&Raug) + B.transpose()*P*B).try_inverse()?*(B.transpose()*P*A);
        P = A.transpose()*P*A - A.transpose()*P*B*K + SMatrix::from_diagonal(&Qaug);
    }

    if !K.iter().all(|x| x.is_finite()) || !P.iter().all(|x| x.is_finite()) {
        return None;
    }

    Some((K, P))
}