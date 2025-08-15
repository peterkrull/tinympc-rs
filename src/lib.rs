#![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

    Initial work for this will consist of porting TinyMPC to Rust with no_std compatability.

*/

use nalgebra::{SMatrix, SVector, Scalar, SimdValue, RealField, convert};

#[derive(Debug)]
pub struct TinySolver<const Nx: usize, const Nu: usize, const Nh: usize, F> {
    pub settings: TinySettings<F>,          // Problem settings
    pub cache: TinyCache<Nx,Nu,F>,       // Problem cache
    pub work: TinyWorkspace<Nx,Nu,Nh,F>, // Solver workspace
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
pub struct TinyWorkspace<const Nx: usize, const Nu: usize, const Nh: usize, F> {

    // Linear state space model
    A: SMatrix<F,Nx,Nx>,
    B: SMatrix<F,Nx,Nu>,

    // State and inputs
    x: SMatrix<F,Nx,Nh>,
    u: SMatrix<F,Nu,Nh>,

    // State and input constraints (min,max)
    x_bound: Option<(SMatrix<F,Nx,Nh>,SMatrix<F,Nx,Nh>)>,
    u_bound: Option<(SMatrix<F,Nu,Nh>,SMatrix<F,Nu,Nh>)>,

    // Linear cost matrices
    q: SMatrix<F,Nx,Nh>,
    r: SMatrix<F,Nu,Nh>,

    // Linear state and input cost vector
    Q: SVector<F,Nx>,
    R: SVector<F,Nu>,

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

impl <const Nx: usize, const Nu: usize, const Nh: usize, F> TinySolver<Nx,Nu,Nh,F>
where F: Scalar + SimdValue + RealField + Copy
{

    /// Creates a new [`TinySolver<Nx, Nu, Nz, Nh, F>`].
    ///
    /// ## Arguments
    /// - `A`: State space propagation matrix
    /// - `B`: State space input matrix
    /// - `Q`: State penalty vector
    /// - `R`: Input penalty vector
    ///
    /// Important note about `C`
    #[must_use]
    pub fn new(
        A: SMatrix<F,Nx,Nx>,
        B: SMatrix<F,Nx,Nu>,
        Q: SVector<F, Nx>,
        R: SVector<F, Nu>,
        rho : F
    ) -> Option<Self> {
        // Many things need to be passed into here

        let Qaug = Q.add_scalar(rho);
        let Raug = R.add_scalar(rho);

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

    /// # Solve for the optimal MPC solution
    /// 
    /// This function contains the iterative solver which approximates the minimizing control input `u`.
    /// 
    /// Pass the initial condition state `x` , which may be determined using eg. a Kalman filter or observer.
    /// 
    /// The function returns;
    /// - the reason for termination, either being convergence or maximum number of iterations
    /// - the optimal actuation `u` to apply to the system 
    ///
    pub fn tiny_solve(
        &mut self, 
        x: SVector<F,Nx>, 
        xref: &SMatrix<F,Nx,Nh>,
        uref: &SMatrix<F,Nu,Nh>,
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

        // Forward-pass for rest of horizon
        for i in 1..Nh {
            self.work.u.set_column(i, &(-self.cache.Klqr * self.work.x.column(i-1) - self.work.d.column(i)));
            self.work.x.set_column(i, &(self.work.A * self.work.x.column(i-1) + self.work.B * self.work.u.column(i)));
        }
    }

    /// Project slack (auxiliary) variables into their feasible domain, defined by projection functions related to each constraint
    fn update_slack(&mut self) {

        self.work.znew = self.work.y + self.work.u;
        self.work.vnew = self.work.g + self.work.x;

        // Box constraints on input
        if let Some((u_min,u_max)) = self.work.u_bound {
            self.work.znew.zip_zip_apply(&u_min,&u_max, |u,min, max|{ *u = (*u).clamp(min, max) });
        }

        // Box constraints on state
        if let Some((x_min,x_max)) = self.work.x_bound {
            self.work.vnew.zip_zip_apply(&x_min,&x_max, |x,min, max|{ *x = (*x).clamp(min, max) });
        }
    }

    /// Update next iteration of dual variables by performing the augmented lagrangian multiplier update
    fn update_dual(&mut self) {
        // Gadient ascent
        self.work.y += self.work.u - self.work.znew;
        self.work.g += self.work.x - self.work.vnew;
    }

    /// Update linear control cost terms in the Riccati feedback using the changing slack and dual variables from ADMM
    fn update_linear_cost(
        &mut self,
        xref: &SMatrix<F,Nx,Nh>, 
        uref: &SMatrix<F,Nu,Nh>,
    ) {
        // Input cost
        uref.column_iter().enumerate().for_each(|(i,uref)| {
            self.work.r.set_column(i, &(-uref.component_mul(&self.work.R)))
        });
        self.work.r += (self.work.y - self.work.znew).scale(self.settings.rho);

        // State cost
        xref.column_iter().enumerate().for_each(|(i,xref)| {
            self.work.q.set_column(i, &(-xref.component_mul(&self.work.Q)))
        });
        self.work.q += (self.work.g - self.work.vnew).scale(self.settings.rho);

        // Terminal condition
        let q_f = -self.cache.Plqr * xref.column(Nh - 1);
        let admm_term_f = (self.work.g.column(Nh - 1) - self.work.vnew.column(Nh - 1)).scale(self.settings.rho);
        self.work.p.set_column(Nh - 1, &(q_f + admm_term_f));
    }

    /// Update linear terms from Riccati backward pass
    fn backward_pass_grad(&mut self) {
        for i in (0..Nh-1).rev() {
            let p_next = self.work.p.column(i + 1);
            let r_curr = self.work.r.column(i);
            
            let d_val = self.cache.RpBPBi * (self.work.B.transpose() * p_next + r_curr);
            self.work.d.set_column(i, &d_val);
            
            let p_val = self.work.q.column(i) + self.cache.AmBKt * p_next - self.cache.Klqr.transpose() * r_curr;
            self.work.p.set_column(i, &p_val);
        }
    }

    /// Check for termination condition by evaluating whether the largest absolute primal and dual residuals for states and inputs are below threhold.
    fn check_termination(&mut self, current_iter: usize) -> bool {

        let mut do_terminate = false;

        if self.work.iter <= self.settings.check_termination + 1 || current_iter % self.settings.check_termination == 0 {

            // Calculate residuals on slack variables
            let prim_residual_state = (self.work.x - self.work.vnew).abs().max();
            let dual_residual_state = (self.work.v - self.work.vnew).abs().max() * self.settings.rho;
            let prim_residual_input = (self.work.u - self.work.znew).abs().max();
            let dual_residual_input = (self.work.z - self.work.znew).abs().max() * self.settings.rho;


            // If all residuals are below tolerance, we terminate
            do_terminate =
                prim_residual_state < self.settings.prim_tol &&
                prim_residual_input < self.settings.prim_tol &&
                dual_residual_state < self.settings.dual_tol &&
                dual_residual_input < self.settings.dual_tol;
        }

        // Save previous slack variables
        self.work.v = self.work.vnew;
        self.work.z = self.work.znew;

        do_terminate
    }


    /// Set or un-set varying min-max bounds on inputs for entire horizon
    pub fn set_u_bounds(&mut self, u_bound: Option<(SMatrix<F,Nu,Nh>,SMatrix<F,Nu,Nh>)>) {
        self.work.u_bound = u_bound;
    }

    /// Set or un-set varying min-max bounds on states for entire horizon
    pub fn set_x_bounds(&mut self, x_bound: Option<(SMatrix<F,Nx,Nh>,SMatrix<F,Nx,Nh>)>) {
        self.work.x_bound = x_bound;
    }

    /// Set or un-set the constant min-max bounds on inputs for entire horizon
    pub fn set_const_u_bounds(&mut self, u_bound: Option<(SVector<F,Nu>,SVector<F,Nu>)>) {

        if let Some((vec_min,vec_max)) = u_bound {

            let mut min: SMatrix<F,Nu,Nh> = SMatrix::zeros();
            let mut max: SMatrix<F,Nu,Nh> = SMatrix::zeros();
            
            for i in 0..Nh {
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

            let mut min: SMatrix<F,Nx,Nh> = SMatrix::zeros();
            let mut max: SMatrix<F,Nx,Nh> = SMatrix::zeros();
            
            for i in 0..Nh {
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

    /// Get the system state `x` for the time `o`
    pub fn get_x_at(&self,i: usize) -> SVector<F,Nx> {
        self.work.x.column(i).into()
    }

    /// Get the system input `u` for the time `o`
    pub fn get_u_at(&self,i: usize) -> SVector<F,Nu> {
        self.work.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<F,Nu> {
        self.get_u_at(0)
    }
    
    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<F,Nx,Nh> {
        &self.work.x
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<F,Nu,Nh> {
        &self.work.u
    }
    
    pub fn horizon_length(&self)    -> usize { Nh }
    pub fn num_states(&self)        -> usize { Nx }
    pub fn num_inputs(&self)        -> usize { Nu }
}

#[derive(Debug)]
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

    // Ensure none of the entries are NaN or infinite
    if !K.iter().all(|x| x.is_finite()) || !P.iter().all(|x| x.is_finite()) {
        return None;
    }

    Some((K, P))
}