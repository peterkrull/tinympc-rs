#![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

*/

use nalgebra::{convert, RealField, SMatrix, SVector, Scalar, SimdValue};

#[derive(Debug)]
pub struct TinyMpc<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    pub config: Config<F>,
    pub cache: Cache<Nx, Nu, F>,
    pub state: State<Nx, Nu, Hx, Hu, F>,
}

#[derive(Debug)]
pub struct Config<F> {
    pub prim_tol: F,
    pub dual_tol: F,
    pub max_iter: usize,
    pub do_check: usize,
}

/// Contains all pre-computed values
#[derive(Debug)]
pub struct Cache<const Nx: usize, const Nu: usize, F> {
    /// Penalty-parameter for this cache
    pub rho: F,

    /// Augmented state penalty vector
    pub Q_aug: SVector<F, Nx>,

    /// Augmented input penalty vector
    pub R_aug: SVector<F, Nu>,

    /// Infinite-time horizon LQR gain
    pub Klqr: SMatrix<F, Nu, Nx>,

    /// Infinite-time horizon LQR Hessian
    pub Plqr: SMatrix<F, Nx, Nx>,

    /// Precomputed `inv((R + I*rho) + B^T * Plqr * B)`
    pub RpBPBi: SMatrix<F, Nu, Nu>,

    /// Precomputed `(A - B * Klqr)^T`
    pub AmBKt: SMatrix<F, Nx, Nx>,
}

/// Errors that can occur during system setup
#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidHorizonLengths,
    QNotPositiveSemidefinite,
    RNotPositiveDefinite,
    RpBPBNotInvertible,
    NonFiniteValues,
}

#[derive(Debug, PartialEq)]
pub enum TerminationReason {
    Converged,
    MaxIters,
}

impl<const Nx: usize, const Nu: usize, T> Cache<Nx, Nu, T>
where
    T: Scalar + SimdValue + RealField + Copy,
{
    fn compute(
        rho: T,
        iters: usize,
        A: &SMatrix<T, Nx, Nx>,
        B: &SMatrix<T, Nx, Nu>,
        Q: &SVector<T, Nx>,
        R: &SVector<T, Nu>,
    ) -> Result<Self, Error> {
        if !Q.iter().all(|q| q >= &T::zero()) {
            return Err(Error::QNotPositiveSemidefinite);
        }

        if !R.iter().all(|r| r > &T::zero()) {
            return Err(Error::RNotPositiveDefinite);
        }

        // Shadow the original values with the augmented variants
        let Q_aug = Q.add_scalar(rho);
        let R_aug = R.add_scalar(rho);

        let Q_diag = SMatrix::from_diagonal(&Q_aug);
        let R_diag = SMatrix::from_diagonal(&R_aug);

        let mut Klqr = SMatrix::zeros();
        let mut Plqr = Q_diag.clone_owned();

        for _ in 0..iters {
            Klqr = (R_diag + B.transpose() * Plqr * B).try_inverse().ok_or(Error::RpBPBNotInvertible)? * (B.transpose() * Plqr * A);
            Plqr = A.transpose() * Plqr * A - A.transpose() * Plqr * B * Klqr + Q_diag;
        }

        let RpBPBi = (R_diag + B.transpose() * Plqr * B).try_inverse().ok_or(Error::RpBPBNotInvertible)?;
        let AmBKt = (A - B * Klqr).transpose();

        ([].iter())
            .chain(RpBPBi.iter())
            .chain(AmBKt.iter())
            .all(|x| x.is_finite())
            .then(|| Cache {
                rho,
                Q_aug,
                R_aug,
                Klqr,
                Plqr,
                RpBPBi,
                AmBKt,
            })
            .ok_or(Error::NonFiniteValues)
    }
}

#[derive(Debug)]
pub struct State<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    // Linear state space model
    A: SMatrix<F, Nx, Nx>,
    B: SMatrix<F, Nx, Nu>,

    // State and inputs
    x: SMatrix<F, Nx, Hx>,
    u: SMatrix<F, Nu, Hu>,

    // State and input constraints (min,max)
    x_bound: Option<(SMatrix<F, Nx, Hx>, SMatrix<F, Nx, Hx>)>,
    u_bound: Option<(SMatrix<F, Nu, Hu>, SMatrix<F, Nu, Hu>)>,

    // Linear cost matrices
    x_cost: SMatrix<F, Nx, Hx>,
    u_cost: SMatrix<F, Nu, Hu>,

    // Linear state and input cost vector
    Q: SVector<F, Nx>,
    R: SVector<F, Nu>,

    // Riccati backward pass terms
    x_ricc: SMatrix<F, Nx, Hx>,
    u_ricc: SMatrix<F, Nu, Hu>,

    // Auxiliary variables
    x_slac: SMatrix<F, Nx, Hx>,
    u_slac: SMatrix<F, Nu, Hu>,

    // Dual variables
    x_dual: SMatrix<F, Nx, Hx>,
    u_dual: SMatrix<F, Nu, Hu>,

    // Number of iterations for latest solve
    iter: usize,
}

impl<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F>
    TinyMpc<Nx, Nu, Hx, Hu, F>
where
    F: Scalar + SimdValue + RealField + Copy,
{
    #[must_use]
    pub fn new(
        A: SMatrix<F, Nx, Nx>,
        B: SMatrix<F, Nx, Nu>,
        Q: SVector<F, Nx>,
        R: SVector<F, Nu>,
        rho: F,
    ) -> Result<Self, Error> {
        // Guard against invalid horizon lengths
        if Hx <= Hu || Hu == 0 {
            return Err(Error::InvalidHorizonLengths);
        }

        Ok(Self {
            config: Config {
                prim_tol: convert(1e-3),
                dual_tol: convert(1e-3),
                max_iter: 100,
                do_check: 10,
            },
            cache: Cache::compute(rho, 1000, &A, &B, &Q, &R)?,
            state: State {
                A,
                B,
                Q,
                R,
                x: SMatrix::zeros(),
                u: SMatrix::zeros(),
                x_cost: SMatrix::zeros(),
                u_cost: SMatrix::zeros(),
                x_ricc: SMatrix::zeros(),
                u_ricc: SMatrix::zeros(),
                x_slac: SMatrix::zeros(),
                u_slac: SMatrix::zeros(),
                x_dual: SMatrix::zeros(),
                u_dual: SMatrix::zeros(),
                u_bound: None,
                x_bound: None,
                iter: 0,
            },
        })
    }

    pub fn solve(
        &mut self,
        xnow: SVector<F, Nx>,
        xref: &SMatrix<F, Nx, Hx>,
        uref: &SMatrix<F, Nu, Hu>,
    ) -> (TerminationReason, SVector<F, Nu>) {
        let mut termination_reason = TerminationReason::MaxIters;

        // Better warm-starting of dual variables from prior solution
        self.shift_dual_variables();

        // Iteratively solve MPC problem
        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {
            self.state.iter += 1;

            let x_slac_old = self.state.x_slac.clone_owned();
            let u_slac_old = self.state.u_slac.clone_owned();

            // Update linear control cost terms
            self.update_cost(xref, uref);

            // Backward pass to update Ricatti variables
            self.backward_pass();

            // Roll out to get new trajectory
            self.forward_pass(xnow);

            // Project into feasible domain
            self.update_constraints();

            // Check for early-stop condition
            if self.check_termination(x_slac_old, u_slac_old) {
                termination_reason = TerminationReason::Converged;
                break;
            }
        }

        (termination_reason, self.get_u())
    }

    /// Shift the dual variables by one time step for more accurate hot starting
    fn shift_dual_variables(&mut self) {
        /// This method uses unsafe to memmove the columns, effectively shifting all columns over
        fn shift_left<F, const ROWS: usize, const COLS: usize>(
            matrix: &mut SMatrix<F, ROWS, COLS>,
        ) {
            if COLS > 1 {
                let element_count = ROWS * (COLS - 1);
                let ptr = matrix.as_mut_slice().as_mut_ptr();

                unsafe {
                    core::ptr::copy(ptr.add(ROWS), ptr, element_count);
                }
            }
        }

        shift_left(&mut self.state.u_dual);
        shift_left(&mut self.state.x_dual);
    }

    /// Update linear control cost terms
    fn update_cost(&mut self, xref: &SMatrix<F, Nx, Hx>, uref: &SMatrix<F, Nu, Hu>) {
        // Input cost (up to Hu)
        uref.column_iter().enumerate().for_each(|(i, uref)| {
            self.state
                .u_cost
                .set_column(i, &(-uref.component_mul(&self.state.R)))
        });
        self.state.u_cost += (self.state.u_dual - self.state.u_slac).scale(self.cache.rho);

        // State cost (up to Hx)
        xref.column_iter().enumerate().for_each(|(i, xref)| {
            self.state
                .x_cost
                .set_column(i, &(-xref.component_mul(&self.state.Q)))
        });
        self.state.x_cost += (self.state.x_dual - self.state.x_slac).scale(self.cache.rho);

        // Terminal condition at the end of the prediction horizon Hx
        let q_f = -self.cache.Plqr * xref.column(Hx - 1);
        let admm_term_f = (self.state.x_dual.column(Hx - 1) - self.state.x_slac.column(Hx - 1))
            .scale(self.cache.rho);
        self.state.x_ricc.set_column(Hx - 1, &(q_f + admm_term_f));
    }

    /// Update linear terms from Riccati backward pass
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = &self.cache;

        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx - 1).rev() {
            let x_ricc_next = s.x_ricc.column(i + 1);

            // Control action is only optimized up to Hu
            if i < Hu {
                let r_curr = s.u_cost.column(i);
                s.u_ricc.set_column(i, &(c.RpBPBi * (s.B.transpose() * x_ricc_next + r_curr)));
                s.x_ricc.set_column(i, &(s.x_cost.column(i) + c.AmBKt * x_ricc_next - c.Klqr.transpose() * r_curr));
            } else {
                let r_curr = s.u_cost.column(Hu - 1);
                s.x_ricc.set_column(i, &(s.x_cost.column(i) + c.AmBKt * x_ricc_next - c.Klqr.transpose() * r_curr));
            }
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    fn forward_pass(&mut self, xnow: SVector<F, Nx>) {
        let s = &mut self.state;
        let c = &self.cache;

        // Forward-pass with initial state
        s.u.set_column(0, &(-c.Klqr * xnow - s.u_ricc.column(0)));
        s.x.set_column(0, &(s.A * xnow + s.B * s.u.column(0)));

        // Roll out trajectory up to the control horizon Hu
        for i in 1..Hu {
            s.u.set_column(i, &(-c.Klqr * s.x.column(i - 1) - s.u_ricc.column(i)));
            s.x.set_column(i, &(s.A * s.x.column(i - 1) + s.B * s.u.column(i)));
        }

        // For the rest of the prediction horizon (Hx), hold the last control input constant
        let u_final = s.u.column(Hu - 1).clone_owned();
        for i in Hu..Hx {
            s.x.set_column(i, &(s.A * s.x.column(i - 1) + s.B * u_final));
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    fn update_constraints(&mut self) {
        self.state.u_slac = self.state.u_dual + self.state.u;
        self.state.x_slac = self.state.x_dual + self.state.x;

        if let Some((u_min, u_max)) = &self.state.u_bound {
            self.state
                .u_slac
                .zip_zip_apply(u_min, u_max, |u, min, max| *u = u.clamp(min, max));
        }

        if let Some((x_min, x_max)) = &self.state.x_bound {
            self.state
                .x_slac
                .zip_zip_apply(x_min, x_max, |x, min, max| *x = x.clamp(min, max));
        }

        self.state.u_dual += self.state.u - self.state.u_slac;
        self.state.x_dual += self.state.x - self.state.x_slac;
    }

    /// Check for termination condition by evaluating residuals
    fn check_termination(
        &mut self,
        x_slac_old: SMatrix<F, Nx, Hx>,
        u_slac_old: SMatrix<F, Nu, Hu>,
    ) -> bool {

        if (self.state.iter - 1) % self.config.do_check != 0 {
            return false
        }

        let prim_residual_state = (self.state.x - self.state.x_slac).abs().max();
        let dual_residual_state = (x_slac_old - self.state.x_slac).abs().max() * self.cache.rho;
        let prim_residual_input = (self.state.u - self.state.u_slac).abs().max();
        let dual_residual_input = (u_slac_old - self.state.u_slac).abs().max() * self.cache.rho;

        return prim_residual_state < self.config.prim_tol
            && prim_residual_input < self.config.prim_tol
            && dual_residual_state < self.config.dual_tol
            && dual_residual_input < self.config.dual_tol;

    }

    /// Set or un-set varying min-max bounds on inputs for entire horizon
    pub fn set_u_bounds(&mut self, u_bound: Option<(SMatrix<F, Nu, Hu>, SMatrix<F, Nu, Hu>)>) {
        self.state.u_bound = u_bound;
    }

    /// Set or un-set varying min-max bounds on states for entire horizon
    pub fn set_x_bounds(&mut self, x_bound: Option<(SMatrix<F, Nx, Hx>, SMatrix<F, Nx, Hx>)>) {
        self.state.x_bound = x_bound;
    }

    /// Set or un-set the constant min-max bounds on inputs for entire horizon
    pub fn set_const_u_bounds(&mut self, u_bound: Option<(SVector<F, Nu>, SVector<F, Nu>)>) {
        if let Some((vec_min, vec_max)) = u_bound {
            let mut min: SMatrix<F, Nu, Hu> = SMatrix::zeros();
            let mut max: SMatrix<F, Nu, Hu> = SMatrix::zeros();

            for i in 0..Hu {
                min.set_column(i, &vec_min);
                max.set_column(i, &vec_max);
            }
            self.state.u_bound = Some((min, max));
        } else {
            self.state.u_bound = None
        }
    }

    /// Set or un-set the constant min-max bounds on states for entire horizon
    pub fn set_const_x_bounds(&mut self, x_bound: Option<(SVector<F, Nx>, SVector<F, Nx>)>) {
        if let Some((vec_min, vec_max)) = x_bound {
            let mut min: SMatrix<F, Nx, Hx> = SMatrix::zeros();
            let mut max: SMatrix<F, Nx, Hx> = SMatrix::zeros();

            for i in 0..Hx {
                min.set_column(i, &vec_min);
                max.set_column(i, &vec_max);
            }
            self.state.x_bound = Some((min, max));
        } else {
            self.state.x_bound = None
        }
    }

    pub fn reset_dual_variables(&mut self) {
        self.state.u_dual = SMatrix::zeros();
        self.state.x_dual = SMatrix::zeros();
    }

    pub fn get_num_iters(&self) -> usize {
        self.state.iter
    }

    /// Get the system state `x` for the time `i`
    pub fn get_x_at(&self, i: usize) -> SVector<F, Nx> {
        self.state.x.column(i).into()
    }

    /// Get the system input `u` for the time `i`
    pub fn get_u_at(&self, i: usize) -> SVector<F, Nu> {
        self.state.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<F, Nu> {
        self.state.u.column(0).into()
    }

    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<F, Nx, Hx> {
        &self.state.x
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<F, Nu, Hu> {
        &self.state.u
    }

    pub fn prediction_horizon_length(&self) -> usize {
        Hx
    }
    pub fn control_horizon_length(&self) -> usize {
        Hu
    }
    pub fn num_states(&self) -> usize {
        Nx
    }
    pub fn num_inputs(&self) -> usize {
        Nu
    }
}
