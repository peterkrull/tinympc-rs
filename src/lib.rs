// #![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

*/

use nalgebra::{convert, RealField, SMatrix, SVector, Scalar};

use crate::constraint::DynConstraint;

pub mod constraint;

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

impl<const Nx: usize, const Nu: usize, T> Cache<Nx, Nu, T>
where
    T: Scalar + RealField + Copy,
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
pub struct AdaptiveCache<const Nx: usize, const Nu: usize, const NumRho: usize, F> {
    pub caches: [Cache<Nx, Nu, F>; NumRho]
}

impl<const Nx: usize, const Nu: usize, const NumRho: usize, T> AdaptiveCache<Nx, Nu, NumRho, T>
where
    T: Scalar + RealField + Copy,
{
    pub fn compute(
        base_rho: T,
        iters: usize,
        A: &SMatrix<T, Nx, Nx>,
        B: &SMatrix<T, Nx, Nu>,
        Q: &SVector<T, Nx>,
        R: &SVector<T, Nu>,
    ) -> Result<Self, Error> {

        let mut initializing = [const {None}; NumRho];

        let base_index = NumRho / 2;

        for index in 0..NumRho {
            let power = index as isize - base_index as isize;
            
            // For powers of 2, we can use bit shifting (1 << n) which is 2^n
            let factor = convert((1 << power.abs()) as f64);
            let rho = if power.is_positive() {
                base_rho * factor
            } else if power.is_negative() {
                base_rho / factor
            } else {
                base_rho
            };

            let cache = Cache::compute(rho, iters, A, B, Q, R)?;
            initializing[index] = Some(cache)
        }

        // This will always succeed since Cache::compute short circuits above.
        let caches = initializing.map(|x|x.unwrap());

        Ok(AdaptiveCache{ caches})
    }
}

#[derive(Debug)]
pub struct State<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    // Linear state space model
    A: SMatrix<F, Nx, Nx>,
    B: SMatrix<F, Nx, Nu>,

    // State and input predictions
    x: SMatrix<F, Nx, Hx>,
    u: SMatrix<F, Nu, Hu>,

    // Linear cost matrices
    x_cost: SMatrix<F, Nx, Hx>,
    u_cost: SMatrix<F, Nu, Hu>,

    // Riccati backward pass terms
    x_ricc: SMatrix<F, Nx, Hx>,
    u_ricc: SMatrix<F, Nu, Hu>,

    // Number of iterations for latest solve
    iter: usize,
}

impl<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F>
    TinyMpc<Nx, Nu, Hx, Hu, F>
where
    F: Scalar + RealField + Copy,
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
        if Hx < Hu || Hu == 0 {
            return Err(Error::InvalidHorizonLengths);
        }

        Ok(Self {
            config: Config {
                prim_tol: convert(1e-3),
                dual_tol: convert(1e-3),
                max_iter: 20,
                do_check: 5,
            },
            cache: Cache::compute(rho, Hx, &A, &B, &Q, &R)?,
            state: State {
                A,
                B,
                x: SMatrix::zeros(),
                u: SMatrix::zeros(),
                x_cost: SMatrix::zeros(),
                u_cost: SMatrix::zeros(),
                x_ricc: SMatrix::zeros(),
                u_ricc: SMatrix::zeros(),
                iter: 0,
            },
        })
    }

    pub fn solve(
        &mut self,
        xnow: SVector<F, Nx>,
        xref: Option<&SMatrix<F, Nx, Hx>>,
        uref: Option<&SMatrix<F, Nu, Hu>>,
        mut xcon: Option<&mut [&mut DynConstraint<F, Nx, Hx>]>,
        mut ucon: Option<&mut [&mut DynConstraint<F, Nu, Hu>]>,
    ) -> (TerminationReason, SVector<F, Nu>) {
        let mut termination_reason = TerminationReason::MaxIters;

        // Better warm-starting of dual variables from prior solution
        self.shift_constraint_variables(&mut xcon, &mut ucon);

        // Iteratively solve MPC problem
        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {

            // Update linear control cost terms
            self.update_cost(xref, uref, &xcon, &ucon);

            // Backward pass to update Ricatti variables
            self.backward_pass();

            // Roll out to get new trajectory
            self.forward_pass(xnow);

            // Project into feasible domain
            self.update_constraints(&mut xcon, &mut ucon);

            // Check for early-stop condition
            if self.check_termination(&xcon, &ucon) {
                self.state.iter += 1; 
                termination_reason = TerminationReason::Converged;
                break;
            }

            self.state.iter += 1;
        }

        (termination_reason, self.get_u())
    }

    /// Shift the dual variables by one time step for more accurate hot starting
    fn shift_constraint_variables(&mut self,
        xcon: &mut Option<&mut [&mut DynConstraint<F, Nx, Hx>]>,
        ucon: &mut Option<&mut [&mut DynConstraint<F, Nu, Hu>]>,
    ) {
        if let Some(cons) = xcon.as_mut() {
            for con in cons.iter_mut() {
                con.time_shift_variables();
            }
        }

        if let Some(cons) = ucon.as_mut() {
            for con in cons.iter_mut() {
                con.time_shift_variables();
            }
        }
    }

    /// Update linear control cost terms
    fn update_cost(&mut self, 
        xref: Option<&SMatrix<F, Nx, Hx>>,
        uref: Option<&SMatrix<F, Nu, Hu>>,
        xcon: &Option<&mut [&mut DynConstraint<F, Nx, Hx>]>,
        ucon: &Option<&mut [&mut DynConstraint<F, Nu, Hu>]>,
    ) {
        let s = &mut self.state;
        let c = &self.cache;

        s.x_cost = SMatrix::<F, Nx, Hx>::zeros();
        s.u_cost = SMatrix::<F, Nu, Hu>::zeros();

        // Add cost contribution for state constraint violations
        if let Some(cons) = xcon {
            for con in cons.iter() {
                con.add_cost(&mut s.x_cost);
            }
            s.x_cost.scale_mut(c.rho);
        }
        
        // Add cost contribution for input constraint violations
        if let Some(cons) = ucon {
            for con in cons.iter() {
                con.add_cost(&mut s.u_cost);
            }
            s.u_cost.scale_mut(c.rho);
        }

        // Extract ADMM cost term for Riccati terminal condition
        s.x_ricc.set_column(Hx - 1, &s.x_cost.column(Hx - 1));

        // Add tracking cost for stages
        if let Some(xref) = xref {
            for i in 0..Hx {
                let mut x_cost_col = s.x_cost.column_mut(i);
                x_cost_col -= xref.column(i).component_mul(&c.Q_aug);
            }
        }
        if let Some(uref) = uref {
        for i in 0..Hu {
                let mut u_cost_col = s.u_cost.column_mut(i);
                u_cost_col -= uref.column(i).component_mul(&c.R_aug);
            }
        }

        // Add the terminal tracking penalty, which uses Plqr
        if let Some(xref) = xref {
            let mut x_ricc_terminal = s.x_ricc.column_mut(Hx - 1);
            x_ricc_terminal -= c.Plqr * xref.column(Hx - 1);
        }
    }

    /// Update linear terms from Riccati backward pass
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = &self.cache;

        // Handles this special case correctly.
        // This will be optimized away in the case they are different.
        if Hu == Hx {
            let i = Hu - 1;
            let x_ricc_next = s.x_ricc.column(i);
            let r_curr = s.u_cost.column(i);
            s.u_ricc.set_column(i, &(c.RpBPBi * (s.B.transpose() * x_ricc_next + r_curr)));
        }

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
    fn update_constraints(&mut self,
        xcon: &mut Option<&mut [&mut DynConstraint<F, Nx, Hx>]>,
        ucon: &mut Option<&mut [&mut DynConstraint<F, Nu, Hu>]>,
    ) {
        let s = &mut self.state;

        if let Some(cons) = xcon.as_mut() {
            for con in cons.iter_mut() {
                con.constrain(&s.x);
            }
        }

        if let Some(cons) = ucon.as_mut() {
            for con in cons.iter_mut() {
                con.constrain(&s.u);
            }
        }
    }

    /// Check for termination condition by evaluating residuals
    fn check_termination(
        &mut self,
        xcon: &Option<&mut [&mut DynConstraint<F, Nx, Hx>]>,
        ucon: &Option<&mut [&mut DynConstraint<F, Nu, Hu>]>,
    ) -> bool {
        let s = &mut self.state;
        let c = &self.cache;
        let cfg = &self.config;

        if s.iter % cfg.do_check != 0 {
            return false
        }

        let mut max_prim_residual = F::zero();
        let mut max_dual_residual = F::zero();

        if let Some(cons) = xcon {
            for con in cons.iter() {
                max_prim_residual = max_prim_residual.max(con.max_prim_residual);
                max_dual_residual = max_dual_residual.max(con.max_dual_residual);
            }
        }

        if let Some(cons) = ucon {
            for con in cons.iter() {
                max_prim_residual = max_prim_residual.max(con.max_prim_residual);
                max_dual_residual = max_dual_residual.max(con.max_dual_residual);
            }
        }

        // TODO Do adaptive rho
        // if max_prim_residual * convert(10.0) > max_dual_residual {
        // } else if max_dual_residual * convert(10.0) > max_prim_residual {
        // }

        max_prim_residual < cfg.prim_tol && max_dual_residual * c.rho < cfg.dual_tol
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
