// #![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use nalgebra::{RealField, SMatrix, SMatrixView, SVector, SVectorView, Scalar, convert};

use crate::{
    constraint::{Constraint, Project},
    rho_cache::Cache,
};

pub mod constraint;
pub mod rho_cache;

pub type LtiFn<F, const Nx: usize, const Nu: usize> =
    fn(SVectorView<F, Nx>, SVectorView<F, Nu>) -> SVector<F, Nx>;

/// Errors that can occur during system setup
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Error {
    /// The value of Hx must be larger than Hu `(Hx > Hu && Hu > 0)`
    InvalidHorizonLength,
    /// The value of rho must be strictly positive `(rho > 0)`
    RhoNotPositiveDefinite,
    /// The entries of Q must be non-negative `all(Q) >= 0`
    QNotPositiveSemidefinite,
    /// The entries of Q must be strictly positive `all(R) > 0`
    RNotPositiveDefinite,
    /// The matrix `R_aug + B^T * P * B` is not invertible
    RpBPBNotInvertible,
    /// The resulting matrices contained non-finite elements (Inf or NaN)
    NonFiniteValues,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TerminationReason {
    Converged,
    MaxIters,
}

#[derive(Debug)]
pub struct TinyMpc<
    T,
    C: rho_cache::Cache<T, Nx, Nu>,
    const Nx: usize,
    const Nu: usize,
    const Hx: usize,
    const Hu: usize,
> {
    pub config: Config<T>,
    cache: C,
    state: State<T, Nx, Nu, Hx, Hu>,
}

#[derive(Debug)]
pub struct Config<T> {
    /// The convergence tolerance for the primal residual (default 0.001)
    pub prim_tol: T,

    /// The convergence tolerance for the dual residual  (default 0.001)
    pub dual_tol: T,

    /// Maximum iterations without converging before terminating (default 50)
    pub max_iter: usize,

    /// Number of iterations between evaluating convergence (default 5)
    pub do_check: usize,
}

macro_rules! timed {
    ($description:literal $($token:tt)+) => {
        // let time = embassy_time::Instant::now();
        $($token)+
        // let elapsed = time.elapsed();
        // defmt::info!("Elapsed: {} us :: ({})", elapsed.as_micros(), $description);
        // embassy_time::Timer::after_micros(500).await;
    };
}

#[derive(Debug)]
pub struct State<T, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize> {
    // Linear state space model
    A: SMatrix<T, Nx, Nx>,
    B: SMatrix<T, Nx, Nu>,
    Bt: SMatrix<T, Nu, Nx>,

    /// State tracking error penalty vector
    Q: SVector<T, Nx>,

    /// Input tracking error penalty vector
    R: SVector<T, Nu>,

    /// In case A and B are sparse, a function doing the
    /// `x' <- A*x + B*u` calculation manually is likely faster.
    sys: Option<LtiFn<T, Nx, Nu>>,

    // State and input predictions
    x: SMatrix<T, Nx, Hx>,
    u: SMatrix<T, Nu, Hu>,

    // Scratch buffers for zero allocation operations
    x_scratch: SMatrix<T, Nx, Hx>,
    u_scratch: SMatrix<T, Nu, Hu>,

    // Linear cost matrices
    x_cost: SMatrix<T, Nx, Hx>,
    u_cost: SMatrix<T, Nu, Hu>,

    // Riccati backward pass terms
    x_ricc: SMatrix<T, Nx, Hx>,
    u_ricc: SMatrix<T, Nu, Hu>,

    // Number of iterations for latest solve
    iter: usize,
}

impl<T, C: Cache<T, Nx, Nu>, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize>
    TinyMpc<T, C, Nx, Nu, Hx, Hu>
where
    T: Scalar + RealField + Copy,
{
    #[must_use]
    pub fn new(
        A: SMatrix<T, Nx, Nx>,
        B: SMatrix<T, Nx, Nu>,
        Q: SVector<T, Nx>,
        R: SVector<T, Nu>,
        rho: T,
    ) -> Result<Self, Error> {
        // Guard against invalid horizon lengths
        if Hx <= Hu || Hu == 0 {
            return Err(Error::InvalidHorizonLength);
        }

        Ok(Self {
            config: Config {
                prim_tol: convert(1e-3),
                dual_tol: convert(1e-3),
                max_iter: 50,
                do_check: 5,
            },
            cache: C::new(rho, Hx, &A, &B, &Q, &R)?,
            state: State {
                A,
                B,
                Q,
                R,
                Bt: B.transpose(),
                sys: None,
                x: SMatrix::zeros(),
                u: SMatrix::zeros(),
                x_scratch: SMatrix::zeros(),
                u_scratch: SMatrix::zeros(),
                x_cost: SMatrix::zeros(),
                u_cost: SMatrix::zeros(),
                x_ricc: SMatrix::zeros(),
                u_ricc: SMatrix::zeros(),
                iter: 0,
            },
        })
    }

    pub fn with_sys(mut self, sys: LtiFn<T, Nx, Nu>) -> Self {
        self.state.sys = Some(sys);
        self
    }

    #[inline(never)]
    pub fn solve(
        &mut self,
        xnow: SVector<T, Nx>,
        xref: Option<SMatrixView<'_, T, Nx, Hx>>,
        uref: Option<SMatrixView<'_, T, Nu, Hu>>,
        xcon: Option<&mut [&mut Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>]>,
        ucon: Option<&mut [&mut Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>]>,
    ) -> (TerminationReason, SVector<T, Nu>) {
        let mut reason = TerminationReason::MaxIters;

        let mut xcon = xcon.unwrap_or(&mut [][..]);
        let mut ucon = ucon.unwrap_or(&mut [][..]);

        // Better warm-starting of dual variables from prior solution
        timed! {
            "shift constraint variables"
            self.shift_constraint_variables(&mut xcon, &mut ucon);
        }

        // Iteratively solve MPC problem
        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {
            // defmt::debug!("Iteration number: {}", self.state.iter);

            timed! {
                "Update linear control cost terms"
                self.update_cost(xref, uref, xcon, ucon);
            }

            timed! {
                "Backward pass to update Ricatti variables"
                self.backward_pass();
            }

            timed! {
                "Roll out to get new trajectory"
                self.forward_pass(xnow);
            }

            timed! {
                "Project into feasible domain"
                self.update_constraints(xcon, ucon);
            }

            // Check for early-stop condition
            if self.check_termination(xcon, ucon) {
                reason = TerminationReason::Converged;
                self.state.iter += 1;
                break;
            }

            self.state.iter += 1;
        }

        (reason, self.get_u())
    }

    fn should_calculate_residuals(&self) -> bool {
        self.state.iter % self.config.do_check == 0
    }

    /// Shift the dual variables by one time step for more accurate hot starting
    #[inline(never)]
    fn shift_constraint_variables(
        &mut self,
        xcon: &mut [&mut Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        for con in xcon {
            con.time_shift_variables();
        }

        for con in ucon {
            con.time_shift_variables();
        }
    }

    /// Update linear control cost terms
    #[inline(never)]
    fn update_cost(
        &mut self,
        xref: Option<SMatrixView<T, Nx, Hx>>,
        uref: Option<SMatrixView<T, Nu, Hu>>,
        xcon: &mut [&mut Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        s.x_cost = SMatrix::<T, Nx, Hx>::zeros();
        s.u_cost = SMatrix::<T, Nu, Hu>::zeros();

        // Add cost contribution for state constraint violations
        if !xcon.is_empty() {
            for con in xcon {
                con.add_cost(s.x_cost.as_view_mut(), s.x_scratch.as_view_mut());
            }
            s.x_cost.scale_mut(c.rho);
        }

        // Add cost contribution for input constraint violations
        if !ucon.is_empty() {
            for con in ucon {
                con.add_cost(s.u_cost.as_view_mut(), s.u_scratch.as_view_mut());
            }
            s.u_cost.scale_mut(c.rho);
        }

        // Extract ADMM cost term for Riccati terminal condition
        s.x_ricc.set_column(Hx - 1, &s.x_cost.column(Hx - 1));

        // Add tracking cost for stages
        if let Some(xref) = xref {
            for i in 0..Hx {
                let mut x_cost_col = s.x_cost.column_mut(i);
                x_cost_col -= &xref.column(i).component_mul(&s.Q);
            }
        }

        if let Some(uref) = uref {
            for i in 0..Hu {
                let mut u_cost_col = s.u_cost.column_mut(i);
                u_cost_col -= &uref.column(i).component_mul(&s.R);
            }
        }

        // Add the terminal tracking penalty, which uses Plqr
        if let Some(xref) = xref {
            let mut x_ricc_terminal = s.x_ricc.column_mut(Hx - 1);
            x_ricc_terminal -= &c.Plqr * xref.column(Hx - 1);
        }
    }

    /// Backward pass to update Ricatti variables
    #[inline(never)]
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Scratch vectors for zero-allocation, minimal copy operations
        let mut u_scratch_vec = s.u_scratch.column_mut(0);

        // Handles this special case correctly.
        // This will be optimized away in the case they are different.
        if Hu == Hx - 1 {
            let i = Hu - 1;

            // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
            s.Bt.mul_to(&s.x_ricc.column(i + 1), &mut u_scratch_vec);
            u_scratch_vec += &s.u_cost.column(i);
            c.RpBPBi.mul_to(&u_scratch_vec, &mut s.u_ricc.column_mut(i));
        }


        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx - 1).rev() {
            let x_ricc_next = s.x_ricc.column(i + 1);

            // Control action is only optimized up to Hu
            if i < Hu {
                // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
                s.Bt.mul_to(&x_ricc_next, &mut u_scratch_vec);
                u_scratch_vec += &s.u_cost.column(i);
                c.RpBPBi.mul_to(&u_scratch_vec, &mut s.u_ricc.column_mut(i));

                // Calc :: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[i]
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x_scratch.column_mut(0));
                c.Klqrt
                    .mul_to(&s.u_cost.column(i), &mut s.x_scratch.column_mut(1));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost
                    .column(i)
                    .add_to(&s.x_scratch.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x_scratch.column(1);
            } else {
                // Update: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[Hu - 1]
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x_scratch.column_mut(0));
                c.Klqrt
                    .mul_to(&s.u_cost.column(Hu - 1), &mut s.x_scratch.column_mut(1));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost
                    .column(i)
                    .add_to(&s.x_scratch.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x_scratch.column(1);
            }
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    #[inline(never)]
    fn forward_pass(&mut self, mut xnow: SVector<T, Nx>) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        s.x.set_column(0, &xnow);

        if let Some(sys) = s.sys {

            // Roll out trajectory up to the control horizon (Hu)
            for i in 0..Hu {

                // Calc :: u[i] = -K * x[i] + u_ricc[i]
                let mut u_col = s.u.column_mut(i);
                c.negKlqr.mul_to(&s.x.column(i), &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                s.x.set_column(i + 1, &sys(s.x.column(i), s.u.column(i)));
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx {

                // Calc :: x[i+1] = A * x[i] +  B * u[Hu - 1]
                s.x.set_column(i, &sys(s.x.column(i - 1), u_final));
            }
        } else {

            // Roll out trajectory up to the control horizon Hu
            for i in 0..Hu {
                xnow.copy_from(&s.x.column(i));

                // Calc :: u[i] = -K * x[i] + u_ricc[i]
                let mut u_col = s.u.column_mut(i);
                c.negKlqr.mul_to(&xnow, &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                let mut x_col = s.x.column_mut(i + 1);
                s.A.mul_to(&xnow, &mut x_col);
                x_col.gemm(T::one(), &s.B, &u_col, T::one());
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx - 1 {
                xnow.copy_from(&s.x.column(i));

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                let mut x_col = s.x.column_mut(i + 1);
                s.A.mul_to(&xnow, &mut x_col);
                x_col.gemm(T::one(), &s.B, &u_final, T::one());
            }
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    #[inline(never)]
    fn update_constraints(
        &mut self,
        xcon: &mut [&mut Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let update_res = self.should_calculate_residuals();
        let s = &mut self.state;

        for con in xcon {
            con.constrain(update_res, s.x.as_view(), s.x_scratch.as_view_mut());
        }

        for con in ucon {
            con.constrain(update_res, s.u.as_view(), s.u_scratch.as_view_mut());
        }
    }

    /// Check for termination condition by evaluating residuals
    #[inline(never)]
    fn check_termination(
        &mut self,
        xcon: &mut [&mut Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) -> bool {
        let c = self.cache.get_active();
        let cfg = &self.config;

        if !self.should_calculate_residuals() {
            return false;
        }

        let mut max_prim_residual = T::zero();
        let mut max_dual_residual = T::zero();

        for con in xcon.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
        }

        for con in ucon.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
        }

        let terminate =
            max_prim_residual < cfg.prim_tol && max_dual_residual * c.rho < cfg.dual_tol;

        // Try to adapt rho
        if !terminate {
            if let Some(scalar) = self
                .cache
                .update_active(max_prim_residual, max_dual_residual)
            {
                println!("Switching to cache: rho = {}", self.cache.get_active().rho);

                for con in xcon.iter_mut() {
                    con.rescale_dual(scalar)
                }

                for con in ucon.iter_mut() {
                    con.rescale_dual(scalar)
                }
            }
        }

        terminate
    }

    pub fn get_num_iters(&self) -> usize {
        self.state.iter
    }

    /// Get the system state `x` for the time `i`
    pub fn get_x_at(&self, i: usize) -> SVector<T, Nx> {
        self.state.x.column(i).into()
    }

    /// Get the system input `u` for the time `i`
    pub fn get_u_at(&self, i: usize) -> SVector<T, Nu> {
        self.state.u.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<T, Nu> {
        self.state.u.column(0).into()
    }

    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<T, Nx, Hx> {
        &self.state.x
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<T, Nu, Hu> {
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
