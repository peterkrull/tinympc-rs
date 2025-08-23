#![no_std]
#![allow(clippy::op_ref)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]


use nalgebra::{
    RealField, SMatrix, SMatrixView, SVector, SVectorView, SVectorViewMut, Scalar, convert,
};

use crate::{
    constraint::{Constraint, Project},
    rho_cache::Cache,
};

pub mod constraint;
mod optim;
pub mod rho_cache;

pub type LtiFn<F, const Nx: usize, const Nu: usize> =
    fn(SVectorViewMut<F, Nx>, SVectorView<F, Nx>, SVectorView<F, Nu>);

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
    /// The solver converged to within the defined tolerances
    Converged,
    /// The solver reached the maximum number of iterations allowed
    MaxIters,
}

#[derive(Debug)]
pub struct TinyMpc<
    T,
    CACHE: Cache<T, Nx, Nu>,
    const Nx: usize,
    const Nu: usize,
    const Hx: usize,
    const Hu: usize,
> {
    cache: CACHE,
    state: State<T, Nx, Nu, Hx, Hu>,
    pub config: Config<T>,
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

    // Linear cost matrices
    x_cost: SMatrix<T, Nx, Hx>,
    u_cost: SMatrix<T, Nu, Hu>,

    // Riccati backward pass terms
    x_ricc: SMatrix<T, Nx, Hx>,
    u_ricc: SMatrix<T, Nu, Hu>,

    // Number of iterations for latest solve
    iter: usize,
}

pub struct Problem<
    'a,
    T,
    C,
    const Nx: usize,
    const Nu: usize,
    const Hx: usize,
    const Hu: usize,
    XProj = (),
    UProj = (),
> where
    T: Scalar + RealField + Copy,
    C: Cache<T, Nx, Nu>,
    XProj: Project<T, Nx, Hx>,
    UProj: Project<T, Nu, Hu>,
{
    mpc: &'a mut TinyMpc<T, C, Nx, Nu, Hx, Hu>,
    xnow: SVector<T, Nx>,
    xref: Option<SMatrixView<'a, T, Nx, Hx>>,
    uref: Option<SMatrixView<'a, T, Nu, Hu>>,
    xcon: Option<&'a mut [Constraint<T, XProj, Nx, Hx>]>,
    ucon: Option<&'a mut [Constraint<T, UProj, Nu, Hu>]>,
}

impl<'a, T, C, XProj, UProj, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize>
    Problem<'a, T, C, Nx, Nu, Hx, Hu, XProj, UProj>
where
    T: Scalar + RealField + Copy,
    C: Cache<T, Nx, Nu>,
    XProj: Project<T, Nx, Hx>,
    UProj: Project<T, Nu, Hu>,
{
    pub fn x_reference(mut self, xref: impl Into<SMatrixView<'a, T, Nx, Hx>>) -> Self {
        self.xref = Some(xref.into());
        self
    }

    pub fn u_reference(mut self, uref: impl Into<SMatrixView<'a, T, Nu, Hu>>) -> Self {
        self.uref = Some(uref.into());
        self
    }

    pub fn x_constraints<Proj: Project<T, Nx, Hx>>(
        self,
        xcon: &'a mut [Constraint<T, Proj, Nx, Hx>],
    ) -> Problem<'a, T, C, Nx, Nu, Hx, Hu, Proj, UProj> {
        Problem {
            mpc: self.mpc,
            xnow: self.xnow,
            xref: self.xref,
            uref: self.uref,
            xcon: Some(xcon),
            ucon: self.ucon,
        }
    }

    pub fn u_constraints<Proj: Project<T, Nu, Hu>>(
        self,
        ucon: &'a mut [Constraint<T, Proj, Nu, Hu>],
    ) -> Problem<'a, T, C, Nx, Nu, Hx, Hu, XProj, Proj> {
        Problem {
            mpc: self.mpc,
            xnow: self.xnow,
            xref: self.xref,
            uref: self.uref,
            xcon: self.xcon,
            ucon: Some(ucon),
        }
    }

    /// Run the TinyMPC solver
    pub fn solve(mut self) -> (TerminationReason, SVector<T, Nu>) {
        let xcon = self.xcon.take().unwrap_or(&mut [][..]);
        let ucon = self.ucon.take().unwrap_or(&mut [][..]);
        self.mpc.solve(self.xnow, self.xref, self.uref, xcon, ucon)
    }
}

impl<T, C: Cache<T, Nx, Nu>, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize>
    TinyMpc<T, C, Nx, Nu, Hx, Hu>
where
    T: Scalar + RealField + Copy,
{
    #[must_use = "Creatig a new TinyMpc type without assigning it does nothing"]
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
                x_cost: SMatrix::zeros(),
                u_cost: SMatrix::zeros(),
                x_ricc: SMatrix::zeros(),
                u_ricc: SMatrix::zeros(),
                iter: 0,
            },
        })
    }

    pub fn initial_condition(&mut self, xnow: SVector<T, Nx>) -> Problem<'_, T, C, Nx, Nu, Hx, Hu> {
        Problem {
            mpc: self,
            xnow,
            xref: None,
            uref: None,
            xcon: None,
            ucon: None,
        }
    }

    pub fn with_sys(mut self, sys: LtiFn<T, Nx, Nu>) -> Self {
        self.state.sys = Some(sys);
        self
    }

    pub fn solve(
        &mut self,
        mut xnow: SVector<T, Nx>,
        xref: Option<SMatrixView<T, Nx, Hx>>,
        uref: Option<SMatrixView<T, Nu, Hu>>,
        xcon: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) -> (TerminationReason, SVector<T, Nu>) {
        let mut reason = TerminationReason::MaxIters;

        self.warm_start(xcon, ucon);

        self.state.x.set_column(0, &xnow);

        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {
            self.update_cost(xref, uref, xcon, ucon);

            self.backward_pass();

            self.forward_pass(&mut xnow);

            self.update_constraints(xcon, ucon);

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
    #[inline]
    fn warm_start(
        &mut self,
        xcon: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        for con in xcon {
            optim::left_shift_matrix(&mut con.dual);
        }

        for con in ucon {
            optim::left_shift_matrix(&mut con.dual);
        }
    }

    /// Update linear control cost terms
    #[inline]
    fn update_cost(
        &mut self,
        xref: Option<SMatrixView<T, Nx, Hx>>,
        uref: Option<SMatrixView<T, Nu, Hu>>,
        xcon: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Add cost contribution for state constraint violations
        let mut xcon_iter = xcon.iter_mut();
        if let Some(xcon_first) = xcon_iter.next() {
            xcon_first.set_cost(&mut s.x_cost);
            xcon_iter.for_each(|con| con.add_cost(&mut s.x_cost));
            s.x_cost.scale_mut(c.rho);
        } else {
            s.x_cost = SMatrix::<T, Nx, Hx>::zeros()
        }

        // Add cost contribution for input constraint violations
        let mut ucon_iter = ucon.iter_mut();
        if let Some(ucon_first) = ucon_iter.next() {
            ucon_first.set_cost(&mut s.u_cost);
            ucon_iter.for_each(|con| con.add_cost(&mut s.u_cost));
            s.u_cost.scale_mut(c.rho);
        } else {
            s.u_cost = SMatrix::<T, Nu, Hu>::zeros()
        }

        // Extract ADMM cost term for Riccati terminal condition
        s.x_ricc.set_column(Hx - 1, &s.x_cost.column(Hx - 1));

        // Add state tracking cost for stages
        if let Some(xref) = xref {
            for i in 0..Hx {
                let mut x_cost_col = s.x_cost.column_mut(i);
                x_cost_col -= &xref.column(i).component_mul(&s.Q);
            }
        }

        // Add input tracking cost for stages
        if let Some(uref) = uref {
            for i in 0..Hu {
                let mut u_cost_col = s.u_cost.column_mut(i);
                u_cost_col -= &uref.column(i).component_mul(&s.R);
            }
        }

        // Add the terminal tracking penalty, which uses Plqr
        if let Some(xref) = xref {
            s.x_ricc
                .column_mut(Hx - 1)
                .gemm(T::one(), &c.Plqr, &xref.column(Hx - 1), T::one());
        }
    }

    /// Backward pass to update Ricatti variables
    #[inline]
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Scratch vectors for zero-allocation, minimal copy operations
        let mut u_scratch = s.u.column_mut(0);

        // This special case will be optimized away in the case they are different.
        if Hu == Hx - 1 {
            let i = Hu - 1;

            // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
            s.Bt.mul_to(&s.x_ricc.column(i + 1), &mut u_scratch);
            u_scratch += &s.u_cost.column(i);
            c.RpBPBi.mul_to(&u_scratch, &mut s.u_ricc.column_mut(i));
        }

        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx - 1).rev() {
            let (x_ricc_next, mut x_ricc_now) = optim::column_pair_mut(&mut s.x_ricc, i + 1, i);

            // Control action is only optimized up to Hu
            if i < Hu {
                // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
                s.Bt.mul_to(&x_ricc_next, &mut u_scratch);
                u_scratch += &s.u_cost.column(i);
                c.RpBPBi.mul_to(&u_scratch, &mut s.u_ricc.column_mut(i));

                // Calc :: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[i]
                c.AmBKt.mul_to(&x_ricc_next, &mut x_ricc_now);
                x_ricc_now.gemm(T::one(), &c.Klqrt, &s.u_cost.column(i), T::one());
                x_ricc_now += &s.x_cost.column(i);
            } else {
                // Calc :: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[Hu - 1]
                c.AmBKt.mul_to(&x_ricc_next, &mut x_ricc_now);
                x_ricc_now.gemm(T::one(), &c.Klqrt, &s.u_cost.column(Hu - 1), T::one());
                x_ricc_now += &s.x_cost.column(i);
            }
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    #[inline]
    fn forward_pass(&mut self, xnow: &mut SVector<T, Nx>) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        s.x.set_column(0, xnow);

        if let Some(sys) = s.sys {
            // Roll out trajectory up to the control horizon (Hu)
            for i in 0..Hu {
                let (x_now, x_next) = optim::column_pair_mut(&mut s.x, i, i + 1);

                // Calc :: u[i] = -K * x[i] + u_ricc[i]
                let mut u_col = s.u.column_mut(i);
                c.Klqr.mul_to(&x_now, &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                sys(x_next, x_now.as_view(), s.u.column(i));
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx - 1 {
                let (x_now, x_next) = optim::column_pair_mut(&mut s.x, i, i + 1);

                // Calc :: x[i+1] = A * x[i] +  B * u[Hu - 1]
                sys(x_next, x_now.as_view(), u_final);
            }
        } else {
            // Roll out trajectory up to the control horizon Hu
            for i in 0..Hu {
                let (x_now, mut x_next) = optim::column_pair_mut(&mut s.x, i, i + 1);

                // Calc :: u[i] = -K * x[i] + u_ricc[i]
                let mut u_col = s.u.column_mut(i);
                c.Klqr.mul_to(&x_now, &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                s.A.mul_to(&x_now, &mut x_next);
                x_next.gemm(T::one(), &s.B, &u_col, T::one());
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx - 1 {
                let (x_now, mut x_next) = optim::column_pair_mut(&mut s.x, i, i + 1);

                // Calc :: x[i+1] = A * x[i] +  B * u[Hu - 1]
                s.A.mul_to(&x_now, &mut x_next);
                x_next.gemm(T::one(), &s.B, &u_final, T::one());
            }
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    #[inline]
    fn update_constraints(
        &mut self,
        xcon: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let update_res = self.should_calculate_residuals();
        let s = &mut self.state;

        // We are done using the cost matrices for this iteration,
        // so they can safely be used as scratch buffers.
        let u_scratch = &mut s.u_cost;
        let x_scratch = &mut s.x_cost;

        for con in xcon {
            con.constrain(update_res, s.x.as_view(), x_scratch.as_view_mut());
        }

        for con in ucon {
            con.constrain(update_res, s.u.as_view(), u_scratch.as_view_mut());
        }
    }

    /// Check for termination condition by evaluating residuals
    #[inline]
    fn check_termination(
        &mut self,
        xcon: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        ucon: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
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
        if !terminate
            && let Some(scalar) = self
                .cache
                .update_active(max_prim_residual, max_dual_residual)
        {
            for con in xcon.iter_mut() {
                con.rescale_dual(scalar)
            }

            for con in ucon.iter_mut() {
                con.rescale_dual(scalar)
            }
        }

        terminate
    }

    /// Get the number of iterations of the previous solve
    pub fn get_num_iters(&self) -> usize {
        self.state.iter
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
}
