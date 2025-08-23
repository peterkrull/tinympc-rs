#![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use nalgebra::{RealField, SMatrix, SMatrixView, SVector, SVectorView, Scalar, convert};

use crate::{
    constraint::{Constraint, DynConstraint, Project},
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
    'b,
    T,
    CACHE,
    const Nx: usize,
    const Nu: usize,
    const Hx: usize,
    const Hu: usize,
> where
    T: Scalar + RealField + Copy,
    CACHE: Cache<T, Nx, Nu>,
{
    mpc: &'a mut TinyMpc<T, CACHE, Nx, Nu, Hx, Hu>,
    xnow: SVector<T, Nx>,
    xref: Option<SMatrixView<'a, T, Nx, Hx>>,
    uref: Option<SMatrixView<'a, T, Nu, Hu>>,
    xcon: Option<&'a mut [DynConstraint<'b, T, Nx, Hx>]>,
    ucon: Option<&'a mut [DynConstraint<'b, T, Nu, Hu>]>,
}

impl<'a, 'b, T, CACHE, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize>
    Problem<'a, 'b, T, CACHE, Nx, Nu, Hx, Hu>
where
    T: Scalar + RealField + Copy,
    CACHE: Cache<T, Nx, Nu>,
{
    pub fn x_reference(mut self, xref: impl Into<SMatrixView<'a, T, Nx, Hx>>) -> Self {
        self.xref = Some(xref.into());
        self
    }

    pub fn u_reference(mut self, uref: impl Into<SMatrixView<'a, T, Nu, Hu>>) -> Self {
        self.uref = Some(uref.into());
        self
    }

    pub fn x_constraints(mut self, xcon: &'a mut [DynConstraint<'b, T, Nx, Hx>]) -> Self {
        self.xcon = Some(xcon);
        self
    }

    pub fn u_constraints(mut self, ucon: &'a mut [DynConstraint<'b, T, Nu, Hu>]) -> Self {
        self.ucon = Some(ucon);
        self
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

    pub fn initial_condition<'b>(
        &mut self,
        xnow: SVector<T, Nx>,
    ) -> Problem<'_, 'b, T, C, Nx, Nu, Hx, Hu> {
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
        /// Does copying in as large chunks as possible. Last two columns will be identical
        fn left_shift_matrix<F, const ROWS: usize, const COLS: usize>(
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

        for con in xcon {
            left_shift_matrix(&mut con.dual);
        }

        for con in ucon {
            left_shift_matrix(&mut con.dual);
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

        s.x_cost = SMatrix::<T, Nx, Hx>::zeros();
        s.u_cost = SMatrix::<T, Nu, Hu>::zeros();

        // The data in the prediction matrices is outdated and will be
        // overwritten later this iteration. For now they can be used as
        // a scratch buffer for temporary calculations.
        let u_scratch = &mut s.u;
        let x_scratch = &mut s.x;

        // Add cost contribution for state constraint violations
        let mut xcon_iter = xcon.iter_mut();
        if let Some(xcon_first) = xcon_iter.next() {
            xcon_first.set_cost(s.x_cost.as_view_mut());
            for con in xcon_iter {
                con.add_cost(s.x_cost.as_view_mut(), x_scratch.as_view_mut());
            }
            s.x_cost.scale_mut(c.rho);
        } else {
            s.x_cost = SMatrix::<T, Nx, Hx>::zeros()
        }

        // Add cost contribution for input constraint violations
        let mut ucon_iter = ucon.iter_mut();
        if let Some(ucon_first) = ucon_iter.next() {
            ucon_first.set_cost(s.u_cost.as_view_mut());
            for con in ucon_iter {
                con.add_cost(s.u_cost.as_view_mut(), u_scratch.as_view_mut());
            }
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
            let mut x_scratch = x_scratch.column_mut(0);
            let mut x_ricc_terminal = s.x_ricc.column_mut(Hx - 1);
            c.Plqr.mul_to(&xref.column(Hx - 1), &mut x_scratch);
            x_ricc_terminal -= &x_scratch;
        }
    }

    /// Backward pass to update Ricatti variables
    #[inline]
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Scratch vectors for zero-allocation, minimal copy operations
        // Due to borrowing rules, we cannot define the x_scratch* vectors
        // at the start, since that would require borrowing s.x mutably twice.
        // Instead they are used in line in the places we need them. Included
        // here as comments for clarity.
        let mut u_scratch = s.u.column_mut(0);
        // let mut x_scratch0 = s.x.column_mut(0);
        // let mut x_scratch1 = s.x.column_mut(1);

        // Handles this special case correctly.
        // This will be optimized away in the case they are different.
        if Hu == Hx - 1 {
            let i = Hu - 1;

            // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
            s.Bt.mul_to(&s.x_ricc.column(i + 1), &mut u_scratch);
            u_scratch += &s.u_cost.column(i);
            c.RpBPBi.mul_to(&u_scratch, &mut s.u_ricc.column_mut(i));
        }

        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx - 1).rev() {
            let x_ricc_next = s.x_ricc.column(i + 1);

            // Control action is only optimized up to Hu
            if i < Hu {
                // Calc :: u_ricc[i] = RpBPBi * (B^T * x_ricc[i + 1] + u_cost[i])
                s.Bt.mul_to(&x_ricc_next, &mut u_scratch);
                u_scratch += &s.u_cost.column(i);
                c.RpBPBi.mul_to(&u_scratch, &mut s.u_ricc.column_mut(i));

                // Calc :: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[i]
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x.column_mut(0));
                c.Klqrt.mul_to(&s.u_cost.column(i), &mut s.x.column_mut(1));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost.column(i).add_to(&s.x.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x.column(1);
            } else {
                // Update: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i + 1] + Klqr^T * u_cost[Hu - 1]
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x.column_mut(0));
                c.Klqrt
                    .mul_to(&s.u_cost.column(Hu - 1), &mut s.x.column_mut(1));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost.column(i).add_to(&s.x.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x.column(1);
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
                c.negKlqr.mul_to(xnow, &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Calc :: x[i+1] = A * x[i] +  B * u[i]
                let mut x_col = s.x.column_mut(i + 1);
                s.A.mul_to(xnow, &mut x_col);
                x_col.gemm(T::one(), &s.B, &u_col, T::one());
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx - 1 {
                xnow.copy_from(&s.x.column(i));

                // Calc :: x[i+1] = A * x[i] +  B * u[Hu - 1]
                let mut x_col = s.x.column_mut(i + 1);
                s.A.mul_to(xnow, &mut x_col);
                x_col.gemm(T::one(), &s.B, &u_final, T::one());
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
