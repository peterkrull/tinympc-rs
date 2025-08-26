// #![no_std]
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
pub mod rho_cache;

mod util;

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

    // Tracking dynamics mismatch
    w: SMatrix<T, Nx, Hx>,

    // Linear cost matrices
    q: SMatrix<T, Nx, Hx>,
    r: SMatrix<T, Nu, Hu>,

    // Riccati backward pass terms
    p: SMatrix<T, Nx, Hx>,
    d: SMatrix<T, Nu, Hu>,

    // State and input tracking error predictions
    ex: SMatrix<T, Nx, Hx>,
    eu: SMatrix<T, Nu, Hu>,

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
    x_now: SVector<T, Nx>,
    x_ref: Option<SMatrixView<'a, T, Nx, Hx>>,
    u_ref: Option<SMatrixView<'a, T, Nu, Hu>>,
    x_con: Option<&'a mut [Constraint<T, XProj, Nx, Hx>]>,
    u_con: Option<&'a mut [Constraint<T, UProj, Nu, Hu>]>,
}

impl<'a, T, C, XProj, UProj, const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize>
    Problem<'a, T, C, Nx, Nu, Hx, Hu, XProj, UProj>
where
    T: Scalar + RealField + Copy,
    C: Cache<T, Nx, Nu>,
    XProj: Project<T, Nx, Hx>,
    UProj: Project<T, Nu, Hu>,
{
    pub fn x_reference(mut self, x_ref: impl Into<SMatrixView<'a, T, Nx, Hx>>) -> Self {
        self.x_ref = Some(x_ref.into());
        self
    }

    pub fn u_reference(mut self, u_ref: impl Into<SMatrixView<'a, T, Nu, Hu>>) -> Self {
        self.u_ref = Some(u_ref.into());
        self
    }

    pub fn x_constraints<Proj: Project<T, Nx, Hx>>(
        self,
        x_con: &'a mut [Constraint<T, Proj, Nx, Hx>],
    ) -> Problem<'a, T, C, Nx, Nu, Hx, Hu, Proj, UProj> {
        Problem {
            mpc: self.mpc,
            x_now: self.x_now,
            x_ref: self.x_ref,
            u_ref: self.u_ref,
            x_con: Some(x_con),
            u_con: self.u_con,
        }
    }

    pub fn u_constraints<Proj: Project<T, Nu, Hu>>(
        self,
        u_con: &'a mut [Constraint<T, Proj, Nu, Hu>],
    ) -> Problem<'a, T, C, Nx, Nu, Hx, Hu, XProj, Proj> {
        Problem {
            mpc: self.mpc,
            x_now: self.x_now,
            x_ref: self.x_ref,
            u_ref: self.u_ref,
            x_con: self.x_con,
            u_con: Some(u_con),
        }
    }

    /// Run the TinyMPC solver
    pub fn solve(mut self) -> (TerminationReason, SVector<T, Nu>) {
        let x_con = self.x_con.take().unwrap_or(&mut [][..]);
        let u_con = self.u_con.take().unwrap_or(&mut [][..]);
        self.mpc
            .solve(self.x_now, self.x_ref, self.u_ref, x_con, u_con)
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
                Bt: B.transpose(),
                w: SMatrix::zeros(),
                q: SMatrix::zeros(),
                r: SMatrix::zeros(),
                p: SMatrix::zeros(),
                d: SMatrix::zeros(),
                ex: SMatrix::zeros(),
                eu: SMatrix::zeros(),
                iter: 0,
            },
        })
    }

    pub fn initial_condition(
        &mut self,
        x_now: SVector<T, Nx>,
    ) -> Problem<'_, T, C, Nx, Nu, Hx, Hu> {
        Problem {
            mpc: self,
            x_now,
            x_ref: None,
            u_ref: None,
            x_con: None,
            u_con: None,
        }
    }

    pub fn solve(
        &mut self,
        x_now: SVector<T, Nx>,
        x_ref: Option<SMatrixView<T, Nx, Hx>>,
        u_ref: Option<SMatrixView<T, Nu, Hu>>, // unused for now
        x_con: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        u_con: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) -> (TerminationReason, SVector<T, Nu>) {
        let mut reason = TerminationReason::MaxIters;

        // For now assume we have a state reference
        let u_ref = u_ref.unwrap();
        let x_ref = x_ref.unwrap();

        // Set initial error state
        self.state.ex.set_column(0, &(x_now - x_ref.column(0)));

        // Construct tracking dynamics mismatch matrix (leveas last empty?)
        for i in 0..Hx - 1 {
            self.state
                .w
                .set_column(i, &(self.state.A * x_ref.column(i) - x_ref.column(i + 1)))
        }

        self.warm_start(x_con, u_con);

        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {
            // This goes first
            self.update_cost(x_con, u_con);

            self.backward_pass();

            self.forward_pass();

            self.update_constraints(x_ref, u_ref, x_con, u_con);

            if self.check_termination(x_con, u_con) {
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
        x_con: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        u_con: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        for con in x_con {
            util::shift_columns_left(&mut con.dual);
            util::shift_columns_left(&mut con.slac);
        }

        for con in u_con {
            util::shift_columns_left(&mut con.dual);
            util::shift_columns_left(&mut con.slac);
        }

        // util::shift_columns_left(&mut self.state.d);
        // util::shift_columns_left(&mut self.state.p);
    }
    /// Update linear control cost terms

    #[inline]

    fn update_cost(
        &mut self,
        x_con: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        u_con: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Add cost contribution for state constraint violations
        let mut x_con_iter = x_con.iter_mut();
        if let Some(x_con_first) = x_con_iter.next() {
            x_con_first.set_cost(&mut s.q);
            x_con_iter.for_each(|con| con.add_cost(&mut s.q));
            s.q.scale_mut(c.rho);
        } else {
            s.q = SMatrix::<T, Nx, Hx>::zeros()
        }

        // Add cost contribution for input constraint violations
        let mut u_con_iter = u_con.iter_mut();
        if let Some(u_con_first) = u_con_iter.next() {
            u_con_first.set_cost(&mut s.r);
            u_con_iter.for_each(|con| con.add_cost(&mut s.r));
            s.r.scale_mut(c.rho);
        } else {
            s.r = SMatrix::<T, Nu, Hu>::zeros()
        }

        // Extract ADMM cost term for Riccati terminal condition
        s.p.set_column(Hx - 1, &(s.q.column(Hx - 1)));
    }

    /// Backward pass to update Ricatti variables
    #[inline]
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        for i in (0..Hx - 1).rev() {
            let (mut p_now, p_fut) = util::column_pair_mut(&mut s.p, i, i + 1);
            p_now.copy_from(&(c.AmBKt * (p_fut + &c.Plqr * &s.w.column(i)) + s.q.column(i) - c.Klqr.transpose() * s.r.column(i.min(Hu - 1))));
        }

        for i in (0..Hu).rev() {
            let p_fut = s.p.column(i+1);
            s.d.set_column(i, &(c.RpBPBi * (&s.Bt * &(&p_fut + &c.Plqr * &s.w.column(i)) + s.r.column(i))));
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    #[inline]
    fn forward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Roll out trajectory up to the control horizon (Hu)
        for i in 0..Hu {
            let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
            s.eu.set_column(i, &(-&c.Klqr * &ex_now - s.d.column(i)));
            ex_fut.copy_from(&(&s.A * &ex_now + &s.B * s.eu.column(i) + s.w.column(i)));
        }

        // Roll out rest of trajectory keeping u constant
        for i in Hu..Hx - 1 {
            let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
            ex_fut.copy_from(&(&s.A * &ex_now + &s.B * s.eu.column(Hu - 1) + s.w.column(i)));
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    #[inline]
    fn update_constraints(
        &mut self,
        x_ref: SMatrixView<T, Nx, Hx>,
        u_ref: SMatrixView<T, Nu, Hu>,
        x_con: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        u_con: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) {
        let update_res = self.should_calculate_residuals();
        let s = &mut self.state;

        // We are done using the cost matrices for this iteration,
        // so they can safely be used as scratch buffers.
        let u_scratch = &mut s.r;
        let x_scratch = &mut s.q;

        for con in x_con {
            con.constrain(update_res, x_ref, s.ex.as_view());
        }

        for con in u_con {
            con.constrain(update_res, u_ref, s.eu.as_view());
        }
    }

    /// Check for termination condition by evaluating residuals
    #[inline]
    fn check_termination(
        &mut self,
        x_con: &mut [Constraint<T, impl Project<T, Nx, Hx>, Nx, Hx>],
        u_con: &mut [Constraint<T, impl Project<T, Nu, Hu>, Nu, Hu>],
    ) -> bool {
        let c = self.cache.get_active();
        let cfg = &self.config;

        if !self.should_calculate_residuals() {
            return false;
        }

        let mut max_prim_residual = T::zero();
        let mut max_dual_residual = T::zero();

        for con in x_con.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
        }

        for con in u_con.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
        }

        let terminate =
            max_prim_residual < cfg.prim_tol && max_dual_residual * c.rho < cfg.dual_tol;

        let is_last = self.state.iter == cfg.max_iter - 1;

        // Try to adapt rho
        if (!terminate || is_last)
            && let Some(scalar) = self
                .cache
                .update_active(max_prim_residual, max_dual_residual)
        {
            for con in x_con.iter_mut() {
                con.rescale_dual(scalar)
            }

            for con in u_con.iter_mut() {
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
        self.state.eu.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<T, Nu> {
        self.state.eu.column(0).into()
    }

    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<T, Nx, Hx> {
        &self.state.ex
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<T, Nu, Hu> {
        &self.state.eu
    }
}
