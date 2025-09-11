#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_snake_case)]

use nalgebra::{RealField, SMatrix, SVector, SVectorView, SVectorViewMut, Scalar, convert};

pub mod cache;
pub mod constraint;
pub mod project;

pub use cache::{Cache, Error};
pub use constraint::Constraint;
pub use project::*;

mod util;

pub type LtiFn<T, const NX: usize, const NU: usize> =
    fn(SVectorViewMut<T, NX>, SVectorView<T, NX>, SVectorView<T, NU>);

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
    CACHE: Cache<T, NX, NU>,
    const NX: usize,
    const NU: usize,
    const HX: usize,
    const HU: usize,
> {
    cache: CACHE,
    state: State<T, NX, NU, HX, HU>,
    pub config: Config<T>,
}

#[derive(Debug)]
pub struct Config<T> {
    /// The convergence tolerance for the primal residual (default 0.001)
    pub prim_tol: T,

    /// The convergence tolerance for the dual residual (default 0.001)
    pub dual_tol: T,

    /// Maximum iterations without converging before terminating (default 50)
    pub max_iter: usize,

    /// Number of iterations between evaluating convergence (default 5)
    pub do_check: usize,

    /// Relaxation, values `1.5-1.8` may improve convergence (default 1.0)
    pub relaxation: T,
}

#[derive(Debug)]
pub struct State<T, const NX: usize, const NU: usize, const HX: usize, const HU: usize> {
    // Linear state space model
    A: SMatrix<T, NX, NX>,
    B: SMatrix<T, NX, NU>,

    // For sparse system dynamics
    sys: Option<LtiFn<T, NX, NU>>,

    // State and input tracking error predictions
    ex: SMatrix<T, NX, HX>,
    eu: SMatrix<T, NU, HU>,

    // State tracking dynamics mismatch
    cx: SMatrix<T, NX, HX>,
    cp: SMatrix<T, NX, HX>,

    // Linear cost matrices
    q: SMatrix<T, NX, HX>,
    r: SMatrix<T, NU, HU>,

    // Riccati backward pass terms
    p: SMatrix<T, NX, HX>,
    d: SMatrix<T, NU, HU>,

    // Number of iterations for latest solve
    iter: usize,
}

pub struct Problem<
    'a,
    T,
    C,
    const NX: usize,
    const NU: usize,
    const HX: usize,
    const HU: usize,
    XProj = (),
    UProj = (),
> where
    T: Scalar + RealField + Copy,
    C: Cache<T, NX, NU>,
    XProj: Project<T, NX, HX>,
    UProj: Project<T, NU, HU>,
{
    mpc: &'a mut TinyMpc<T, C, NX, NU, HX, HU>,
    x_now: SVector<T, NX>,
    x_ref: Option<&'a SMatrix<T, NX, HX>>,
    u_ref: Option<&'a SMatrix<T, NU, HU>>,
    x_con: Option<&'a mut [Constraint<T, XProj, NX, HX>]>,
    u_con: Option<&'a mut [Constraint<T, UProj, NU, HU>]>,
}

impl<'a, T, C, XProj, UProj, const NX: usize, const NU: usize, const HX: usize, const HU: usize>
    Problem<'a, T, C, NX, NU, HX, HU, XProj, UProj>
where
    T: Scalar + RealField + Copy,
    C: Cache<T, NX, NU>,
    XProj: Project<T, NX, HX>,
    UProj: Project<T, NU, HU>,
{
    /// Set the reference for state variables
    pub fn x_reference(mut self, x_ref: &'a SMatrix<T, NX, HX>) -> Self {
        self.x_ref = Some(x_ref);
        self
    }

    /// Set the reference for input variables
    pub fn u_reference(mut self, u_ref: &'a SMatrix<T, NU, HU>) -> Self {
        self.u_ref = Some(u_ref);
        self
    }

    /// Set constraints on the state variables
    pub fn x_constraints<Proj: Project<T, NX, HX>>(
        self,
        x_con: &'a mut [Constraint<T, Proj, NX, HX>],
    ) -> Problem<'a, T, C, NX, NU, HX, HU, Proj, UProj> {
        Problem {
            mpc: self.mpc,
            x_now: self.x_now,
            x_ref: self.x_ref,
            u_ref: self.u_ref,
            x_con: Some(x_con),
            u_con: self.u_con,
        }
    }

    /// Set constraints on the input variables
    pub fn u_constraints<Proj: Project<T, NU, HU>>(
        self,
        u_con: &'a mut [Constraint<T, Proj, NU, HU>],
    ) -> Problem<'a, T, C, NX, NU, HX, HU, XProj, Proj> {
        Problem {
            mpc: self.mpc,
            x_now: self.x_now,
            x_ref: self.x_ref,
            u_ref: self.u_ref,
            x_con: self.x_con,
            u_con: Some(u_con),
        }
    }

    /// Run the solver
    pub fn solve(self) -> Solution<'a, T, NX, NU, HX, HU> {
        self.mpc
            .solve(self.x_now, self.x_ref, self.u_ref, self.x_con, self.u_con)
    }
}

impl<T, C: Cache<T, NX, NU>, const NX: usize, const NU: usize, const HX: usize, const HU: usize>
    TinyMpc<T, C, NX, NU, HX, HU>
where
    T: Scalar + RealField + Copy,
{
    #[must_use]
    #[inline(always)]
    pub fn new(A: SMatrix<T, NX, NX>, B: SMatrix<T, NX, NU>, cache: C) -> Self {
        // Compile-time guard against invalid horizon lengths
        const {
            assert!(
                HX > HU,
                "The prediction horizon `HX` must be larger than the control horizon `HU`"
            );
            assert!(HU > 0, "The control horizon `HU` must be non-zero");
        }

        Self {
            config: Config {
                prim_tol: convert(1e-2),
                dual_tol: convert(1e-2),
                max_iter: 50,
                do_check: 5,
                relaxation: T::one(),
            },
            cache,
            state: State {
                A,
                B,
                sys: None,
                cx: SMatrix::zeros(),
                cp: SMatrix::zeros(),
                q: SMatrix::zeros(),
                r: SMatrix::zeros(),
                p: SMatrix::zeros(),
                d: SMatrix::zeros(),
                ex: SMatrix::zeros(),
                eu: SMatrix::zeros(),
                iter: 0,
            },
        }
    }

    pub fn with_sys(mut self, sys: LtiFn<T, NX, NU>) -> Self {
        self.state.sys = Some(sys);
        self
    }

    #[inline(always)]
    pub fn initial_condition(
        &mut self,
        x_now: SVector<T, NX>,
    ) -> Problem<'_, T, C, NX, NU, HX, HU> {
        Problem {
            mpc: self,
            x_now,
            x_ref: None,
            u_ref: None,
            x_con: None,
            u_con: None,
        }
    }

    #[inline(always)]
    pub fn solve<'a>(
        &'a mut self,
        x_now: SVector<T, NX>,
        x_ref: Option<&'a SMatrix<T, NX, HX>>,
        u_ref: Option<&'a SMatrix<T, NU, HU>>,
        x_con: Option<&mut [Constraint<T, impl Project<T, NX, HX>, NX, HX>]>,
        u_con: Option<&mut [Constraint<T, impl Project<T, NU, HU>, NU, HU>]>,
    ) -> Solution<'a, T, NX, NU, HX, HU> {
        let mut reason = TerminationReason::MaxIters;

        // We flatten the None variant into an empty slice
        let x_con = x_con.unwrap_or(&mut [][..]);
        let u_con = u_con.unwrap_or(&mut [][..]);

        // Set initial error state and warm-start constraints
        self.set_initial_conditions(x_now, x_ref, u_ref);
        self.warm_start_constraints(x_con, u_con);

        let mut prim_residual = T::zero();
        let mut dual_residual = T::zero();

        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {
            profiling::scope!("solve loop", format!("iter: {}", self.state.iter));

            self.update_cost(x_con, u_con);

            self.backward_pass();

            self.forward_pass();

            self.update_constraints(x_ref, u_ref, x_con, u_con);

            if self.check_termination(&mut prim_residual, &mut dual_residual, x_con, u_con) {
                reason = TerminationReason::Converged;
                self.state.iter += 1;
                break;
            }

            self.state.iter += 1;
        }

        Solution {
            x_ref,
            u_ref,
            x: &self.state.ex,
            u: &self.state.eu,
            reason,
            iterations: self.state.iter,
            prim_residual,
            dual_residual: dual_residual * self.cache.get_active().rho,
        }
    }

    #[inline(always)]
    fn should_compute_residuals(&self) -> bool {
        self.state.iter % self.config.do_check == 0
    }

    #[inline(always)]
    #[profiling::function]
    fn set_initial_conditions(
        &mut self,
        x_now: SVector<T, NX>,
        x_ref: Option<&SMatrix<T, NX, HX>>,
        u_ref: Option<&SMatrix<T, NU, HU>>,
    ) {
        if let Some(x_ref) = x_ref {
            profiling::scope!("affine state reference term");
            x_now.sub_to(&x_ref.column(0), &mut self.state.ex.column_mut(0));
            self.state.A.mul_to(&x_ref, &mut self.state.cx);
            for i in 0..HX - 1 {
                let mut cx_col = self.state.cx.column_mut(i);
                cx_col.axpy(-T::one(), &x_ref.column(i + 1), T::one());
            }
        } else {
            self.state.ex.set_column(0, &x_now);
        }

        if let Some(u_ref) = u_ref {
            profiling::scope!("affine input reference term");
            for i in 0..HX - 1 {
                let mut cx_col = self.state.cx.column_mut(i);
                let u_ref_col = u_ref.column(i.min(HU - 1));
                cx_col.gemv(-T::one(), &self.state.B, &u_ref_col, T::one());
            }
        }

        self.update_tracking_mismatch_plqr();
    }

    #[inline(always)]
    fn update_tracking_mismatch_plqr(&mut self) {
        // Note: using `sygemv` to exploit the symmetry of Plqr is actually
        // slower than just doing a regular matrix-vector multiplication,
        // since sygemv adds additional indexing overhead.
        let cache = self.cache.get_active();
        cache.Plqr.mul_to(&self.state.cx, &mut self.state.cp);
    }

    /// Shift the dual variables by one time step for more accurate hot starting
    #[inline(always)]
    #[profiling::function]
    fn warm_start_constraints(
        &mut self,
        x_con: &mut [Constraint<T, impl Project<T, NX, HX>, NX, HX>],
        u_con: &mut [Constraint<T, impl Project<T, NU, HU>, NU, HU>],
    ) {
        for con in x_con {
            util::shift_columns_left(&mut con.dual);
            util::shift_columns_left(&mut con.slac);
        }

        for con in u_con {
            util::shift_columns_left(&mut con.dual);
            util::shift_columns_left(&mut con.slac);
        }
    }

    /// Update linear control cost terms based on constraint violations
    #[inline(always)]
    #[profiling::function]
    fn update_cost(
        &mut self,
        x_con: &mut [Constraint<T, impl Project<T, NX, HX>, NX, HX>],
        u_con: &mut [Constraint<T, impl Project<T, NU, HU>, NU, HU>],
    ) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        // Add cost contribution for state constraint violations
        let mut x_con_iter = x_con.iter_mut();
        if let Some(x_con_first) = x_con_iter.next() {
            profiling::scope!("update state cost");
            x_con_first.set_cost(&mut s.q);
            for x_con_next in x_con_iter {
                x_con_next.add_cost(&mut s.q);
            }
            s.q.scale_mut(c.rho);
        } else {
            s.q = SMatrix::<T, NX, HX>::zeros()
        }

        // Add cost contribution for input constraint violations
        let mut u_con_iter = u_con.iter_mut();
        if let Some(u_con_first) = u_con_iter.next() {
            profiling::scope!("update input cost");
            u_con_first.set_cost(&mut s.r);
            for u_con_next in u_con_iter {
                u_con_next.add_cost(&mut s.r);
            }
            s.r.scale_mut(c.rho);
        } else {
            s.r = SMatrix::<T, NU, HU>::zeros()
        }

        // Extract ADMM cost term for Riccati terminal condition
        s.p.set_column(HX - 1, &(s.q.column(HX - 1)));
    }

    /// Backward pass to update Ricatti variables
    #[inline(always)]
    #[profiling::function]
    fn backward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        for i in (0..HX - 1).rev() {
            let (mut p_now, mut p_fut) = util::column_pair_mut(&mut s.p, i, i + 1);
            let mut r_col = s.r.column_mut(i.min(HU - 1));

            // Reused calculation: [[[i+1]]] <- (p[i+1] + Plqr * w[i])
            p_fut.axpy(T::one(), &s.cp.column(i), T::one());

            // Calc: p[i] = AmBKt * [[[i+1]]] - Klqr' * r[:,u_index] + q[i]
            p_now.gemv(T::one(), &c.AmBKt, &p_fut, T::zero());
            p_now.gemv_tr(T::one(), &c.nKlqr, &r_col, T::one());
            p_now.axpy(T::one(), &s.q.column(i), T::one());

            if i < HU {
                let mut d_col = s.d.column_mut(i);

                // Calc: d[i] = RpBPBi * (B' * [[[i+1]]] + r[i])
                r_col.gemv_tr(T::one(), &s.B, &p_fut, T::one());
                d_col.gemv(T::one(), &c.RpBPBi, &r_col, T::zero());
            }
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    #[inline(always)]
    #[profiling::function]
    fn forward_pass(&mut self) {
        let s = &mut self.state;
        let c = self.cache.get_active();

        if let Some(system) = s.sys {
            // Roll out trajectory up to the control horizon (HU)
            for i in 0..HU {
                let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
                let mut u_col = s.eu.column_mut(i);

                u_col.gemv(T::one(), &c.nKlqr, &ex_now, T::zero());
                u_col.axpy(-T::one(), &s.d.column(i), T::one());

                system(ex_fut.as_view_mut(), ex_now.as_view(), u_col.as_view());
                ex_fut.axpy(T::one(), &s.cx.column(i), T::one());
            }

            // Roll out rest of trajectory keeping u constant
            for i in HU..HX - 1 {
                let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
                let u_col = s.eu.column(HU - 1);

                system(ex_fut.as_view_mut(), ex_now.as_view(), u_col.as_view());
                ex_fut.axpy(T::one(), &s.cx.column(i), T::one());
            }
        } else {
            // Roll out trajectory up to the control horizon (HU)
            for i in 0..HU {
                let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
                let mut u_col = s.eu.column_mut(i);

                // Calc: u[i] = -Klqr * ex[i] + d[i]
                u_col.gemv(T::one(), &c.nKlqr, &ex_now, T::zero());
                u_col.axpy(-T::one(), &s.d.column(i), T::one());

                // Calc x[i+1] = A * x[i] + B * u[i] + w[i]
                ex_fut.gemv(T::one(), &s.A, &ex_now, T::zero());
                ex_fut.gemv(T::one(), &s.B, &u_col, T::one());
                ex_fut.axpy(T::one(), &s.cx.column(i), T::one());
            }

            // Roll out rest of trajectory keeping u constant
            for i in HU..HX - 1 {
                let (ex_now, mut ex_fut) = util::column_pair_mut(&mut s.ex, i, i + 1);
                let u_col = s.eu.column(HU - 1);

                // Calc x[i+1] = A * x[i] + B * u[i] + w[i]
                ex_fut.gemv(T::one(), &s.A, &ex_now, T::zero());
                ex_fut.gemv(T::one(), &s.B, &u_col, T::one());
                ex_fut.axpy(T::one(), &s.cx.column(i), T::one());
            }
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    #[inline(always)]
    #[profiling::function]
    fn update_constraints(
        &mut self,
        x_ref: Option<&SMatrix<T, NX, HX>>,
        u_ref: Option<&SMatrix<T, NU, HU>>,
        x_con: &mut [Constraint<T, impl Project<T, NX, HX>, NX, HX>],
        u_con: &mut [Constraint<T, impl Project<T, NU, HU>, NU, HU>],
    ) {
        let compute_residuals = self.should_compute_residuals();
        let s = &mut self.state;

        let (x_points, u_points) = if self.config.relaxation != T::one() {
            profiling::scope!("apply relaxation to state and input");

            // Use Riccati matrices to store relaxed x and u matrices.
            s.q.copy_from(&s.ex);
            s.r.copy_from(&s.eu);

            let alpha = self.config.relaxation;

            s.q.scale_mut(alpha);
            s.r.scale_mut(alpha);

            for con in x_con.as_mut() {
                s.q += con.slac.scale(T::one() - alpha);
            }

            for con in u_con.as_mut() {
                s.r += con.slac.scale(T::one() - alpha);
            }

            // Buffers now contain: x' = alpha * x + (1 - alpha) * z
            (&s.q, &s.r)
        } else {
            // Just use original predictions
            (&s.ex, &s.eu)
        };

        // Use cost matrices as scratch buffers
        let u_scratch = &mut s.d;
        let x_scratch = &mut s.p;

        for con in x_con {
            con.constrain(compute_residuals, x_points, x_ref, x_scratch);
        }

        for con in u_con {
            con.constrain(compute_residuals, u_points, u_ref, u_scratch);
        }
    }

    /// Check for termination condition by evaluating residuals
    #[inline(always)]
    #[profiling::function]
    fn check_termination(
        &mut self,
        max_prim_residual: &mut T,
        max_dual_residual: &mut T,
        x_con: &mut [Constraint<T, impl Project<T, NX, HX>, NX, HX>],
        u_con: &mut [Constraint<T, impl Project<T, NU, HU>, NU, HU>],
    ) -> bool {
        let c = self.cache.get_active();
        let cfg = &self.config;

        if !self.should_compute_residuals() {
            return false;
        }

        *max_prim_residual = T::zero();
        *max_dual_residual = T::zero();

        for con in x_con.iter() {
            *max_prim_residual = (*max_prim_residual).max(con.max_prim_residual);
            *max_dual_residual = (*max_dual_residual).max(con.max_dual_residual);
        }

        for con in u_con.iter() {
            *max_prim_residual = (*max_prim_residual).max(con.max_prim_residual);
            *max_dual_residual = (*max_dual_residual).max(con.max_dual_residual);
        }

        let terminate =
            *max_prim_residual < cfg.prim_tol && *max_dual_residual * c.rho < cfg.dual_tol;

        // Try to adapt rho
        if !terminate
            && let Some(scalar) = self
                .cache
                .update_active(*max_prim_residual, *max_dual_residual)
        {
            profiling::scope!("cache updated, rescale all dual variables");

            self.update_tracking_mismatch_plqr();

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
    pub fn get_u_at(&self, i: usize) -> SVector<T, NU> {
        self.state.eu.column(i).into()
    }

    /// Get the system input `u` for the current time
    pub fn get_u(&self) -> SVector<T, NU> {
        self.state.eu.column(0).into()
    }

    /// Get reference to matrix containing state predictions
    pub fn get_x_matrix(&self) -> &SMatrix<T, NX, HX> {
        &self.state.ex
    }

    /// Get reference to matrix containing input predictions
    pub fn get_u_matrix(&self) -> &SMatrix<T, NU, HU> {
        &self.state.eu
    }
}

pub struct Solution<'a, T, const NX: usize, const NU: usize, const HX: usize, const HU: usize> {
    x_ref: Option<&'a SMatrix<T, NX, HX>>,
    u_ref: Option<&'a SMatrix<T, NU, HU>>,
    x: &'a SMatrix<T, NX, HX>,
    u: &'a SMatrix<T, NU, HU>,
    pub reason: TerminationReason,
    pub iterations: usize,
    pub prim_residual: T,
    pub dual_residual: T,
}

impl<T: RealField + Copy, const NX: usize, const NU: usize, const HX: usize, const HU: usize>
    Solution<'_, T, NX, NU, HX, HU>
{
    /// Get the predictiction of states for this solution
    pub fn x_prediction(&self) -> SMatrix<T, NX, HX> {
        if let Some(x_ref) = self.x_ref.as_ref() {
            self.x + *x_ref
        } else {
            self.x.clone_owned()
        }
    }

    /// Get the predictiction of inputs for this solution
    pub fn u_prediction(&self) -> SMatrix<T, NU, HU> {
        if let Some(u_ref) = self.u_ref.as_ref() {
            self.u + *u_ref
        } else {
            self.u.clone_owned()
        }
    }

    /// Get the current contron input to be applied
    pub fn u_now(&self) -> SVector<T, NU> {
        if let Some(u_ref) = self.u_ref.as_ref() {
            self.u.column(0) + u_ref.column(0)
        } else {
            self.u.column(0).into()
        }
    }
}
