// #![no_std]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/*

    This work is heavily based off of TinyMPC [ https://tinympc.org/ ]

*/

use nalgebra::{convert, RealField, SMatrix, SMatrixView, SVector, SVectorView, Scalar};

use crate::constraint::{Constraint, Project};

pub mod constraint;

pub type LtiFn<F, const Nx: usize, const Nu: usize> = fn(SVectorView<F, Nx>, SVectorView<F, Nu>) -> SVector<F, Nx>;

/// Errors that can occur during system setup
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Error {
    InvalidHorizonLengths,
    RhoNotPositiveDefinite,
    QNotPositiveSemidefinite,
    RNotPositiveDefinite,
    RpBPBNotInvertible,
    NonFiniteValues,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TerminationReason {
    Converged,
    MaxIters,
}

#[derive(Debug)]
pub struct TinyMpc<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    pub config: Config<F>,
    cache: Cache<Nx, Nu, F>,
    state: State<Nx, Nu, Hx, Hu, F>,
}

#[derive(Debug)]
pub struct Config<F> {
    /// The convergence tolerance for the primal residual (default 0.001)
    pub prim_tol: F,
    
    /// The convergence tolerance for the dual residual  (default 0.001)
    pub dual_tol: F,
    
    /// Maximum iterations without converging before terminating (default 50)
    pub max_iter: usize,

    /// Number of iterations between evaluating convergence (default 5)
    pub do_check: usize,
}

/// Contains all pre-computed values for a given problem and value of rho
#[derive(Debug)]
pub struct Cache<const Nx: usize, const Nu: usize, F> {
    /// Penalty-parameter for this cache
    rho: F,

    /// Augmented state penalty vector
    Q_aug: SVector<F, Nx>,

    /// Augmented input penalty vector
    R_aug: SVector<F, Nu>,

    /// Infinite-time horizon LQR gain
    Klqr: SMatrix<F, Nu, Nx>,

    /// Transpoed infinite-time horizon LQR gain
    Klqrt: SMatrix<F, Nx, Nu>,

    /// Infinite-time horizon LQR cost-to-go
    Plqr: SMatrix<F, Nx, Nx>,

    /// Precomputed `inv(R_aug + B^T * Plqr * B)`
    RpBPBi: SMatrix<F, Nu, Nu>,

    /// Precomputed `(A - B * Klqr)^T`
    AmBKt: SMatrix<F, Nx, Nx>,
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

        if !rho.is_positive() {
            return Err(Error::RhoNotPositiveDefinite);
        }

        if !Q.iter().all(|q| q >= &T::zero()) {
            return Err(Error::QNotPositiveSemidefinite);
        }

        if !R.iter().all(|r| r > &T::zero()) {
            return Err(Error::RNotPositiveDefinite);
        }

        let Q_aug = Q.add_scalar(rho);
        let R_aug = R.add_scalar(rho);

        let Q_diag = SMatrix::from_diagonal(&Q_aug);
        let R_diag = SMatrix::from_diagonal(&R_aug);

        let mut Klqr = SMatrix::zeros();
        let mut Plqr = Q_diag.clone_owned();

        const INVERR: Error = Error::RpBPBNotInvertible;

        for _ in 0..iters {
            Klqr = (R_diag + B.transpose() * Plqr * B).try_inverse().ok_or(INVERR)? * (B.transpose() * Plqr * A);
            Plqr = A.transpose() * Plqr * A - A.transpose() * Plqr * B * Klqr + Q_diag;
        }

        let Klqrt = Klqr.transpose();

        let RpBPBi = (R_diag + B.transpose() * Plqr * B).try_inverse().ok_or(INVERR)?;
        let AmBKt = (A - B * Klqr).transpose();

        // If RpBPBi and AmBKt are finite, so are all the other values
        ([].iter())
            .chain(RpBPBi.iter())
            .chain(AmBKt.iter())
            .all(|x| x.is_finite())
            .then(|| Cache {
                rho,
                Q_aug,
                R_aug,
                Klqr,
                Klqrt,
                Plqr,
                RpBPBi,
                AmBKt,
            })
            .ok_or(Error::NonFiniteValues)
    }
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
pub struct State<const Nx: usize, const Nu: usize, const Hx: usize, const Hu: usize, F> {
    // Linear state space model
    A: SMatrix<F, Nx, Nx>,
    B: SMatrix<F, Nx, Nu>,
    Bt: SMatrix<F, Nu, Nx>,

    /// In case A and B are sparse, a function doing the
    /// `x' <- A*x + B*u` calculation manually is likely faster.
    sys: Option<LtiFn<F, Nx, Nu>>,

    // State and input predictions
    x: SMatrix<F, Nx, Hx>,
    u: SMatrix<F, Nu, Hu>,

    // Scratch buffers for zero allocation operations
    x_scratch: SMatrix<F, Nx, Hx>,
    u_scratch: SMatrix<F, Nu, Hu>,

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
                max_iter: 50,
                do_check: 5,
            },
            cache: Cache::compute(rho, Hx, &A, &B, &Q, &R)?,
            state: State {
                A,
                B,
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

    pub fn with_sys(mut self, sys: LtiFn<F, Nx, Nu>) -> Self {
        self.state.sys = Some(sys);
        self
    }

    #[inline(never)]
    pub fn solve(
        &mut self,
        xnow: SVector<F, Nx>,
        xref: Option<SMatrixView<'_, F, Nx, Hx>>,
        uref: Option<SMatrixView<'_, F, Nu, Hu>>,
        xcon: Option<&mut [&mut Constraint<F, impl Project<F, Nx, Hx>, Nx, Hx>]>,
        ucon: Option<&mut [&mut Constraint<F, impl Project<F, Nu, Hu>, Nu, Hu>]>,
    ) -> (TerminationReason, SVector<F, Nu>) {
        let mut reason = TerminationReason::MaxIters;

        let mut xcon = xcon.unwrap_or(&mut [][..]);
        let mut ucon = ucon.unwrap_or(&mut [][..]);

        // Better warm-starting of dual variables from prior solution
        timed!{
            "shift constraint variables"
            self.shift_constraint_variables(&mut xcon, &mut ucon);
        }

        // Iteratively solve MPC problem
        self.state.iter = 0;
        while self.state.iter < self.config.max_iter {

            // defmt::debug!("Iteration number: {}", self.state.iter);

            timed!{
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
    fn shift_constraint_variables(&mut self,
        xcon: &mut [&mut Constraint<F, impl Project<F, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<F, impl Project<F, Nu, Hu>, Nu, Hu>],
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
    fn update_cost(&mut self, 
        xref: Option<SMatrixView<F, Nx, Hx>>,
        uref: Option<SMatrixView<F, Nu, Hu>>,
        xcon: &mut [&mut Constraint<F, impl Project<F, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<F, impl Project<F, Nu, Hu>, Nu, Hu>],
    ) {
        let s = &mut self.state;
        let c = &self.cache;

        s.x_cost = SMatrix::<F, Nx, Hx>::zeros();
        s.u_cost = SMatrix::<F, Nu, Hu>::zeros();

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
                x_cost_col -= &xref.column(i).component_mul(&c.Q_aug);
            }
        }
        
        if let Some(uref) = uref {
            for i in 0..Hu {
                let mut u_cost_col = s.u_cost.column_mut(i);
                u_cost_col -= &uref.column(i).component_mul(&c.R_aug);
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
        let c = &self.cache;

        // Scratch vectors for zero-allocation, minimal copy operations
        let mut u_scratch_vec = s.u_scratch.column_mut(0);

        // Handles this special case correctly.
        // This will be optimized away in the case they are different.
        if Hu == Hx {
            let i = Hu - 1;

            // Update: u_ricc[i] = RpBPBi * (B^T * x_ricc[i] + u_cost[i])
            s.Bt.mul_to(&s.x_ricc.column(i), &mut u_scratch_vec);
            u_scratch_vec += &s.u_cost.column(i);
            c.RpBPBi.mul_to(&u_scratch_vec, &mut s.u_ricc.column_mut(i));
        }

        // The backward pass integrates cost-to-go over the full prediction horizon Hx
        for i in (0..Hx - 1).rev() {
            let x_ricc_next = s.x_ricc.column(i + 1);

            // Control action is only optimized up to Hu
            if i < Hu {
                // Update: u_ricc[i] = RpBPBi * (B^T * x_ricc[i] + u_cost[i])
                s.Bt.mul_to(&x_ricc_next, &mut u_scratch_vec);
                u_scratch_vec += &s.u_cost.column(i);
                c.RpBPBi.mul_to(&u_scratch_vec, &mut s.u_ricc.column_mut(i));

                // Update: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i+1] + Klqr^T * u_cost[i]
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x_scratch.column_mut(0));
                c.Klqrt.mul_to(&s.u_cost.column(i), &mut s.x_scratch.column_mut(1));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost.column(i).add_to(&s.x_scratch.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x_scratch.column(1);

            } else {

                // Update: x_ricc[i] = x_cost[i] + AmBKt * x_ricc[i+1] + Klqr^T * u_cost[Hu - 1]
                c.Klqrt.mul_to(&s.u_cost.column(Hu - 1), &mut s.x_scratch.column_mut(1));
                c.AmBKt.mul_to(&x_ricc_next, &mut s.x_scratch.column_mut(0));
                let mut x_ricc_vec = s.x_ricc.column_mut(i);
                s.x_cost.column(i).add_to(&s.x_scratch.column(0), &mut x_ricc_vec);
                x_ricc_vec -= &s.x_scratch.column(1);
            }
        }
    }

    /// Use LQR feedback policy to roll out trajectory
    #[inline(never)]
    fn forward_pass(&mut self, mut xnow: SVector<F, Nx>) {
        let s = &mut self.state;
        let c = &self.cache;

        let neg_Klqr = -c.Klqr;

        if let Some(sys) = s.sys {
            // Forward-pass with initial state
            s.u.set_column(0, &(-c.Klqr * xnow - s.u_ricc.column(0)));
            s.x.set_column(0, &sys(xnow.as_view(), s.u.column(0)));

            // Roll out trajectory up to the control horizon Hu
            for i in 1..Hu {
                s.u.set_column(i, &(-c.Klqr * s.x.column(i - 1) - s.u_ricc.column(i)));
                s.x.set_column(i, &sys(s.x.column(i - 1), s.u.column(i)));
            }

            // For the rest of the prediction horizon (Hx), hold the last control input constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx {
                s.x.set_column(i, &sys(s.x.column(i - 1), u_final));
            }
        } else {
            // Forward-pass of u[0] with initial state x[0]
            let mut u_col = s.u.column_mut(0);
            neg_Klqr.mul_to(&xnow, &mut u_col);
            u_col -= &s.u_ricc.column(0);

            // Forward-pass of x[1] with initial state x[0] and u[0]
            let mut x_col = s.x.column_mut(0);
            s.A.mul_to(&xnow, &mut x_col);
            x_col.gemm(F::one(), &s.B, &u_col, F::one());

            // Roll out trajectory up to the control horizon Hu
            for i in 1..Hu {

                xnow.copy_from(&s.x.column(i-1).clone_owned());

                // Forward-pass of u[i] with state x[i]
                let mut u_col = s.u.column_mut(i);
                neg_Klqr.mul_to(&xnow, &mut u_col);
                u_col -= &s.u_ricc.column(i);

                // Forward-pass of x[i + 1] with state x[i] and u[i]
                let mut x_col = s.x.column_mut(i);
                s.A.mul_to(&xnow, &mut x_col);
                x_col.gemm(F::one(), &s.B, &u_col, F::one());
            }

            // Roll out rest of trajectory keeping u constant
            let u_final = s.u.column(Hu - 1);
            for i in Hu..Hx {
                
                xnow.copy_from(&s.x.column(i-1).clone_owned());
                
                // Forward-pass of x[i + 1] with state x[i] and u[Hu - 1]
                let mut x_col = s.x.column_mut(i);
                s.A.mul_to(&xnow, &mut x_col);
                x_col.gemm(F::one(), &s.B, &u_final, F::one());
            }
        }
    }

    /// Project slack variables into their feasible domain and update dual variables
    #[inline(never)]
    fn update_constraints(&mut self,
        xcon: &mut [&mut Constraint<F, impl Project<F, Nx, Hx>, Nx, Hx>],
        ucon: &mut [&mut Constraint<F, impl Project<F, Nu, Hu>, Nu, Hu>],
    ) {
        let update_res = self.should_calculate_residuals();
        let s = &mut self.state;

        for con in xcon {
            con.constrain(s.x.as_view(), s.x_scratch.as_view_mut(), update_res);
        }

        for con in ucon {
            con.constrain(s.u.as_view(), s.u_scratch.as_view_mut(), update_res);
        }
    }

    /// Check for termination condition by evaluating residuals
    #[inline(never)]
    fn check_termination(
        &mut self,
        xcon: &[&mut Constraint<F, impl Project<F, Nx, Hx>, Nx, Hx>],
        ucon: &[&mut Constraint<F, impl Project<F, Nu, Hu>, Nu, Hu>],
    ) -> bool {
        let c = &self.cache;
        let cfg = &self.config;

        if !self.should_calculate_residuals() {
            return false
        }

        let mut max_prim_residual = F::zero();
        let mut max_dual_residual = F::zero();

        for con in xcon.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
        }

        for con in ucon.iter() {
            max_prim_residual = max_prim_residual.max(con.max_prim_residual);
            max_dual_residual = max_dual_residual.max(con.max_dual_residual);
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
