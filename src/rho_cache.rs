use nalgebra::{RealField, SMatrix, SVector, Scalar, convert};

use crate::Error;

pub trait Cache<T, const Nx: usize, const Nu: usize>: Sized {
    fn new(
        rho: T,
        iters: usize,
        A: &SMatrix<T, Nx, Nx>,
        B: &SMatrix<T, Nx, Nu>,
        Q: &SVector<T, Nx>,
        R: &SVector<T, Nu>,
    ) -> Result<Self, Error>;

    /// Updates which cache is active by evaluating the primal and dual residuals.
    ///
    /// Returns: A scalar (old_rho/new_rho) to be applied to constraint duals in case the cache changed
    fn update_active(&mut self, prim_residual: T, dual_residual: T) -> Option<T>;

    /// Get a reference to the currently active cache.
    fn get_active(&self) -> &SingleCache<T, Nx, Nu>;
}

/// Contains all pre-computed values for a given problem and value of rho
#[derive(Debug)]
pub struct SingleCache<T, const Nx: usize, const Nu: usize> {
    /// Penalty-parameter for this cache
    pub(crate) rho: T,

    /// Infinite-time horizon LQR gain
    pub(crate) Klqr: SMatrix<T, Nu, Nx>,

    /// Infinite-time horizon LQR cost-to-go
    pub(crate) Plqr: SMatrix<T, Nx, Nx>,

    /// Precomputed `inv(R_aug + B^T * Plqr * B)`
    pub(crate) RpBPBi: SMatrix<T, Nu, Nu>,

    /// Precomputed `(A - B * Klqr)^T`
    pub(crate) AmBKt: SMatrix<T, Nx, Nx>,
}

impl<T: Scalar + RealField + Copy, const Nx: usize, const Nu: usize> Cache<T, Nx, Nu>
    for SingleCache<T, Nx, Nu>
{
    fn new(
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
            Klqr = (R_diag + B.transpose() * Plqr * B)
                .try_inverse()
                .ok_or(INVERR)?
                * (B.transpose() * Plqr * A);
            Plqr = A.transpose() * Plqr * A - A.transpose() * Plqr * B * Klqr + Q_diag;
        }

        let RpBPBi = (R_diag + B.transpose() * Plqr * B)
            .try_inverse()
            .ok_or(INVERR)?;
        let AmBKt = (A - B * Klqr).transpose();

        // If RpBPBi and AmBKt are finite, so are all the other values
        ([].iter())
            .chain(RpBPBi.iter())
            .chain(AmBKt.iter())
            .all(|x| x.is_finite())
            .then_some(SingleCache {
                rho,
                Klqr,
                Plqr,
                RpBPBi,
                AmBKt,
            })
            .ok_or(Error::NonFiniteValues)
    }

    fn update_active(&mut self, _prim_residual: T, _dual_residual: T) -> Option<T> {
        None
    }

    fn get_active(&self) -> &SingleCache<T, Nx, Nu> {
        self
    }
}

/// Contains all pre-computed values for a given problem and value of rho
#[derive(Debug)]
pub struct LookupCache<T, const Nx: usize, const Nu: usize, const NUM: usize> {
    threshold: T,
    active_index: usize,
    caches: [SingleCache<T, Nx, Nu>; NUM],
}

impl<T, const Nx: usize, const Nu: usize, const NUM: usize> Cache<T, Nx, Nu>
    for LookupCache<T, Nx, Nu, NUM>
where
    T: Scalar + RealField + Copy,
{
    fn new(
        central_rho: T,
        iters: usize,
        A: &SMatrix<T, Nx, Nx>,
        B: &SMatrix<T, Nx, Nu>,
        Q: &SVector<T, Nx>,
        R: &SVector<T, Nu>,
    ) -> Result<Self, Error> {
        let threshold = convert(15.0);
        let active_index = NUM / 2;

        let caches = crate::util::try_array_from_fn(|index| {
            let diff = index as i32 - active_index as i32;
            let expo = convert::<f64, T>(1.6).powf(convert(diff as f64));
            let rho = central_rho * expo;
            SingleCache::new(rho, iters, A, B, Q, R) // returns error
        })?;

        Ok(Self {
            threshold,
            active_index,
            caches,
        })
    }

    fn update_active(&mut self, prim_residual: T, dual_residual: T) -> Option<T> {
        let mut cache = &self.caches[self.active_index];
        let prev_rho = cache.rho;

        // Since we are using a scaled dual formulation
        let dual_residual = dual_residual * prev_rho;

        // For much larger primal residuals, increase rho
        if prim_residual > dual_residual * self.threshold {
            if self.active_index < NUM - 1 {
                self.active_index += 1;
                cache = &self.caches[self.active_index];
                println!(
                    "+ Increasing rho to: {} (index: {})",
                    cache.rho, self.active_index
                );
            }
        }
        // For much larger dual residuals, decrease rho
        else if dual_residual > prim_residual * self.threshold {
            if self.active_index > 0 {
                self.active_index -= 1;
                cache = &self.caches[self.active_index];
                println!(
                    "- Decreasing rho to: {} (index: {})",
                    cache.rho, self.active_index
                );
            }
        }

        // If the value of rho changed we must also rescale all duals
        (prev_rho != cache.rho).then(|| prev_rho / cache.rho)
    }

    fn get_active(&self) -> &SingleCache<T, Nx, Nu> {
        &self.caches[self.active_index]
    }
}
