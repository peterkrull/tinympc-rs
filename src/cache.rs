use nalgebra::{RealField, SMatrix, SVector, Scalar, convert};

use crate::Error;

pub trait Cache<T, const NX: usize, const NU: usize>: Sized {
    fn new(
        rho: T,
        iters: usize,
        A: &SMatrix<T, NX, NX>,
        B: &SMatrix<T, NX, NU>,
        Q: &SVector<T, NX>,
        R: &SVector<T, NU>,
    ) -> Result<Self, Error>;

    /// Updates which cache is active by evaluating the primal and dual residuals.
    ///
    /// Returns: A scalar (old_rho/new_rho) to be applied to constraint duals in case the cache changed
    fn update_active(&mut self, prim_residual: T, dual_residual: T) -> Option<T>;

    /// Get a reference to the currently active cache.
    fn get_active(&self) -> &SingleCache<T, NX, NU>;
}

/// Contains all pre-computed values for a given problem and value of rho
#[derive(Debug)]
pub struct SingleCache<T, const NX: usize, const NU: usize> {
    /// Penalty-parameter for this cache
    pub(crate) rho: T,

    /// (Negated) Infinite-time horizon LQR gain
    pub(crate) nKlqr: SMatrix<T, NU, NX>,

    /// Transposed Infinite-time horizon LQR gain
    pub(crate) nKlqrt: SMatrix<T, NX, NU>,

    /// Infinite-time horizon LQR cost-to-go
    pub(crate) Plqr: SMatrix<T, NX, NX>,

    /// Precomputed `inv(R_aug + B^T * Plqr * B)`
    pub(crate) RpBPBi: SMatrix<T, NU, NU>,

    /// Precomputed `(A - B * Klqr)^T`
    pub(crate) AmBKt: SMatrix<T, NX, NX>,
}

impl<T: Scalar + RealField + Copy, const NX: usize, const NU: usize> Cache<T, NX, NU>
    for SingleCache<T, NX, NU>
{
    fn new(
        rho: T,
        iters: usize,
        A: &SMatrix<T, NX, NX>,
        B: &SMatrix<T, NX, NU>,
        Q: &SVector<T, NX>,
        R: &SVector<T, NU>,
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
                nKlqr: -Klqr,
                nKlqrt: -Klqr.transpose(),
                Plqr,
                RpBPBi,
                AmBKt,
            })
            .ok_or(Error::NonFiniteValues)
    }

    fn update_active(&mut self, _prim_residual: T, _dual_residual: T) -> Option<T> {
        None
    }

    fn get_active(&self) -> &SingleCache<T, NX, NU> {
        self
    }
}

/// Contains an array of pre-computed values for a given problem and value of rho
#[derive(Debug)]
pub struct ArrayCache<T, const NX: usize, const NU: usize, const NUM: usize> {
    threshold: T,
    active_index: usize,
    caches: [SingleCache<T, NX, NU>; NUM],
}

impl<T, const NX: usize, const NU: usize, const NUM: usize> Cache<T, NX, NU>
    for ArrayCache<T, NX, NU, NUM>
where
    T: Scalar + RealField + Copy,
{
    fn new(
        central_rho: T,
        iters: usize,
        A: &SMatrix<T, NX, NX>,
        B: &SMatrix<T, NX, NU>,
        Q: &SVector<T, NX>,
        R: &SVector<T, NU>,
    ) -> Result<Self, Error> {
        let threshold = convert(10.0);
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

    #[inline(always)]
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
            }
        }
        // For much larger dual residuals, decrease rho
        else if dual_residual > prim_residual * self.threshold {
            if self.active_index > 0 {
                self.active_index -= 1;
                cache = &self.caches[self.active_index];
            }
        }

        // If the value of rho changed we must also rescale all duals
        (prev_rho != cache.rho).then(|| prev_rho / cache.rho)
    }

    #[inline(always)]
    fn get_active(&self) -> &SingleCache<T, NX, NU> {
        &self.caches[self.active_index]
    }
}
