use nalgebra::{RealField, SMatrix, Scalar, convert};

/// Errors that can occur during cache setup
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Error {
    /// The value of rho must be strictly positive `(rho > 0)`
    RhoNotPositive,
    /// The matrix `R_aug + B^T * P * B` is not invertible
    RpBPBNotInvertible,
    /// The resulting matrices contained non-finite elements (Inf or NaN)
    NonFiniteValues,
}

pub trait Cache<T, const NX: usize, const NU: usize> {
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

    /// Infinite-time horizon LQR cost-to-go
    pub(crate) Plqr: SMatrix<T, NX, NX>,

    /// Precomputed `inv(R_aug + B^T * Plqr * B)`
    pub(crate) RpBPBi: SMatrix<T, NU, NU>,

    /// Precomputed `(A - B * Klqr)^T`
    pub(crate) AmBKt: SMatrix<T, NX, NX>,
}

impl<T, const NX: usize, const NU: usize> SingleCache<T, NX, NU>
where
    T: Scalar + RealField + Copy,
{
    pub fn new(
        rho: T,
        iters: usize,
        A: &SMatrix<T, NX, NX>,
        B: &SMatrix<T, NX, NU>,
        Q: &SMatrix<T, NX, NX>,
        R: &SMatrix<T, NU, NU>,
        S: &SMatrix<T, NX, NU>,
    ) -> Result<Self, Error> {
        if !rho.is_positive() {
            return Err(Error::RhoNotPositive);
        }

        let Q = Q.symmetric_part();
        let R = R.symmetric_part();

        // ADMM-augmented cost matrices for LQR problem
        let Q_aug = Q + SMatrix::from_diagonal_element(rho);
        let R_aug = R + SMatrix::from_diagonal_element(rho);

        let mut Klqr = SMatrix::zeros();
        let mut Plqr = Q_aug.clone_owned();

        for _ in 0..iters {
            Klqr = (R_aug + B.transpose() * Plqr * B)
                .try_inverse()
                .ok_or(Error::RpBPBNotInvertible)?
                * (S.transpose() + B.transpose() * Plqr * A);
            Plqr = A.transpose() * Plqr * A - A.transpose() * Plqr * B * Klqr + Q_aug;
        }

        let RpBPBi = (R_aug + B.transpose() * Plqr * B)
            .try_inverse()
            .ok_or(Error::RpBPBNotInvertible)?;
        let AmBKt = (A - B * Klqr).transpose();
        let nKlqr = -Klqr;

        // If RpBPBi and AmBKt are finite, so are all the other values
        ([].iter())
            .chain(RpBPBi.iter())
            .chain(AmBKt.iter())
            .all(|x| x.is_finite())
            .then_some(SingleCache {
                rho,
                nKlqr,
                Plqr,
                RpBPBi,
                AmBKt,
            })
            .ok_or(Error::NonFiniteValues)
    }
}

impl<T, const NX: usize, const NU: usize> Cache<T, NX, NU> for SingleCache<T, NX, NU>
where
    T: Scalar + RealField + Copy,
{
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

impl<T, const NX: usize, const NU: usize, const NUM: usize> ArrayCache<T, NX, NU, NUM>
where
    T: Scalar + RealField + Copy,
{
    pub fn new(
        central_rho: T,
        threshold: T,
        factor: T,
        iters: usize,
        A: &SMatrix<T, NX, NX>,
        B: &SMatrix<T, NX, NU>,
        Q: &SMatrix<T, NX, NX>,
        R: &SMatrix<T, NU, NU>,
        S: &SMatrix<T, NX, NU>,
    ) -> Result<Self, Error> {
        let active_index = NUM / 2;
        let caches = crate::util::try_array_from_fn(|index| {
            let diff = index as i32 - active_index as i32;
            let mult = factor.powf(convert(diff as f64));
            let rho = central_rho * mult;
            SingleCache::new(rho, iters, A, B, Q, R, S)
        })?;

        Ok(Self {
            threshold,
            active_index,
            caches,
        })
    }
}

impl<T, const NX: usize, const NU: usize, const NUM: usize> Cache<T, NX, NU>
    for ArrayCache<T, NX, NU, NUM>
where
    T: Scalar + RealField + Copy,
{
    #[inline(always)]
    fn update_active(&mut self, prim_residual: T, dual_residual: T) -> Option<T> {
        let mut cache = &self.caches[self.active_index];
        let prev_rho = cache.rho;

        // TODO: It seems to work better without this?
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
