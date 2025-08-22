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

    /// Negated infinite-time horizon LQR gain
    pub(crate) negKlqr: SMatrix<T, Nu, Nx>,

    /// Transposed infinite-time horizon LQR gain
    pub(crate) Klqrt: SMatrix<T, Nx, Nu>,

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

        let Q_diag = SMatrix::from_diagonal(&Q.add_scalar(rho));
        let R_diag = SMatrix::from_diagonal(&R.add_scalar(rho));

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

        let Klqrt = Klqr.transpose();
        let negKlqr = - Klqr;

        let RpBPBi = (R_diag + B.transpose() * Plqr * B)
            .try_inverse()
            .ok_or(INVERR)?;
        let AmBKt = (A - B * Klqr).transpose();

        // If RpBPBi and AmBKt are finite, so are all the other values
        ([].iter())
            .chain(RpBPBi.iter())
            .chain(AmBKt.iter())
            .all(|x| x.is_finite())
            .then(|| SingleCache {
                rho,
                negKlqr,
                Klqrt,
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

use std::mem::MaybeUninit;

/// Creates an array from a closure that can fail.
///
/// If the closure returns `Err` for any element, this function will return that `Err`.
/// All previously initialized elements will be properly dropped.
pub fn try_array_from_fn<T: Sized, E, const N: usize>(
    mut cb: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    // Create an uninitialized array of `MaybeUninit`.
    let mut array = [const { MaybeUninit::<T>::uninit() }; N];

    for i in 0..N {
        match cb(i) {
            Ok(val) => {
                // If the closure succeeds, write the value to the array.
                array[i].write(val);
            }
            Err(e) => {
                // If the closure fails, we must drop the elements that
                // were already successfully initialized.
                // The slice `0..i` contains all the initialized elements.
                for j in 0..i {
                    // Safety: We know elements 0..i have been initialized.
                    unsafe {
                        array[j].assume_init_drop();
                    }
                }
                // Return the error to the caller.
                return Err(e);
            }
        }
    }

    // If the loop completes, all elements are initialized, and we can
    // safely transition from `[MaybeUninit<T>; N]` to `[T; N]`.
    // Safety: We've just initialized every element in the loop above.
    let array = unsafe { array.map(|elem| elem.assume_init()) };

    Ok(array)
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
        let threshold = convert(10.0);
        let active_index = NUM / 2;

        let caches = try_array_from_fn(|index| {
            let diff = index as i32 - active_index as i32;
            let expo = convert::<f64, T>(1.6).powf(convert(diff as f64));
            let rho = central_rho * expo;
            println!("Creating cache for rho {rho}, at index {index} (expo {expo})");
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
        if prim_residual * self.threshold > dual_residual {
            self.active_index = (self.active_index + 1).min(NUM - 1);
            cache = &self.caches[self.active_index];
        }
        // For much larger dual residuals, decrease rho
        else if dual_residual * self.threshold > prim_residual {
            self.active_index = self.active_index.saturating_sub(1);
            cache = &self.caches[self.active_index];
        }

        // If the value of rho changed we must also rescale all duals
        let rescale = (prev_rho != cache.rho).then(|| prev_rho / cache.rho);

        rescale
    }

    fn get_active(&self) -> &SingleCache<T, Nx, Nu> {
        &self.caches[self.active_index]
    }
}
