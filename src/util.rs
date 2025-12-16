use nalgebra::{Const, Matrix, RealField, SMatrix, SVectorViewMut, Scalar, ViewStorageMut};

/// Helper function to get a mutable view into two colums of a matrix.
///
/// # Panics
///
///  if indices are out of bounds or if they are the same
#[inline(always)]
pub(crate) fn column_pair_mut<T: Scalar, const R: usize, const C: usize>(
    matrix: &mut SMatrix<T, R, C>,
    column0: usize,
    column1: usize,
) -> (SVectorViewMut<'_, T, R>, SVectorViewMut<'_, T, R>) {
    assert_ne!(column0, column1, "columns must not be same");
    assert!(column0 < C, "column0 is out of bounds");
    assert!(column1 < C, "column1 is out of bounds");

    let ptr = matrix.as_mut_ptr();

    let shape = (Const::<R>, Const::<1>);
    let strides = (Const::<1>, Const::<R>);

    let ptr0 = unsafe { ptr.add(R * column0) };
    let ptr1 = unsafe { ptr.add(R * column1) };

    let view0 = unsafe { ViewStorageMut::from_raw_parts(ptr0, shape, strides) };
    let view1 = unsafe { ViewStorageMut::from_raw_parts(ptr1, shape, strides) };

    (Matrix::from_data(view0), Matrix::from_data(view1))
}

/// Shifts all columns such that `column[i] <- column[i + 1]` with the last two being identical.
#[inline(always)]
pub(crate) fn shift_columns_left<T: Scalar, const R: usize, const C: usize>(
    matrix: &mut SMatrix<T, R, C>,
) {
    if C > 1 {
        let element_count = R * (C - 1);
        let ptr = matrix.as_mut_ptr();

        unsafe {
            core::ptr::copy(ptr.add(R), ptr, element_count);
        }
    }
}

/// Creates an array from a closure that can fail.
///
/// If a closure returns `Err`, this function will return that `Err`.
/// All previously initialized elements will be properly dropped.
#[inline(always)]
pub(crate) fn try_array_from_fn<T: Sized, E, const N: usize>(
    mut cb: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    use core::mem::MaybeUninit;

    let mut array = [const { MaybeUninit::<T>::uninit() }; N];

    for i in 0..N {
        match cb(i) {
            Ok(val) => {
                // If the closure succeeds, write the value to the array.
                array[i].write(val);
            }
            Err(e) => {
                // If the closure fails, we must drop the initialized elements.
                for element in array.iter_mut().take(i) {
                    unsafe {
                        element.assume_init_drop();
                    }
                }
                // Return the error to the caller.
                return Err(e);
            }
        }
    }

    // Safety: We've just initialized every element in the loop above.
    Ok(unsafe { array.map(|elem| elem.assume_init()) })
}

#[profiling::function]
pub(crate) fn frobenius_norm<'a, T, const N: usize, const H: usize>(mat: &SMatrix<T, N, H>) -> T
where
    T: RealField + Copy,
{
    mat.iter().copied().fold(T::zero(), |a, b| a + b * b).sqrt()
}
