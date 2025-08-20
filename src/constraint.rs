use nalgebra::{RealField, SMatrix, SVector};

pub trait Project<T, const N: usize, const H: usize> {
    /// Applies the projection to a series of points, modifying them in place
    fn project(&self, points: &mut SMatrix<T, N, H>);
}

impl <T, const N: usize, const H: usize> Project<T, N, H> for &dyn Project<T, N, H> {
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        (**self).project(points);
    }
}

impl <P: Project<T, N, H>, T, const N: usize, const H: usize> Project<T, N, H> for &P  {
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        (**self).project(points);
    }
}

impl <'a, P: Project<T, N, H>, T: RealField + Copy, const N: usize, const H: usize> From<&'a P> for Constraint<T, &'a dyn Project<T, N, H>, N, H> {
    fn from(value: &'a P) -> Self {
        Constraint::new(value as &dyn Project<T, N, H>)
    }
}

pub trait ProjectExt<T: RealField + Copy, const N: usize, const H: usize>: Project<T, N, H> + Sized {
    fn into_constraint(&self) -> Constraint<T, &Self, N, H> {
        Constraint::new(self)
    }

    fn into_dyn_constraint(&self) -> DynConstraint<'_, T, N, H> {
        Constraint::new(self as &dyn Project<T, N, H>)
    }
}

impl <S: Project<T, N, H>, T: RealField + Copy, const N: usize, const H: usize> ProjectExt<T, N, H> for S {}

/// Simple box constraint that is constant throughout the horizon.
pub struct BoxFixed<T, const N: usize> {
    pub lower: SVector<Option<T>, N>,
    pub upper: SVector<Option<T>, N>,
}

impl <T: RealField, const N: usize> BoxFixed<T, N> {
    /// Construct a new unconstrained `BoxProjection` that is 
    pub fn new() -> Self {
        Self {
            lower: SVector::from_element(None),
            upper: SVector::from_element(None),
        }
    }

    pub fn with_upper(self, upper: impl Into<SVector<Option<T>, N>>) -> Self {
        Self {
            upper: upper.into(),
            ..self
        }
    }

    pub fn with_lower(self, lower: impl Into<SVector<Option<T>, N>>) -> Self {
        Self {
            lower: lower.into(),
            ..self
        }
    }
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for BoxFixed<T, N> {
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        for n in 0..N {
            
            if self.lower[n].is_none() && self.upper[n].is_none() {
                continue
            }
            
            let mut row_n = points.row_mut(n);

            row_n.apply(|x| *x = match (self.lower[n], self.upper[n]) {
                (Some(min), Some(max)) => x.clamp(min, max),
                (Some(min), None) => x.max(min),
                (None, Some(max)) => x.min(max),
                (None, None) => unreachable!("What in the dog garn"),
            });
        }
    }
}

pub type DynConstraint<'a, F, const N: usize, const H: usize> =  Constraint<F, &'a dyn Project<F, N, H>, N, H>;

pub struct Constraint<T, P: Project<T, N, H>, const N: usize, const H: usize> {
    pub max_prim_residual: T,
    pub max_dual_residual: T,
    slac: SMatrix<T, N, H>,
    dual: SMatrix<T, N, H>,
    projector: P
}

impl <T: RealField + Copy, const N: usize, const H: usize, P: Project<T, N, H>> Constraint<T, P, N, H> {
    pub fn new(projector: P) -> Self {
        Self {
            max_prim_residual: T::zero(),
            max_dual_residual: T::zero(),
            slac: SMatrix::zeros(),
            dual: SMatrix::zeros(),
            projector,
        }
    }

    /// Shifts slack and dual variables forward by 1 time step.
    /// Used at the beginning of a solve to correctlt hot-start the values.
    pub fn time_shift_variables(&mut self) {

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

        left_shift_matrix(&mut self.dual);
        left_shift_matrix(&mut self.slac);
    }

    /// Constrains the set of points, and computes the maximum primal and dual residuals
    pub fn constrain(&mut self, points: &SMatrix<T, N, H>) {
        let old_slac = self.slac.clone();

        self.slac = points + self.dual;
        self.projector.project(&mut self.slac);
        let prim_residual_matrix = points - self.slac;
        self.dual += prim_residual_matrix;

        // Use infinity norm for simplicity and strictness
        self.max_prim_residual = prim_residual_matrix.abs().sum();
        self.max_dual_residual = (old_slac - self.slac).abs().sum();
    }

    pub fn add_cost(&self, cost: &mut SMatrix<T, N, H>) {
        *cost += self.dual - self.slac;
    }
}