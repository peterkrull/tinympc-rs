use nalgebra::{convert, RealField, SMatrix, SMatrixView, SMatrixViewMut, SVector, Unit};

/// Can project a series of points into their feasible region.
pub trait Project<T, const N: usize, const H: usize> {
    /// Applies the projection to a series of points, modifying them in place
    fn project(&self, points: SMatrixViewMut<T, N, H>);
}

impl<T, const N: usize, const H: usize> Project<T, N, H> for &dyn Project<T, N, H> {
    fn project(&self, points: SMatrixViewMut<T, N, H>) {
        (**self).project(points);
    }
}

impl<P: Project<T, N, H>, T, const N: usize, const H: usize> Project<T, N, H> for &P {
    fn project(&self, points: SMatrixViewMut<T, N, H>) {
        (**self).project(points);
    }
}

/// Extension trait for types implementing [`Project`] to convert it directly
/// into a constraint with associated dual and slack variables.
pub trait ProjectExt<T: RealField + Copy, const N: usize, const H: usize>:
    Project<T, N, H> + Sized
{
    fn constraint(&self) -> Constraint<T, &Self, N, H> {
        Constraint::new(self)
    }

    fn dyn_constraint(&self) -> DynConstraint<'_, T, N, H> {
        Constraint::new(self as &dyn Project<T, N, H>)
    }
}

impl<S: Project<T, N, H>, T: RealField + Copy, const N: usize, const H: usize> ProjectExt<T, N, H>
    for S
{
}

/// A box constraint that is constant throughout the horizon.
pub struct Box<T, const N: usize> {
    pub lower: SVector<Option<T>, N>,
    pub upper: SVector<Option<T>, N>,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for Box<T, N> {
    fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
        for h in 0..H {
            let mut column = points.column_mut(h);
            for n in 0..N {
                column[n] = match (self.lower[n], self.upper[n]) {
                    (Some(min), Some(max)) => column[n].clamp(min, max),
                    (Some(min), None) => column[n].max(min),
                    (None, Some(max)) => column[n].min(max),
                    (None, None) => continue,
                };
            }
        }
    }
}

/// A spherical constraint that is constant throughout the horizon
#[derive(Debug, Copy, Clone)]
pub struct Sphere<T, const N: usize> {
    pub center: SVector<Option<T>, N>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for Sphere<T, N> {
    fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
        for h in 0..H {
            let mut point = points.column_mut(h);

            // Compute squared distance only for dimensions with defined centers
            let mut squared_dist = T::zero();
            let mut has_constraint = false;

            for n in 0..N {
                if let Some(center) = self.center[n] {
                    has_constraint = true;
                    let diff = point[n] - center;
                    squared_dist += diff * diff;
                }
            }

            // If no dimensions are constrained or point is within radius, skip
            if !has_constraint || squared_dist <= self.radius * self.radius {
                continue;
            }

            // Calculate scaling factor for projection
            let dist = squared_dist.sqrt();
            let scale = self.radius / dist;

            // Apply scaling only to dimensions with defined centers
            for n in 0..N {
                if let Some(center) = self.center[n] {
                    let diff = point[n] - center;
                    point[n] = center + diff * scale;
                }
            }
        }
    }
}

/// A half-space constraint that is constant throughout the horizon
#[derive(Debug, Copy, Clone)]
pub struct HalfSpace<T, const N: usize> {
    pub center: SVector<T, N>,
    pub normal: Unit<SVector<T, N>>,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for HalfSpace<T, N> {
    fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
        for h in 0..H {
            let mut point = points.column_mut(h);

            let centered = &point - &self.center;
            let dot = centered.dot(&self.normal);

            if dot < T::zero() {
                point -= self.normal.scale(dot)
            }
        }
    }
}

/// Type alias for a [`Constraint`] that dynamically dispatches its projection function
pub type DynConstraint<'a, F, const N: usize, const H: usize> =
    Constraint<F, &'a dyn Project<F, N, H>, N, H>;

/// A [`Constraint`] consists of a projection function and a set of associated slack and dual variables.
pub struct Constraint<T, P: Project<T, N, H>, const N: usize, const H: usize> {
    pub max_prim_residual: T,
    pub max_dual_residual: T,
    slac: SMatrix<T, N, H>,
    dual: SMatrix<T, N, H>,
    projector: P,
}

impl<T: RealField + Copy, const N: usize, const H: usize, P: Project<T, N, H>>
    Constraint<T, P, N, H>
{
    /// Construct a new [`Constraint`] from the provided [`Project`] type.
    pub fn new(projector: P) -> Self {
        Self {
            max_prim_residual: convert(1e9),
            max_dual_residual: convert(1e9),
            slac: SMatrix::zeros(),
            dual: SMatrix::zeros(),
            projector,
        }
    }

    /// Reset all internal state of this constraint
    pub fn reset(&mut self) {
        self.max_prim_residual = convert(1e9);
        self.max_dual_residual = convert(1e9);
        self.slac = SMatrix::zeros();
        self.dual = SMatrix::zeros();
    }

    /// Constrains the set of points, and if `update_res == true`, computes the maximum primal and dual residuals
    #[inline(never)]
    pub fn constrain(
        &mut self,
        update_res: bool,
        points: SMatrixView<T, N, H>,
        mut scratch: SMatrixViewMut<T, N, H>,
    ) {
        if update_res {
            // Save old slac variables for computing dual residual
            scratch.copy_from(&self.slac);
        }

        // Slack update: slac' = point + dual;
        points.add_to(&self.dual, &mut self.slac);

        self.projector.project(self.slac.as_view_mut());

        if update_res {
            // Compute maximum absolute dual residual
            scratch -= &self.slac;
            scratch.apply(|t| *t = t.abs());
            self.max_dual_residual = scratch.max();
        }

        // Compute primal residual
        points.sub_to(&self.slac, &mut scratch);

        // Update dual parameters
        self.dual += &scratch;

        if update_res {
            // Find maximum absolute primal residual
            scratch.apply(|t| *t = t.abs());
            self.max_prim_residual = scratch.max();
        }
    }

    /// Add the cost associated with this constraints violation to a cost sum
    pub(crate) fn add_cost(
        &mut self,
        mut cost: SMatrixViewMut<T, N, H>,
        mut scratch: SMatrixViewMut<T, N, H>,
    ) {
        self.dual.sub_to(&self.slac, &mut scratch);
        cost += &scratch;
    }

    /// Re-scale the dual variables for when the value of rho has changed
    pub(crate) fn rescale_dual(&mut self, scalar: T) {
        self.dual.scale_mut(scalar);
    }

    /// Shifts dual variables forward by 1 time step.
    /// Used at the beginning of a solve to hot-start the values.
    pub(crate) fn time_shift_variables(&mut self) {
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
    }
}
