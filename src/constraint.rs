use nalgebra::{RealField, SMatrix, SMatrixView, SMatrixViewMut, convert};

use crate::project::Project;

/// A [`Constraint`] consists of a projection function and a set of associated slack and dual variables.
pub struct Constraint<T, P: Project<T, N, H>, const N: usize, const H: usize> {
    pub max_prim_residual: T,
    pub max_dual_residual: T,
    pub(crate) slac: SMatrix<T, N, H>,
    pub(crate) dual: SMatrix<T, N, H>,
    pub(crate) projector: P,
}

/// Type alias for a [`Constraint`] that dynamically dispatches its projection function
pub type DynConstraint<'a, F, const N: usize, const H: usize> =
    Constraint<F, &'a dyn Project<F, N, H>, N, H>;

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

    /// Get a mutable reference to the projector.
    /// Allows for modifying or completely swapping the projector at runtime.
    pub fn projector_mut(&mut self) -> &mut P {
        &mut self.projector
    }

    /// Reset all internal state of this constraint
    pub fn reset(&mut self) {
        self.max_prim_residual = convert(1e9);
        self.max_dual_residual = convert(1e9);
        self.slac = SMatrix::zeros();
        self.dual = SMatrix::zeros();
    }

    /// Constrains the set of points, and if `compute_residuals == true`, computes the maximum primal and dual residuals
    #[inline(always)]
    pub fn constrain(
        &mut self,
        compute_residuals: bool,
        points: SMatrixView<T, N, H>,
        reference: Option<SMatrixView<T, N, H>>,
        scratch: SMatrixViewMut<T, N, H>,
    ) {
        match compute_residuals {
            true => self.constrain_calc_residuals(points, reference, scratch),
            false => self.constrain_only(points, reference),
        }
    }

    /// Constrains the set of points, and computes the maximum primal and dual residuals
    #[inline(always)]
    fn constrain_calc_residuals(
        &mut self,
        points: SMatrixView<T, N, H>,
        reference: Option<SMatrixView<T, N, H>>,
        mut scratch: SMatrixViewMut<T, N, H>,
    ) {
        // Initialize with old slac variables for computing dual residual
        scratch.copy_from(&self.slac);

        // Offset the slack variables by the reference before projecting
        points.add_to(&self.dual, &mut self.slac);
        if let Some(reference) = reference {
            self.slac += &reference;
            self.projector.project(self.slac.as_view_mut());
            self.slac -= &reference;
        } else {
            self.projector.project(self.slac.as_view_mut());
        }

        // Compute dual residual
        scratch -= &self.slac;
        scratch.apply(|t| *t = t.abs());
        self.max_dual_residual = scratch.max();

        // Compute primal residual
        points.sub_to(&self.slac, &mut scratch);

        // Update dual parameters
        self.dual += &scratch;

        // Compute primal residual
        scratch.apply(|t| *t = t.abs());
        self.max_prim_residual = scratch.max();
    }

    /// Constrains the set of points
    #[inline(always)]
    fn constrain_only(
        &mut self,
        points: SMatrixView<T, N, H>,
        reference: Option<SMatrixView<T, N, H>>,
    ) {
        // Offset the slack variables by the reference before projecting
        points.add_to(&self.dual, &mut self.slac);
        if let Some(reference) = reference {
            self.slac += &reference;
            self.projector.project(self.slac.as_view_mut());
            self.slac -= &reference;
        } else {
            self.projector.project(self.slac.as_view_mut());
        }

        // Update dual parameters
        self.dual += &points;
        self.dual -= &self.slac;
    }

    /// Add the cost associated with this constraints violation to a cost sum
    #[inline(always)]
    pub(crate) fn add_cost<'a>(&self, cost: impl Into<SMatrixViewMut<'a, T, N, H>>) {
        let mut cost = cost.into();
        cost += &self.dual;
        cost -= &self.slac;
    }

    /// Add the cost associated with this constraints violation to a cost sum
    #[inline(always)]
    pub(crate) fn set_cost<'a>(&mut self, cost: impl Into<SMatrixViewMut<'a, T, N, H>>) {
        self.dual.sub_to(&self.slac, &mut cost.into());
    }

    /// Re-scale the dual variables for when the value of rho has changed
    #[inline(always)]
    pub(crate) fn rescale_dual(&mut self, scalar: T) {
        self.dual.scale_mut(scalar);
    }
}
