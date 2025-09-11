use nalgebra::{RealField, SMatrix, convert};

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
    #[profiling::function]
    pub fn constrain(
        &mut self,
        compute_residuals: bool,
        points: &SMatrix<T, N, H>,
        reference: Option<&SMatrix<T, N, H>>,
        scratch: &mut SMatrix<T, N, H>,
    ) {
        match compute_residuals {
            true => self.constrain_calc_residuals(points, reference, scratch),
            false => self.constrain_only(points, reference),
        }
    }

    /// Constrains the set of points, and computes the maximum primal and dual residuals
    #[inline(always)]
    #[profiling::function]
    fn constrain_calc_residuals(
        &mut self,
        points: &SMatrix<T, N, H>,
        reference: Option<&SMatrix<T, N, H>>,
        mut scratch: &mut SMatrix<T, N, H>,
    ) {
        // Initialize with old slac variables for computing dual residual
        scratch.copy_from(&self.slac);

        // Offset the slack variables by the reference before projecting
        points.add_to(&self.dual, &mut self.slac);
        if let Some(reference) = reference {
            profiling::scope!("reference offset");
            self.slac += reference;
            self.projector.project(&mut self.slac);
            self.slac -= reference;
        } else {
            self.projector.project(&mut self.slac);
        }

        // Compute dual residual
        *scratch -= self.slac;
        self.max_dual_residual = crate::util::frobenius_norm(&scratch);

        // Compute primal residual
        points.sub_to(&self.slac, &mut scratch);

        // Update dual parameters
        self.dual += *scratch;

        // Compute primal residual
        self.max_prim_residual = crate::util::frobenius_norm(&scratch);
    }

    /// Constrains the set of points
    #[inline(always)]
    #[profiling::function]
    fn constrain_only(&mut self, points: &SMatrix<T, N, H>, reference: Option<&SMatrix<T, N, H>>) {
        // Offset the slack variables by the reference before projecting
        points.add_to(&self.dual, &mut self.slac);
        if let Some(reference) = reference {
            profiling::scope!("reference offset");
            self.slac += reference;
            self.projector.project(&mut self.slac);
            self.slac -= reference;
        } else {
            self.projector.project(&mut self.slac);
        }

        // Update dual parameters
        self.dual += points;
        self.dual -= self.slac;
    }

    /// Add the cost associated with this constraints violation to a cost sum
    #[inline(always)]
    #[profiling::function]
    pub(crate) fn add_cost(&self, cost: &mut SMatrix<T, N, H>) {
        *cost += self.dual;
        *cost -= self.slac;
    }

    /// Add the cost associated with this constraints violation to a cost sum
    #[inline(always)]
    #[profiling::function]
    pub(crate) fn set_cost(&mut self, cost: &mut SMatrix<T, N, H>) {
        self.dual.sub_to(&self.slac, cost);
    }

    /// Re-scale the dual variables for when the value of rho has changed
    #[inline(always)]
    #[profiling::function]
    pub(crate) fn rescale_dual(&mut self, scalar: T) {
        self.dual.scale_mut(scalar);
    }
}
