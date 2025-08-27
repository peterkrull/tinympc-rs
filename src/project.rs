use nalgebra::{RealField, SMatrixViewMut, SVector};

use crate::constraint::{Constraint, DynConstraint};

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

impl<T, const N: usize, const H: usize> Project<T, N, H> for () {
    fn project(&self, mut _points: SMatrixViewMut<T, N, H>) {}
}

impl<P: Project<T, N, H>, T, const N: usize, const H: usize, const NUM: usize> Project<T, N, H>
    for [P; NUM]
{
    fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
        for projector in self {
            projector.project(points.as_view_mut());
        }
    }
}

macro_rules! derive_tuple_project {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: Project<T, N, H>),+, T, const N: usize, const H: usize> Project<T, N, H>
            for ( $($project,)+ )
        {
            fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
                $(
                    self.$number.project(points.as_view_mut());
                )+
            }
        }
    };
}

derive_tuple_project! {P0: 0}
derive_tuple_project! {P0: 0, P1: 1}
derive_tuple_project! {P0: 0, P1: 1, P2: 2}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7, P8: 8}
derive_tuple_project! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7, P8: 8, P9: 9}

/// Extension trait for types implementing [`Project`] to convert it directly
/// into a constraint with associated dual and slack variables.
pub trait ProjectExt<T: RealField + Copy, const N: usize, const H: usize>:
    Project<T, N, H> + Sized
{
    fn dynamic(&self) -> &dyn Project<T, N, H> {
        self
    }

    fn constraint(&self) -> Constraint<T, &Self, N, H> {
        Constraint::new(self)
    }

    fn dyn_constraint(&self) -> DynConstraint<'_, T, N, H> {
        Constraint::new(self.dynamic())
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
        let lower = self.lower.map(|x| x.unwrap_or(T::min_value().unwrap()));
        let upper = self.upper.map(|x| x.unwrap_or(T::max_value().unwrap()));

        for h in 0..H {
            let mut column = points.column_mut(h);
            for n in 0..N {
                column[n] = column[n].clamp(lower[n], upper[n]);
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
        // Special case, just snap the points to the center coordinate
        if self.radius.is_zero() {
            for h in 0..H {
                let mut point = points.column_mut(h);
                for n in 0..N {
                    if let Some(center) = self.center[n] {
                        point[n] = center
                    }
                }
            }
            return;
        }

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
pub struct Affine<T, const N: usize> {
    pub normal: SVector<T, N>,
    pub distance: T,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for Affine<T, N> {
    fn project(&self, mut points: SMatrixViewMut<T, N, H>) {
        if self.normal.norm_squared().is_zero() {
            return;
        }
        let normal = self.normal.normalize();

        for h in 0..H {
            let mut point = points.column_mut(h);
            let dot = point.dot(&normal);

            if dot > self.distance {
                // Project onto the boundary: move point towards the plane
                let correction = normal.scale(dot - self.distance);
                point -= correction;
            }
        }
    }
}
