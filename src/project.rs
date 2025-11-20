use nalgebra::{RealField, SMatrix, SVector};

use crate::constraint::{Constraint, DynConstraint};

/// Can project a series of points into their feasible region.
pub trait Project<T, const N: usize, const H: usize> {
    /// Applies the projection to a series of points, modifying them in place
    fn project(&self, points: &mut SMatrix<T, N, H>);
}

impl<T, const N: usize, const H: usize> Project<T, N, H> for &dyn Project<T, N, H> {
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        (**self).project(points);
    }
}

impl<P: Project<T, N, H>, T, const N: usize, const H: usize> Project<T, N, H> for &P {
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        (**self).project(points);
    }
}

impl<T, const N: usize, const H: usize> Project<T, N, H> for () {
    fn project(&self, mut _points: &mut SMatrix<T, N, H>) {}
}

impl<P: Project<T, N, H>, T, const N: usize, const H: usize, const NUM: usize> Project<T, N, H>
    for [P; NUM]
{
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        for projector in self {
            projector.project(points);
        }
    }
}

macro_rules! derive_tuple_project {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: Project<T, N, H>),+, T, const N: usize, const H: usize> Project<T, N, H>
            for ( $($project,)+ )
        {
            fn project(&self, points: &mut SMatrix<T, N, H>) {
                $(
                    self.$number.project(points);
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
pub trait ProjectExt<T: RealField + Copy, const N: usize, const H: usize>: Sized {
    fn dynamic(&self) -> &dyn Project<T, N, H>
    where
        Self: Project<T, N, H>,
    {
        self
    }

    fn constraint(&self) -> Constraint<T, &Self, N, H>
    where
        Self: Project<T, N, H>,
    {
        Constraint::new(self)
    }

    fn dyn_constraint(&self) -> DynConstraint<'_, T, N, H>
    where
        Self: Project<T, N, H>,
    {
        Constraint::new(self.dynamic())
    }
}

impl<S: Sized, T: RealField + Copy, const N: usize, const H: usize> ProjectExt<T, N, H> for S {}

/// A box constraint that is constant throughout the horizon.
pub struct Box<T, const N: usize> {
    pub lower: SVector<Option<T>, N>,
    pub upper: SVector<Option<T>, N>,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for Box<T, N> {
    #[inline(always)]
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        profiling::scope!("projector: Box");
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
    #[inline(always)]
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        profiling::scope!("projector: Sphere");
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

/// An anti-spherical constraint that is constant throughout the horizon
#[derive(Debug, Copy, Clone)]
pub struct AntiSphere<T, const N: usize> {
    pub center: SVector<Option<T>, N>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize, const H: usize> Project<T, N, H> for AntiSphere<T, N> {
    #[inline(always)]
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        // If radius is zero, the feasible region is everything.
        if self.radius.is_zero() {
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

            // If no dimensions are constrained or point is outside/on radius, it's valid
            if !has_constraint || squared_dist >= self.radius * self.radius {
                continue;
            }

            let dist = squared_dist.sqrt();

            // The projection direction is undefined.
            if dist.is_zero() {
                for n in 0..N {
                    if let Some(center) = self.center[n] {
                        point[n] = center + self.radius;
                        // Break after moving along the first available axis
                        break;
                    }
                }
                continue;
            }

            // Calculate scaling factor for projection
            let scale = self.radius / dist;

            // Apply scaling to push the point to the surface
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
    #[inline(always)]
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        profiling::scope!("projector: Affine");
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

/// A second-order cone constraint, constant throughout the horizon.
#[derive(Debug, Clone)]
pub struct SecondOrderCone<T: RealField + Copy, const D: usize> {
    indices: [usize; D],
    tip: SVector<T, D>,
    axis: SVector<T, D>,
    mu: T,
}

impl<T: RealField + Copy, const D: usize> SecondOrderCone<T, D> {
    pub fn new<const N: usize>(indices: [usize; D]) -> SecondOrderCone<T, D> {
        SecondOrderCone {
            indices,
            tip: SVector::zeros(),
            axis: SVector::identity(),
            mu: nalgebra::convert(1.0),
        }
    }

    pub fn along(mut self, axis: SVector<T, D>) -> Self {
        self.axis = axis.normalize();
        self
    }

    pub fn origin(mut self, tip: SVector<T, D>) -> Self {
        self.tip = tip;
        self
    }

    pub fn mu(mut self, mu: T) -> Self {
        self.mu = mu.max(T::default_epsilon());
        self
    }
}

impl<T: RealField + Copy, const N: usize, const H: usize, const D: usize> Project<T, N, H>
    for SecondOrderCone<T, D> {
    #[inline(always)]
    fn project(&self, points: &mut SMatrix<T, N, H>) {
        profiling::scope!("projector: Cone");

        // Cones with mu < 0 are invalid (project to a line)
        // Cones with mu = 0 are just a ray.
        if self.mu <= T::zero() {
            return;
        }

        for h in 0..H {
            let mut point_h = points.column_mut(h);

            // Extract the sub-vector from the full state
            let mut sub_point: SVector<T, D> = SVector::zeros();
            for i in 0..D {
                if self.indices[i] >= N { continue; }
                sub_point[i] = point_h[self.indices[i]];
            }

            // Translate by the tip to get vector v
            let v = sub_point - self.tip;

            // Decompose v into parallel and orthogonal components
            let s_n = v.dot(&self.axis);
            let s_v = v - self.axis.scale(s_n);

            // The radial distance
            let a = s_v.norm();

            // Inside feasible region, do nothing
            if a <= self.mu * s_n {
                continue;
            }

            // Inside polar cone, project to tip
            else if (s_n < T::zero() && (a * self.mu <= -s_n)) || a.is_zero() {
                sub_point = self.tip;
            }

            // Outside both, project onto boundary
            else {

                let mu_sq = self.mu * self.mu;
                let denom = T::one() + mu_sq;

                // Correct Euclidean projection formula
                let c = self.mu * a + s_n;
                let a_proj = (self.mu * c) / denom;
                let s_n_proj = c / denom;

                // Reconstruct the projected vector
                let s_v_proj = s_v.scale(a_proj / a);
                let v_proj = s_v_proj + self.axis.scale(s_n_proj);

                // Translate back from the tip
                sub_point = v_proj + self.tip;
            }

            // 6. Write the projected sub-vector back into the full state
            for i in 0..D {
                if self.indices[i] >= N { continue; }
                point_h[self.indices[i]] = sub_point[i];
            }
        }
    }
}
