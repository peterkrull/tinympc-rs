use std::marker::PhantomData;

use nalgebra::{ComplexField, RealField, SMatrix, SVector, SVectorViewMut, convert};

use crate::constraint::{Constraint, DynConstraint};

/// Can project a multiple points into their feasible region.
pub trait ProjectMulti<T: RealField + Copy, const D: usize, const H: usize> {
    fn project_series(&self, points: &mut SMatrix<T, D, H>);
}

impl<P: ProjectMulti<T, D, H>, T: RealField + Copy, const D: usize, const H: usize>
    ProjectMulti<T, D, H> for &P
{
    fn project_series(&self, points: &mut SMatrix<T, D, H>) {
        (**self).project_series(points);
    }
}

impl<T: RealField + Copy, const D: usize, const H: usize> ProjectMulti<T, D, H>
    for &dyn ProjectMulti<T, D, H>
{
    fn project_series(&self, points: &mut SMatrix<T, D, H>) {
        (**self).project_series(points);
    }
}

impl<T: RealField + Copy, const D: usize, const H: usize> ProjectMulti<T, D, H> for () {
    fn project_series(&self, _points: &mut SMatrix<T, D, H>) {}
}

/// Can project a single point into its feasible region.
pub trait ProjectSingle<T, const D: usize> {
    /// Apply the projection to a single point.
    fn project(&self, point: SVectorViewMut<T, D>);
}

impl<T, const D: usize> ProjectSingle<T, D> for &dyn ProjectSingle<T, D> {
    fn project(&self, point: SVectorViewMut<T, D>) {
        (**self).project(point);
    }
}

impl<P: ProjectSingle<T, D>, T, const D: usize> ProjectSingle<T, D> for &P {
    fn project(&self, point: SVectorViewMut<T, D>) {
        (**self).project(point);
    }
}

impl<T, const D: usize> ProjectSingle<T, D> for () {
    fn project(&self, mut _point: SVectorViewMut<T, D>) {}
}

macro_rules! derive_tuple_project {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: ProjectSingle<T, D>),+, T, const D: usize> ProjectSingle<T, D>
            for ( $($project,)+ )
        {
            fn project(&self, mut point: SVectorViewMut<T, D>) {
                $(
                    self.$number.project(point.as_view_mut());
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

/// Apply the projector `P` to be constant throughout the entire horizon.
pub struct Constant<P> {
    project: P,
}

impl<P> Constant<P> {
    pub fn new<T, const D: usize>(project: P) -> Constant<P>
    where
        P: ProjectSingle<T, D>,
        T: RealField + Copy,
    {
        Constant { project }
    }
}

impl<P: ProjectSingle<T, D>, T: RealField + Copy, const D: usize, const H: usize>
    ProjectMulti<T, D, H> for Constant<P>
{
    fn project_series(&self, points: &mut SMatrix<T, D, H>) {
        for mut column in points.column_iter_mut() {
            self.project.project(column.as_view_mut());
        }
    }
}

pub trait ProjectExt<T: RealField + Copy, const D: usize, const H: usize>:
    ProjectMulti<T, D, H> + Sized
{
    fn dynamic(&self) -> &dyn ProjectMulti<T, D, H> {
        self
    }

    fn constraint(&self) -> Constraint<T, &Self, D, H> {
        Constraint::new(self)
    }

    fn dyn_constraint(&self) -> DynConstraint<'_, T, D, H> {
        Constraint::new(self)
    }

    fn constraint_owned(self) -> Constraint<T, Self, D, H> {
        Constraint::new(self)
    }
}

impl <P: ProjectMulti<T, D, H>, T: RealField + Copy, const D: usize, const H: usize> ProjectExt<T, D, H> for P {}

/// A box projection
#[derive(Debug, Copy, Clone)]
pub struct Box<T, const N: usize> {
    pub lower: SVector<T, N>,
    pub upper: SVector<T, N>,
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for Box<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        for n in 0..N {
            point[n] = point[n].clamp(self.lower[n], self.upper[n]);
        }
    }
}

/// A spherical projection
#[derive(Debug, Copy, Clone)]
pub struct Sphere<T, const D: usize> {
    pub center: SVector<T, D>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for Sphere<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        if self.radius <= T::zero() {
            point.copy_from(&self.center);
        } else {
            let diff = &point - &self.center;
            let dist = diff.norm();

            if dist > self.radius {
                let scale = self.radius / dist;
                point.copy_from(&(self.center + diff * scale));
            }
        }
    }
}

/// An anti-spherical projection
#[derive(Debug, Copy, Clone)]
pub struct AntiSphere<T, const N: usize> {
    pub center: SVector<T, N>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for AntiSphere<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        if self.radius.is_zero() {
            return;
        } else {
            let diff = &point - &self.center;
            let dist = diff.norm();

            if dist < self.radius {
                let scale = self.radius / dist.max(convert(1e-9));
                point.copy_from(&(self.center + diff * scale));
            }
        }
    }
}

/// A half-space projection
#[derive(Debug, Copy, Clone)]
pub struct Affine<T, const N: usize> {
    pub normal: SVector<T, N>,
    pub distance: T,
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for Affine<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        let dot = point.dot(&self.normal);

        if dot > self.distance {
            point -= self.normal.scale(dot - self.distance);
        }
    }
}

/// A circular cone projection
#[derive(Debug, Clone)]
pub struct CircularCone<T: RealField + Copy, const D: usize> {
    vertex: SVector<T, D>,
    axis: SVector<T, D>,
    mu: T,
}

impl<T: RealField + Copy, const D: usize> CircularCone<T, D> {
    /// Create a new [`CircularCone`] projector with default values.
    pub fn new() -> CircularCone<T, D> {
        CircularCone {
            vertex: SVector::zeros(),
            axis: SVector::identity(),
            mu: nalgebra::convert(1.0),
        }
    }

    /// Set the axis along the center of the cone.
    pub fn axis(mut self, axis: impl Into<SVector<T, D>>) -> Self {
        self.axis = axis.into().normalize();
        self
    }

    /// Set the coordinate of the cones vertex / tip.
    pub fn vertex(mut self, vertex: impl Into<SVector<T, D>>) -> Self {
        self.vertex = vertex.into();
        self
    }

    /// Set the `µ` value, or the "aperture" of the cone.
    pub fn mu(mut self, mu: T) -> Self {
        self.mu = mu.max(T::zero());
        self
    }
}

impl<T: RealField + Copy, const D: usize> ProjectSingle<T, D> for CircularCone<T, D> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, D>) {
        // Translate by the tip to get vector v
        let v = &point - &self.vertex;

        // Decompose v into parallel and orthogonal components
        let s_n = v.dot(&self.axis);
        let s_v = v - self.axis.scale(s_n);

        // The radial distance
        let a = s_v.norm();

        // Inside feasible region, do nothing
        if a <= self.mu * s_n {
            return;
        }
        // Inside polar cone, project to tip
        else if (a * self.mu <= -s_n) || a.is_zero() {
            point.copy_from(&self.vertex);
        }
        // Outside both, project onto boundary
        else {
            let alpha = (self.mu * a + s_n) / (T::one() + self.mu * self.mu);
            point.copy_from(&((self.axis + s_v * self.mu / a) * alpha + self.vertex));
        }
    }
}

/// Expand the projection into a higher dimensional space.
#[derive(Debug, Clone)]
pub struct Expand<P, const D: usize, const N: usize> {
    indices: [usize; D],
    projector: P,
    _p: PhantomData<[(); N]>,
}

impl<P, const D: usize, const N: usize> Expand<P, D, N> {
    pub fn new(indices: [usize; D], projector: P) -> Self {
        assert!(indices.iter().all(|e| e < &N));
        Self {
            indices,
            projector,
            _p: PhantomData,
        }
    }
}

impl<P: ProjectSingle<T, D>, T: ComplexField + Copy, const D: usize, const N: usize>
    ProjectSingle<T, N> for Expand<P, D, N>
{
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        assert!(self.indices.iter().all(|e| e < &N));

        let mut sub_point: SVector<T, D> = SVector::zeros();
        for i in 0..D {
            sub_point[i] = point[self.indices[i]];
        }

        self.projector.project(sub_point.as_view_mut());

        for i in 0..D {
            point[self.indices[i]] = sub_point[i];
        }
    }
}
