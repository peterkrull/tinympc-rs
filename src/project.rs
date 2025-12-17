use core::ops::Range;

use nalgebra::{RealField, SMatrix, SVector, SVectorViewMut, convert};

use crate::constraint::{Constraint, DynConstraint};

/// Can project a single point into its feasible region.
pub trait ProjectSingle<T, const D: usize> {
    /// Apply the projection to a single point.
    fn project_single(&self, point: SVectorViewMut<T, D>);
}

impl<T, const D: usize> ProjectSingle<T, D> for &dyn ProjectSingle<T, D> {
    fn project_single(&self, point: SVectorViewMut<T, D>) {
        (**self).project_single(point);
    }
}

impl<P: ProjectSingle<T, D>, T, const D: usize> ProjectSingle<T, D> for &P {
    fn project_single(&self, point: SVectorViewMut<T, D>) {
        (**self).project_single(point);
    }
}

impl<T, const D: usize> ProjectSingle<T, D> for () {
    fn project_single(&self, mut _point: SVectorViewMut<T, D>) {}
}

macro_rules! derive_tuple_project {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: ProjectSingle<T, D>),+, T, const D: usize> ProjectSingle<T, D>
            for ( $($project,)+ )
        {
            fn project_single(&self, mut point: SVectorViewMut<T, D>) {
                $(
                    self.$number.project_single(point.as_view_mut());
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

pub trait ProjectSingleExt<T: RealField + Copy, const D: usize>:
    ProjectSingle<T, D> + Sized
{
    fn time_fixed(self) -> time::Fixed<Self> {
        time::Fixed::new(self)
    }

    fn time_ranged(self, range: Range<usize>) -> time::Ranged<Self> {
        time::Ranged::new(self, range)
    }

    fn dim_lift<const N: usize>(self, indices: [usize; D]) -> dim::Lift<Self, D, N> {
        dim::Lift::new(indices, self)
    }
}

impl<P: ProjectSingle<T, D>, T: RealField + Copy, const D: usize> ProjectSingleExt<T, D> for P {}

/// Can project a multiple points into their feasible region.
pub trait ProjectMulti<T, const D: usize, const H: usize> {
    fn project_multi(&self, points: &mut SMatrix<T, D, H>);
}

impl<P: ProjectMulti<T, D, H>, T, const D: usize, const H: usize> ProjectMulti<T, D, H> for &P {
    fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
        (**self).project_multi(points);
    }
}

impl<T, const D: usize, const H: usize> ProjectMulti<T, D, H> for &dyn ProjectMulti<T, D, H> {
    fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
        (**self).project_multi(points);
    }
}

impl<T, const D: usize, const H: usize> ProjectMulti<T, D, H> for () {
    fn project_multi(&self, _points: &mut SMatrix<T, D, H>) {}
}

macro_rules! derive_tuple_project_multi {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: ProjectMulti<T, D, H>),+, T, const D: usize, const H: usize> ProjectMulti<T, D, H>
            for ( $($project,)+ )
        {
            fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
                $(
                    self.$number.project_multi(points);
                )+
            }
        }
    };
}

derive_tuple_project_multi! {P0: 0}
derive_tuple_project_multi! {P0: 0, P1: 1}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7, P8: 8}
derive_tuple_project_multi! {P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5, P6: 6, P7: 7, P8: 8, P9: 9}

pub trait ProjectMultiExt<T: RealField + Copy, const D: usize, const H: usize>:
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

impl<P: ProjectMulti<T, D, H>, T: RealField + Copy, const D: usize, const H: usize>
    ProjectMultiExt<T, D, H> for P
{
}

/// A box projection
#[derive(Debug, Copy, Clone)]
pub struct Box<T, const N: usize> {
    pub lower: SVector<T, N>,
    pub upper: SVector<T, N>,
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for Box<T, N> {
    fn project_single(&self, mut point: SVectorViewMut<T, N>) {
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
    fn project_single(&self, mut point: SVectorViewMut<T, N>) {
        if self.radius <= T::zero() {
            point.copy_from(&self.center);
        } else {
            let diff = &point - self.center;
            let dist = diff.norm();

            if dist > self.radius {
                let scale = self.radius / dist;
                point.copy_from(&self.center);
                point.axpy(scale, &diff, T::one());
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
    fn project_single(&self, mut point: SVectorViewMut<T, N>) {
        if self.radius > T::zero() {
            let diff = &point - self.center;
            let dist = diff.norm();

            if dist < self.radius {
                let scale = self.radius / dist;
                point.copy_from(&(self.center + diff * scale));
            }
        }
    }
}

/// A half-space projection
#[derive(Debug, Copy, Clone)]
pub struct Affine<T, const N: usize> {
    normal: SVector<T, N>,
    distance: T,
}

impl<T: RealField + Copy, const D: usize> Default for Affine<T, D> {
    fn default() -> Affine<T, D> {
        Affine {
            normal: SVector::identity(),
            distance: T::zero(),
        }
    }
}

impl<T: RealField + Copy, const D: usize> Affine<T, D> {
    /// Create a new [`Affine`] projector with default values.
    #[must_use]
    pub fn new() -> Affine<T, D> {
        Affine::default()
    }

    /// Set the axis along the center of the cone.
    ///
    /// # Panics
    ///
    /// If the provided vector has no magnitude.
    #[must_use]
    pub fn normal(mut self, normal: impl Into<SVector<T, D>>) -> Self {
        self.normal = normal.into();
        assert!(self.normal.norm() > convert(1e-9));
        self.normal = self.normal.normalize();
        self
    }

    /// Set the affine projectors offset distance
    #[must_use]
    pub fn distance(mut self, distance: T) -> Self {
        self.distance = distance;
        self
    }
}

impl<T: RealField + Copy, const N: usize> ProjectSingle<T, N> for Affine<T, N> {
    fn project_single(&self, mut point: SVectorViewMut<T, N>) {
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

impl<T: RealField + Copy, const D: usize> Default for CircularCone<T, D> {
    fn default() -> Self {
        CircularCone {
            vertex: SVector::zeros(),
            axis: SVector::identity(),
            mu: nalgebra::convert(1.0),
        }
    }
}

impl<T: RealField + Copy, const D: usize> CircularCone<T, D> {
    /// Create a new [`CircularCone`] projector with default values.
    #[must_use]
    pub fn new() -> CircularCone<T, D> {
        CircularCone::default()
    }

    /// Set the axis along the center of the cone.
    #[must_use]
    pub fn axis(mut self, axis: impl Into<SVector<T, D>>) -> Self {
        self.axis = axis.into().normalize();
        self
    }

    /// Set the coordinate of the cones vertex / tip.
    #[must_use]
    pub fn vertex(mut self, vertex: impl Into<SVector<T, D>>) -> Self {
        self.vertex = vertex.into();
        self
    }

    /// Set the `µ` value, or the "aperture" of the cone.
    #[must_use]
    pub fn mu(mut self, mu: T) -> Self {
        self.mu = mu.max(T::zero());
        self
    }
}

impl<T: RealField + Copy, const D: usize> ProjectSingle<T, D> for CircularCone<T, D> {
    fn project_single(&self, mut point: SVectorViewMut<T, D>) {
        // Translate by the tip to get vector v
        let v = &point - self.vertex;

        // Decompose v into parallel and orthogonal components
        let s_n = v.dot(&self.axis);
        let s_v = v - self.axis.scale(s_n);

        // The radial distance
        let a = s_v.norm();

        // Inside feasible region, do nothing
        if a <= self.mu * s_n {
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

/// Module for dimension modifiers for projectors
pub mod dim {
    use core::marker::PhantomData;

    use nalgebra::{RealField, SVector, SVectorViewMut};

    use crate::ProjectSingle;

    /// Lift the projection into a higher dimensional space.
    #[derive(Debug, Clone)]
    pub struct Lift<P, const D: usize, const N: usize> {
        indices: [usize; D],
        pub projector: P,
        _p: PhantomData<[(); N]>,
    }

    impl<P, const D: usize, const N: usize> Lift<P, D, N> {
        /// Lift the provided projector into a higher-dimensional space
        ///
        /// # Panics
        ///
        /// If any of the provided indeces exceed the higher dimension of `N`
        pub fn new(indices: [usize; D], projector: P) -> Self {
            assert!(indices.iter().all(|e| e < &N));
            Self {
                indices,
                projector,
                _p: PhantomData,
            }
        }
    }

    impl<P: ProjectSingle<T, D>, T: RealField + Copy, const D: usize, const N: usize>
        ProjectSingle<T, N> for Lift<P, D, N>
    {
        fn project_single(&self, mut point: SVectorViewMut<T, N>) {
            let mut sub_point: SVector<T, D> = SVector::zeros();
            for i in 0..D {
                sub_point[i] = point[self.indices[i]];
            }

            self.projector.project_single(sub_point.as_view_mut());

            for i in 0..D {
                point[self.indices[i]] = sub_point[i];
            }
        }
    }
}

/// Module for time modifiers for projectors
pub mod time {
    use core::ops::Range;

    use nalgebra::{RealField, SMatrix};

    use crate::{ProjectMulti, ProjectSingle};

    /// Apply the projector `P` to be fixed throughout the entire horizon.
    pub struct Fixed<P> {
        pub projector: P,
    }

    impl<P> Fixed<P> {
        pub fn new(projector: P) -> Fixed<P> {
            Fixed { projector }
        }
    }

    impl<P: ProjectSingle<T, D>, T: RealField + Copy, const D: usize, const H: usize>
        ProjectMulti<T, D, H> for Fixed<P>
    {
        fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
            for mut column in points.column_iter_mut() {
                self.projector.project_single(column.as_view_mut());
            }
        }
    }

    /// Apply the projector `P` to be fixed across a range of the horizon.
    pub struct Ranged<P> {
        pub projector: P,
        pub range: Range<usize>,
    }

    impl<P> Ranged<P> {
        pub fn new(projector: P, range: Range<usize>) -> Ranged<P> {
            Ranged { projector, range }
        }
    }

    impl<P: ProjectSingle<T, D>, T: RealField + Copy, const D: usize, const H: usize>
        ProjectMulti<T, D, H> for Ranged<P>
    {
        fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
            for mut column in points
                .column_iter_mut()
                .take(self.range.end)
                .skip(self.range.start)
            {
                self.projector.project_single(column.as_view_mut());
            }
        }
    }

    /// Yields a projector for a given index in the time horizon
    pub struct Func<F> {
        func: F,
    }

    impl<F> Func<F> {
        pub fn new(func: F) -> Func<F> {
            Func { func }
        }
    }

    impl<
        P: ProjectSingle<T, D>,
        T: RealField + Copy,
        F: Fn(usize) -> P,
        const D: usize,
        const H: usize,
    > ProjectMulti<T, D, H> for Func<F>
    {
        fn project_multi(&self, points: &mut SMatrix<T, D, H>) {
            for (index, mut column) in points.column_iter_mut().enumerate() {
                (self.func)(index).project_single(column.as_view_mut());
            }
        }
    }
}
