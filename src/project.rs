use std::marker::PhantomData;

use nalgebra::{ComplexField, RealField, SMatrix, SVector, SVectorViewMut, convert};

use crate::constraint::{Constraint, DynConstraint};

/// Can project a series of points into their feasible region.
pub trait Project<T, const D: usize> {
    /// Applies the projection to a series of points, modifying them in place
    fn project(&self, point: SVectorViewMut<T, D>);
}

impl<T, const D: usize> Project<T, D> for &dyn Project<T, D> {
    fn project(&self, point: SVectorViewMut<T, D>) {
        (**self).project(point);
    }
}

impl<P: Project<T, D>, T, const D: usize> Project<T, D> for &P {
    fn project(&self, point: SVectorViewMut<T, D>) {
        (**self).project(point);
    }
}

impl<T, const D: usize> Project<T, D> for () {
    fn project(&self, mut _points: SVectorViewMut<T, D>) {}
}

impl<P: Project<T, D>, T, const D: usize, const NUM: usize> Project<T, D>
    for [P; NUM]
{
    fn project(&self, mut point: SVectorViewMut<T, D>) {
        for projector in self {
            projector.project(point.as_view_mut());
        }
    }
}

macro_rules! derive_tuple_project {
    ($($project:ident: $number:tt),+) => {
        impl<$($project: Project<T, D>),+, T, const D: usize> Project<T, D>
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

/// Extension trait for types implementing [`Project`] to convert it directly
/// into a constraint with associated dual and slack variables.
pub trait ProjectExt<T: RealField + Copy, const D: usize>: Sized + Project<T, D> {

    fn dynamic(&self) -> &dyn Project<T, D>
    where
        Self: Project<T, D>,
    {
        self
    }

    fn constraint<const H: usize>(&self) -> Constraint<T, &Self, D, H>
    where
        Self: Project<T, D>,
    {
        Constraint::new(self)
    }

    fn constraint_owned<const H: usize>(self) -> Constraint<T, Self, D, H>
    where
        Self: Project<T, D>,
    {
        Constraint::new(self)
    }

    fn dyn_constraint<const H: usize>(&self) -> DynConstraint<'_, T, D, H>
    where
        Self: Project<T, D>,
    {
        Constraint::new(self.dynamic())
    }
}

impl<S: Sized + Project<T, D>, T: RealField + Copy, const D: usize> ProjectExt<T, D> for S {}

/// A box constraint
pub struct Box<T, const N: usize> {
    pub lower: SVector<T, N>,
    pub upper: SVector<T, N>,
}

impl<T: RealField + Copy, const N: usize> Project<T, N> for Box<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        for n in 0..N {
            point[n] = point[n].clamp(self.lower[n], self.upper[n]);
        }
    }
}

/// A spherical constraint
#[derive(Debug, Copy, Clone)]
pub struct Sphere<T, const D: usize> {
    pub center: SVector<T, D>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize> Project<T, N> for Sphere<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        if self.radius.is_zero() {
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

/// An anti-spherical constraint
#[derive(Debug, Copy, Clone)]
pub struct AntiSphere<T, const N: usize> {
    pub center: SVector<T, N>,
    pub radius: T,
}

impl<T: RealField + Copy, const N: usize> Project<T, N> for AntiSphere<T, N> {
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

/// A half-space constraint
#[derive(Debug, Copy, Clone)]
pub struct Affine<T, const N: usize> {
    pub normal: SVector<T, N>,
    pub distance: T,
}

impl<T: RealField + Copy, const N: usize> Project<T, N> for Affine<T, N> {
    #[inline(always)]
    fn project(&self, mut point: SVectorViewMut<T, N>) {
        let dot = point.dot(&self.normal);

        if dot > self.distance {
            point -= self.normal.scale(dot - self.distance);
        }
    }
}

/// A circular cone constraint
#[derive(Debug, Clone)]
pub struct CircularCone<T: RealField + Copy, const D: usize> {
    vertex: SVector<T, D>,
    axis: SVector<T, D>,
    mu: T,
}

impl<T: RealField + Copy, const D: usize> CircularCone<T, D> {
    pub fn new() -> CircularCone<T, D> {
        CircularCone {
            vertex: SVector::zeros(),
            axis: SVector::identity(),
            mu: nalgebra::convert(1.0),
        }
    }

    pub fn axis(mut self, axis: impl Into<SVector<T, D>>) -> Self {
        self.axis = axis.into().normalize();
        self
    }

    pub fn vertex(mut self, vertex: impl Into<SVector<T, D>>) -> Self {
        self.vertex = vertex.into();
        self
    }

    pub fn mu(mut self, mu: T) -> Self {
        self.mu = mu.max(T::default_epsilon());
        self
    }
}

impl<T: RealField + Copy, const D: usize> Project<T, D>
    for CircularCone<T, D> {
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


/// A circular cone constraint, constant throughout the horizon.
#[derive(Debug, Clone)]
pub struct SubSpace<P, const D: usize, const N: usize> {
    indices: [usize; D],
    projector: P,
    _p: PhantomData<[(); N]>
}

impl <P, const D: usize, const N: usize> SubSpace<P, D, N> {
    pub fn new(indices: [usize; D], projector: P) -> Self {
        assert!(indices.iter().all(|e|e < &N));
        Self {
            indices,
            projector,
            _p: PhantomData,
        }
    }
}

impl<P: Project<T, D>, T: ComplexField + Copy, const D: usize, const N: usize> Project<T, N>
    for SubSpace<P, D, N> {
        fn project(&self, mut point: SVectorViewMut<T, N>) {
        assert!(self.indices.iter().all(|e|e < &N));

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

/// A circular cone constraint, constant throughout the horizon.
#[derive(Debug, Clone)]
pub struct Cone<T: RealField + Copy, const D: usize> {
    vertex: SVector<T, D>,
    axis: SVector<T, D>,
    mu: T,
}

impl<T: RealField + Copy, const D: usize> Cone<T, D> {
    pub fn new() -> Cone<T, D> {
        Cone {
            vertex: SVector::zeros(),
            axis: SVector::identity(),
            mu: nalgebra::convert(1.0),
        }
    }

    pub fn axis(mut self, axis: impl Into<SVector<T, D>>) -> Self {
        self.axis = axis.into().normalize();
        self
    }

    pub fn vertex(mut self, vertex: impl Into<SVector<T, D>>) -> Self {
        self.vertex = vertex.into();
        self
    }

    pub fn mu(mut self, mu: T) -> Self {
        self.mu = mu.max(T::default_epsilon());
        self
    }

     fn project(&self, point: &mut SVector<T, D>) {
        profiling::scope!("projector: Cone");

        // Translate by the vertex to get vector v
        let v = *point - self.vertex;

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
            *point = self.vertex;
        }

        // Outside both, project onto boundary
        else {
            let alpha = (self.mu * a + s_n) / (T::one() + self.mu * self.mu);
            *point = (self.axis + s_v * self.mu / a) * alpha + self.vertex;
        }
    }
}
