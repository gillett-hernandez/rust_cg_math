// use std::marker::PhantomData;
// use typenum::Unsigned;

use crate::prelude::*;

use std::f32::consts::{FRAC_PI_2, TAU};

// pub struct R<T: Unsigned>(PhantomData<T>);

pub trait SpaceParameterization: Default + Clone + Copy {
    type SimpleSet;
    type Element;
    const SPACE: Self::SimpleSet;
}

pub type SimpleSet<M> = <M as SpaceParameterization>::SimpleSet;
pub type Element<M> = <M as SpaceParameterization>::Element;

#[derive(Default, Copy, Clone, Debug)]
pub struct ProductSet<A: SpaceParameterization, B: SpaceParameterization> {
    pub a: A,
    pub b: B,
}

impl<A: SpaceParameterization, B: SpaceParameterization> SpaceParameterization
    for ProductSet<A, B>
{
    type SimpleSet = (A::SimpleSet, B::SimpleSet);
    type Element = (A::Element, B::Element);

    const SPACE: Self::SimpleSet = (A::SPACE, B::SPACE);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct R;

impl SpaceParameterization for R {
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Self::SimpleSet = Bounds1D::new(f32::NEG_INFINITY, f32::INFINITY);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct UnitInterval;

impl SpaceParameterization for UnitInterval {
    type SimpleSet = Bounds1D;
    type Element = f32;
    const SPACE: Self::SimpleSet = Bounds1D::new(0.0, 1.0);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Circle;

impl SpaceParameterization for Circle {
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Bounds1D = Bounds1D::new(0.0, TAU);
}

/// includes the interior
pub type DiskSpace = ProductSet<Circle, UnitInterval>;

/// only the surface - space of directions, solid angle, and surface area
/// theta phi parameterization, aka spherical coordinates

#[derive(Default, Copy, Clone, Debug)]
pub struct SphericalCoordinates;

impl SpaceParameterization for SphericalCoordinates {
    type SimpleSet = Bounds2D;

    type Element = (f32, f32);

    const SPACE: Self::SimpleSet = Bounds2D::new(
        Bounds1D::new(0.0, TAU),
        Bounds1D::new(-FRAC_PI_2, FRAC_PI_2),
    );
}

/// includes interior
pub type SphericalCoordinatesBall = ProductSet<SphericalCoordinates, UnitInterval>;

#[derive(Default, Copy, Clone, Debug)]
pub struct DirectionalSector;

impl SpaceParameterization for DirectionalSector {
    // Directions are represented as raw `[f32; 3]` here rather than `Vec3<S>`
    // to keep `SpaceParameterization` non-generic over a SIMD backend. Callers
    // that want a `Vec3<S>` convert at the boundary.
    type SimpleSet = ([f32; 3], f32);
    type Element = [f32; 3];
    const SPACE: Self::SimpleSet = ([0.0, 0.0, 1.0], PI);
}
