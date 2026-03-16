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
    // funnily enough, space-filling non-overlapping sets composed of this look an awful lot like an apollonian gasket
    // but on a sphere
    type SimpleSet = (Vec3, f32);
    // space is slightly redundant, as when the set span is PI, every direction represents the same subset
    type Element = Vec3;
    const SPACE: Self::SimpleSet = (Vec3::Z, PI);
}
