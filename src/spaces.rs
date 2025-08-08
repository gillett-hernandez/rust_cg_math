// use std::marker::PhantomData;
// use typenum::Unsigned;

use crate::prelude::*;

use std::f32::consts::{FRAC_PI_2, TAU};

// pub struct R<T: Unsigned>(PhantomData<T>);

pub trait SpaceParameterization {
    type SimpleSet;
    type Element;
    const SPACE: Self::SimpleSet;
}

pub type SimpleSet<M> = <<M as Measure>::Space as SpaceParameterization>::SimpleSet;
pub type Element<M> = <<M as Measure>::Space as SpaceParameterization>::Element;

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

pub struct R1;

impl SpaceParameterization for R1 {
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Self::SimpleSet = Bounds1D::new(f32::NEG_INFINITY, f32::INFINITY);
}

pub struct UnitInterval;

impl SpaceParameterization for UnitInterval {
    type SimpleSet = Bounds1D;
    type Element = f32;
    const SPACE: Self::SimpleSet = Bounds1D::new(0.0, 1.0);
}

pub struct Circle;

impl SpaceParameterization for Circle {
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Bounds1D = Bounds1D::new(0.0, TAU);
}

/// includes the interior
pub type DiskSpace = ProductSet<Circle, UnitInterval>;

/// only the surface - space of directions, solid angle, and surface area
pub struct Sphere;

impl SpaceParameterization for Sphere {
    type SimpleSet = Bounds2D;

    type Element = (f32, f32);

    const SPACE: Self::SimpleSet = Bounds2D::new(
        Bounds1D::new(0.0, TAU),
        Bounds1D::new(-FRAC_PI_2, FRAC_PI_2),
    );
}

/// includes interior
pub type Ball = ProductSet<Sphere, UnitInterval>;
