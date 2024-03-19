use crate::prelude::*;

use std::ops::{AddAssign, Sub, SubAssign};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::from_array([x, y, z, 1.0]))
    }
    pub const ZERO: Point3 = Point3(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));
    pub const ORIGIN: Point3 = Point3(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));
    pub const INFINITY: Point3 = Point3(f32x4::from_array([INFINITY, INFINITY, INFINITY, 1.0]));
    pub const NEG_INFINITY: Point3 =
        Point3(f32x4::from_array([-INFINITY, -INFINITY, -INFINITY, 1.0]));
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Point3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.0[0]
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.0[1]
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.0[2]
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        self.0[3]
    }
    pub fn normalize(mut self) -> Self {
        self.0 = self.0 / f32x4::splat(self.0[3]);
        self
    }
    pub fn as_array(&self) -> [f32; 4] {
        self.0.into()
    }
}

impl Default for Point3 {
    fn default() -> Self {
        Point3::ORIGIN
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        (self.0 + other.0).into()
    }
}

impl AddAssign<Vec3> for Point3 {
    fn add_assign(&mut self, other: Vec3) {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 += other.0
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        (self.0 - other.0).into()
    }
}

impl SubAssign<Vec3> for Point3 {
    fn sub_assign(&mut self, other: Vec3) {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 -= other.0
    }
}

// // don't implement adding or subtracting floats from Point3, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3((self.0 - other.0) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]))
    }
}

impl From<[f32; 3]> for Point3 {
    fn from(other: [f32; 3]) -> Point3 {
        Point3::new(other[0], other[1], other[2])
    }
}

impl From<f32x4> for Point3 {
    fn from(other: f32x4) -> Point3 {
        Point3(other)
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        // Point3::from_raw(v.0.replace(3, 1.0))
        Point3::ORIGIN + v
        // Point3::from_raw(v.0)
    }
}
