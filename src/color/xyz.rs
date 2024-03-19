use std::ops::{Add, AddAssign, Div, DivAssign, Mul};
use std::simd::f32x4;

use crate::traits::Field;
#[derive(Copy, Clone, Debug)]
pub struct XYZColor(pub f32x4);

impl XYZColor {
    pub const fn new(x: f32, y: f32, z: f32) -> XYZColor {
        // XYZColor { x, y, z, w: 0.0 }
        XYZColor(f32x4::from_array([x, y, z, 0.0]))
    }
    pub const fn from_raw(v: f32x4) -> XYZColor {
        XYZColor(v)
    }
    pub const BLACK: XYZColor = XYZColor::from_raw(f32x4::ZERO);
    pub const ZERO: XYZColor = XYZColor::from_raw(f32x4::ZERO);
}

impl XYZColor {
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
}

impl Mul<f32> for XYZColor {
    type Output = XYZColor;
    fn mul(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 * f32x4::splat(other))
    }
}

impl Mul<XYZColor> for f32 {
    type Output = XYZColor;
    fn mul(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(other.0 * f32x4::splat(self))
    }
}

impl Div<f32> for XYZColor {
    type Output = XYZColor;
    fn div(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 / f32x4::splat(other))
    }
}

impl DivAssign<f32> for XYZColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / f32x4::splat(other);
    }
}

impl Add for XYZColor {
    type Output = XYZColor;
    fn add(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for XYZColor {
    fn add_assign(&mut self, other: XYZColor) {
        self.0 = self.0 + other.0
        // self.0 = (*self + other).0
    }
}

impl From<XYZColor> for f32x4 {
    fn from(v: XYZColor) -> f32x4 {
        v.0
    }
}
