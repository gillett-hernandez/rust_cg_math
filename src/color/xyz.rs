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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_color() -> impl Strategy<Value = XYZColor> {
        (0.0f32..10.0, 0.0f32..10.0, 0.0f32..10.0)
            .prop_map(|(x, y, z)| XYZColor::new(x, y, z))
    }

    #[test]
    fn test_black_is_zero() {
        let b = XYZColor::BLACK;
        assert_eq!(b.x(), 0.0);
        assert_eq!(b.y(), 0.0);
        assert_eq!(b.z(), 0.0);
    }

    #[test]
    fn test_black_add_identity() {
        let c = XYZColor::new(1.0, 2.0, 3.0);
        let result = c + XYZColor::BLACK;
        assert_eq!(result.x(), c.x());
        assert_eq!(result.y(), c.y());
        assert_eq!(result.z(), c.z());
    }

    proptest! {
        #[test]
        fn component_access(x in 0.0f32..10.0, y in 0.0f32..10.0, z in 0.0f32..10.0) {
            let c = XYZColor::new(x, y, z);
            prop_assert_eq!(c.x(), x);
            prop_assert_eq!(c.y(), y);
            prop_assert_eq!(c.z(), z);
        }

        #[test]
        fn addition_commutative(a in arb_color(), b in arb_color()) {
            let ab = a + b;
            let ba = b + a;
            prop_assert!((ab.x() - ba.x()).abs() < 1e-6);
            prop_assert!((ab.y() - ba.y()).abs() < 1e-6);
            prop_assert!((ab.z() - ba.z()).abs() < 1e-6);
        }

        #[test]
        fn scalar_mul_div_roundtrip(c in arb_color(), s in 0.1f32..100.0) {
            let result = (c * s) / s;
            prop_assert!((result.x() - c.x()).abs() < 1e-3, "x: {} vs {}", result.x(), c.x());
            prop_assert!((result.y() - c.y()).abs() < 1e-3, "y: {} vs {}", result.y(), c.y());
            prop_assert!((result.z() - c.z()).abs() < 1e-3, "z: {} vs {}", result.z(), c.z());
        }

        #[test]
        fn scalar_mul_commutativity(c in arb_color(), s in 0.0f32..10.0) {
            let a = c * s;
            let b = s * c;
            prop_assert_eq!(a.x(), b.x());
            prop_assert_eq!(a.y(), b.y());
            prop_assert_eq!(a.z(), b.z());
        }

        #[test]
        fn add_assign_same_as_add(a in arb_color(), b in arb_color()) {
            let sum = a + b;
            let mut assigned = a;
            assigned += b;
            prop_assert_eq!(sum.x(), assigned.x());
            prop_assert_eq!(sum.y(), assigned.y());
            prop_assert_eq!(sum.z(), assigned.z());
        }

        #[test]
        fn div_assign_same_as_div(c in arb_color(), s in 0.1f32..100.0) {
            let divided = c / s;
            let mut assigned = c;
            assigned /= s;
            prop_assert!((divided.x() - assigned.x()).abs() < 1e-6);
            prop_assert!((divided.y() - assigned.y()).abs() < 1e-6);
            prop_assert!((divided.z() - assigned.z()).abs() < 1e-6);
        }
    }
}
