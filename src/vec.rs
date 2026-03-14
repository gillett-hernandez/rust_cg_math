// use std::simd::{f32x4, f32x8};
use crate::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::{Add, MulAssign, Sub};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Copy, Clone, PartialEq, Default)]
pub struct Vec3(pub f32x4);

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3(f32x4::from_array([x, y, z, 0.0]))
    }
    pub const ZERO: Vec3 = Vec3(f32x4::ZERO);
    pub const MASK: f32x4 = f32x4::from_array([1.0, 1.0, 1.0, 0.0]);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub fn from_axis(axis: Axis) -> Vec3 {
        match axis {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Vec3 {
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
    pub fn as_array(&self) -> [f32; 4] {
        self.0.into()
    }
    pub fn cross(&self, other: Vec3) -> Self {
        let (x1, y1, z1) = (self.x(), self.y(), self.z());
        let (x2, y2, z2) = (other.x(), other.y(), other.z());
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0 * Vec3::MASK).reduce_sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3(self.0 / f32x4::splat(norm))
    }
}

impl Mul for Vec3 {
    type Output = f32;
    /// dot product
    fn mul(self, other: Vec3) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).reduce_sum()
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3(self.0 * f32x4::splat(other))
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3(f32x4::splat(self) * other.0)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3(self.0 / f32x4::splat(other))
    }
}

// impl Div for Vec3 {
//     type Output = Vec3;
//     fn div(self, other: Vec3) -> Vec3 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec3::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3(-self.0)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3(f32x4::splat(s) * Vec3::MASK)
    }
}

impl From<Vec3> for f32x4 {
    fn from(v: Vec3) -> f32x4 {
        v.0
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(other: [f32; 3]) -> Vec3 {
        Vec3::new(other[0], other[1], other[2])
    }
}

impl From<[f32; 4]> for Vec3 {
    fn from(other: [f32; 4]) -> Vec3 {
        Vec3(f32x4::from(other))
    }
}

impl From<f32x4> for Vec3 {
    fn from(other: f32x4) -> Vec3 {
        Vec3(other)
    }
}

impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        Vec3(p.0 * Vec3::MASK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_vec() {
        let v = Vec3::new(100.0, 0.2, 1.0);
        assert!(v.norm() > 100.0);
        assert!(v.norm_squared() > 10000.0);
        assert!(v.normalized().norm() - 1.0 < 0.000001);
    }

    fn arb_vec3() -> impl Strategy<Value = Vec3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    fn arb_nonzero_vec3() -> impl Strategy<Value = Vec3> {
        arb_vec3().prop_filter("nonzero", |v| v.norm() > 1e-6)
    }

    #[test]
    fn test_zero_identity() {
        let v = Vec3::new(3.0, -1.0, 7.0);
        let result = v + Vec3::ZERO;
        assert_eq!(result.x(), v.x());
        assert_eq!(result.y(), v.y());
        assert_eq!(result.z(), v.z());
    }

    #[test]
    fn test_w_is_zero() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.w(), 0.0);
    }

    #[test]
    fn test_from_axis() {
        assert_eq!(Vec3::from_axis(Axis::X), Vec3::X);
        assert_eq!(Vec3::from_axis(Axis::Y), Vec3::Y);
        assert_eq!(Vec3::from_axis(Axis::Z), Vec3::Z);
    }

    proptest! {
        #[test]
        fn dot_product_commutative(a in arb_vec3(), b in arb_vec3()) {
            let ab = a * b;
            let ba = b * a;
            prop_assert!((ab - ba).abs() < 1e-3, "a*b={}, b*a={}", ab, ba);
        }

        #[test]
        fn cross_product_orthogonal(a in arb_nonzero_vec3(), b in arb_nonzero_vec3()) {
            let c = a.cross(b);
            if c.norm() > 1e-6 {
                // use relative tolerance since absolute error scales with magnitude
                let scale = a.norm() * b.norm() * c.norm();
                let dot_a = (c * a).abs() / scale;
                let dot_b = (c * b).abs() / scale;
                prop_assert!(dot_a < 1e-4, "(a x b) . a / scale = {}", dot_a);
                prop_assert!(dot_b < 1e-4, "(a x b) . b / scale = {}", dot_b);
            }
        }

        #[test]
        fn cross_product_anticommutative(a in arb_vec3(), b in arb_vec3()) {
            let ab = a.cross(b);
            let ba = b.cross(a);
            let diff = (ab + ba).norm();
            prop_assert!(diff < 1e-3, "a x b + b x a = {}", diff);
        }

        #[test]
        fn normalization_produces_unit(v in arb_nonzero_vec3()) {
            let n = v.normalized();
            let norm = n.norm();
            prop_assert!((norm - 1.0).abs() < 1e-4, "||normalized|| = {}", norm);
        }

        #[test]
        fn norm_homogeneity(v in arb_nonzero_vec3(), k in -100.0f32..100.0) {
            let scaled_norm = (v * k).norm();
            let expected = k.abs() * v.norm();
            let rel_err = if expected > 1e-6 { (scaled_norm - expected).abs() / expected } else { (scaled_norm - expected).abs() };
            prop_assert!(rel_err < 1e-4, "||kv||={}, |k|*||v||={}, rel_err={}", scaled_norm, expected, rel_err);
        }

        #[test]
        fn add_sub_inverse(a in arb_vec3(), b in arb_vec3()) {
            let result = (a + b) - b;
            let diff = (result - a).norm();
            prop_assert!(diff < 1e-2, "(a+b)-b != a, diff={}", diff);
        }

        #[test]
        fn negation_inverse(v in arb_vec3()) {
            let result = v + (-v);
            prop_assert!(result.norm() < 1e-6, "v + (-v) != 0, got {:?}", result);
        }

        #[test]
        fn scalar_mul_distributive(a in arb_vec3(), b in arb_vec3(), s in -10.0f32..10.0) {
            let lhs = (a + b) * s;
            let rhs = a * s + b * s;
            let diff = (lhs - rhs).norm();
            prop_assert!(diff < 1e-1, "distributivity error = {}", diff);
        }

        #[test]
        fn scalar_mul_commutativity(v in arb_vec3(), s in -100.0f32..100.0) {
            let a = v * s;
            let b = s * v;
            prop_assert_eq!(a.x(), b.x());
            prop_assert_eq!(a.y(), b.y());
            prop_assert_eq!(a.z(), b.z());
        }

        #[test]
        fn from_array_roundtrip(x in -1e4f32..1e4, y in -1e4f32..1e4, z in -1e4f32..1e4) {
            let v = Vec3::new(x, y, z);
            prop_assert_eq!(v.x(), x);
            prop_assert_eq!(v.y(), y);
            prop_assert_eq!(v.z(), z);
        }

        #[test]
        fn is_finite_for_normal_values(v in arb_vec3()) {
            prop_assert!(v.is_finite());
        }
    }
}
