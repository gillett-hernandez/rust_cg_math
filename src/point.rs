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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec3;
    use proptest::prelude::*;

    fn arb_vec3() -> impl Strategy<Value = Vec3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    fn arb_point3() -> impl Strategy<Value = Point3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4)
            .prop_map(|(x, y, z)| Point3::new(x, y, z))
    }

    #[test]
    fn test_origin_equals_zero() {
        assert_eq!(Point3::ORIGIN, Point3::ZERO);
        assert_eq!(Point3::ORIGIN, Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_default_is_origin() {
        assert_eq!(Point3::default(), Point3::ORIGIN);
    }

    #[test]
    fn test_w_coordinate() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.w(), 1.0);
    }

    proptest! {
        #[test]
        fn component_access(x in -1e4f32..1e4, y in -1e4f32..1e4, z in -1e4f32..1e4) {
            let p = Point3::new(x, y, z);
            prop_assert_eq!(p.x(), x);
            prop_assert_eq!(p.y(), y);
            prop_assert_eq!(p.z(), z);
            prop_assert_eq!(p.w(), 1.0);
        }

        #[test]
        fn point_sub_point_is_vec(p1 in arb_point3(), p2 in arb_point3()) {
            let v: Vec3 = p2 - p1;
            // p1 + v should equal p2
            let result = p1 + v;
            let diff = (result - p2).norm();
            prop_assert!(diff < 1e-2, "p1 + (p2 - p1) != p2, diff={}", diff);
        }

        #[test]
        fn point_add_sub_vec_roundtrip(p in arb_point3(), v in arb_vec3()) {
            let result = (p + v) - v;
            let diff = (result - p).norm();
            prop_assert!(diff < 1e-2, "(p + v) - v != p, diff={}", diff);
        }

        #[test]
        fn point_sub_vec_roundtrip(p in arb_point3(), v in arb_vec3()) {
            let result = (p - v) + v;
            let diff = (result - p).norm();
            prop_assert!(diff < 1e-2, "(p - v) + v != p, diff={}", diff);
        }

        #[test]
        fn from_vec3_correctness(v in arb_vec3()) {
            let p = Point3::from(v);
            prop_assert!((p.x() - v.x()).abs() < 1e-6);
            prop_assert!((p.y() - v.y()).abs() < 1e-6);
            prop_assert!((p.z() - v.z()).abs() < 1e-6);
        }

        #[test]
        fn from_array_correctness(x in -1e4f32..1e4, y in -1e4f32..1e4, z in -1e4f32..1e4) {
            let p = Point3::from([x, y, z]);
            prop_assert_eq!(p.x(), x);
            prop_assert_eq!(p.y(), y);
            prop_assert_eq!(p.z(), z);
        }

        #[test]
        fn is_finite_for_normal_values(p in arb_point3()) {
            prop_assert!(p.is_finite());
        }
    }

    #[test]
    fn test_infinity_is_not_finite() {
        assert!(!Point3::INFINITY.is_finite());
        assert!(!Point3::NEG_INFINITY.is_finite());
    }
}
