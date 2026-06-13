use crate::prelude::*;
use thermite::simd::Simd;
use thermite::register::LinAlg3Register;

use std::ops::{AddAssign, Sub, SubAssign};

/// 3D affine point, lane 3 held at 1.0. Generic over backend `S`.
pub struct Point3<S: Simd>(pub Vector<S::f32x4>);

impl<S: Simd> Copy for Point3<S> {}
impl<S: Simd> Clone for Point3<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Point3<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<S: Simd> std::fmt::Debug for Point3<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Point3")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl<S: Simd> Point3<S> {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Point3<S> {
        Point3(Vector::<S::f32x4>::new([x, y, z, 1.0]))
    }
    #[inline(always)]
    pub fn origin() -> Point3<S> {
        Point3::new(0.0, 0.0, 0.0)
    }
    #[inline(always)]
    pub fn zero() -> Point3<S> {
        Point3::new(0.0, 0.0, 0.0)
    }
    #[inline(always)]
    pub fn infinity() -> Point3<S> {
        Point3::new(INFINITY, INFINITY, INFINITY)
    }
    #[inline(always)]
    pub fn neg_infinity() -> Point3<S> {
        Point3::new(-INFINITY, -INFINITY, -INFINITY)
    }
    #[inline(always)]
    pub fn is_finite(&self) -> bool {
        let nan = <Vector<S::f32x4> as FloatVector>::is_nan(self.0);
        let inf = <Vector<S::f32x4> as FloatVector>::is_infinite(self.0);
        !(nan.any() || inf.any())
    }
}

impl<S: Simd> Point3<S> {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.0.extract::<0>()
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.0.extract::<1>()
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.0.extract::<2>()
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        self.0.extract::<3>()
    }
    /// Divide by w so the point is at w=1. No-op if already there.
    #[inline(always)]
    pub fn normalize(mut self) -> Self {
        let w = self.w();
        self.0 = self.0 / Vector::<S::f32x4>::splat(w);
        self
    }
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        [self.x(), self.y(), self.z(), self.w()]
    }
}

impl<S: Simd> Default for Point3<S> {
    #[inline(always)]
    fn default() -> Self {
        Point3::origin()
    }
}

impl<S: Simd> Add<Vec3<S>> for Point3<S> {
    type Output = Point3<S>;
    #[inline(always)]
    fn add(self, other: Vec3<S>) -> Point3<S> {
        Point3(self.0 + other.0)
    }
}

impl<S: Simd> AddAssign<Vec3<S>> for Point3<S> {
    #[inline(always)]
    fn add_assign(&mut self, other: Vec3<S>) {
        self.0 += other.0;
    }
}

impl<S: Simd> Sub<Vec3<S>> for Point3<S> {
    type Output = Point3<S>;
    #[inline(always)]
    fn sub(self, other: Vec3<S>) -> Point3<S> {
        Point3(self.0 - other.0)
    }
}

impl<S: Simd> SubAssign<Vec3<S>> for Point3<S> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Vec3<S>) {
        self.0 -= other.0;
    }
}

impl<S: Simd> Sub for Point3<S>
where
    S::f32x4: LinAlg3Register,
{
    type Output = Vec3<S>;
    #[inline(always)]
    fn sub(self, other: Point3<S>) -> Vec3<S> {
        // Subtracting two w=1 points yields w=0 (vector). zero4() makes that
        // explicit even if w lanes accumulated FP noise.
        Vec3((self.0 - other.0).zero4())
    }
}

impl<S: Simd> From<[f32; 3]> for Point3<S> {
    #[inline(always)]
    fn from(other: [f32; 3]) -> Point3<S> {
        Point3::new(other[0], other[1], other[2])
    }
}

impl<S: Simd> From<Vector<S::f32x4>> for Point3<S> {
    #[inline(always)]
    fn from(other: Vector<S::f32x4>) -> Point3<S> {
        Point3(other)
    }
}

impl<S: Simd> From<Vec3<S>> for Point3<S>
where
    S::f32x4: LinAlg3Register,
{
    #[inline(always)]
    fn from(v: Vec3<S>) -> Point3<S> {
        // Force lane 4 to 1.0.
        Point3(v.0.one4())
    }
}

impl<S: Simd> From<Point3<S>> for Vec3<S>
where
    S::f32x4: LinAlg3Register,
{
    #[inline(always)]
    fn from(p: Point3<S>) -> Self {
        // Drop the w lane.
        Vec3(p.0.zero4())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;
    type P3 = Point3<TestS>;

    fn arb_vec3() -> impl Strategy<Value = V3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| V3::new(x, y, z))
    }

    fn arb_point3() -> impl Strategy<Value = P3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| P3::new(x, y, z))
    }

    #[test]
    fn test_origin_equals_zero() {
        assert_eq!(P3::origin(), P3::zero());
        assert_eq!(P3::origin(), P3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_default_is_origin() {
        assert_eq!(P3::default(), P3::origin());
    }

    #[test]
    fn test_w_coordinate() {
        let p = P3::new(1.0, 2.0, 3.0);
        assert_eq!(p.w(), 1.0);
    }

    proptest! {
        #[test]
        fn component_access(x in -1e4f32..1e4, y in -1e4f32..1e4, z in -1e4f32..1e4) {
            let p = P3::new(x, y, z);
            prop_assert_eq!(p.x(), x);
            prop_assert_eq!(p.y(), y);
            prop_assert_eq!(p.z(), z);
            prop_assert_eq!(p.w(), 1.0);
        }

        #[test]
        fn point_sub_point_is_vec(p1 in arb_point3(), p2 in arb_point3()) {
            let v: V3 = p2 - p1;
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
            let p = P3::from(v);
            prop_assert!((p.x() - v.x()).abs() < 1e-6);
            prop_assert!((p.y() - v.y()).abs() < 1e-6);
            prop_assert!((p.z() - v.z()).abs() < 1e-6);
        }

        #[test]
        fn from_array_correctness(x in -1e4f32..1e4, y in -1e4f32..1e4, z in -1e4f32..1e4) {
            let p = P3::from([x, y, z]);
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
        assert!(!P3::infinity().is_finite());
        assert!(!P3::neg_infinity().is_finite());
    }

    #[test]
    fn test_debug_and_clone() {
        let p = P3::new(1.0, 2.0, 3.0);
        let s = format!("{:?}", p);
        assert!(s.contains("Point3"));
        let c = p.clone();
        assert_eq!(c, p);
    }

    #[test]
    fn test_as_array() {
        let p = P3::new(1.0, 2.0, 3.0);
        assert_eq!(p.as_array(), [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn test_add_assign_sub_assign() {
        let mut p = P3::new(1.0, 2.0, 3.0);
        p += V3::new(10.0, 20.0, 30.0);
        assert_eq!(p, P3::new(11.0, 22.0, 33.0));
        p -= V3::new(1.0, 2.0, 3.0);
        assert_eq!(p, P3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn test_from_vector_and_into_vec3() {
        let raw = Vector::<<TestS as thermite::simd::Simd>::f32x4>::new([1.0, 2.0, 3.0, 1.0]);
        let p: P3 = raw.into();
        assert_eq!(p, P3::new(1.0, 2.0, 3.0));
        // Point3 -> Vec3 drops the w lane.
        let v: V3 = p.into();
        assert_eq!(v.as_array(), [1.0, 2.0, 3.0, 0.0]);
    }
}
