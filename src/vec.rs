use crate::prelude::*;
use thermite::register::LinAlg3Register;
use thermite::simd::Simd;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, MulAssign, Sub};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// 3D vector, lane 3 is held at 0. Generic over backend `S` — the inner
/// register is `S::f32x4`.
pub struct Vec3<S: Simd>(pub Vector<S::f32x4>);

// Manual derives because S itself is non-Copy/Clone (it's a tag type, but
// derive generates incorrect `S: Copy` bounds otherwise).
impl<S: Simd> Copy for Vec3<S> {}
impl<S: Simd> Clone for Vec3<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Vec3<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<S: Simd> Default for Vec3<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<S: Simd> fmt::Debug for Vec3<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl<S: Simd> Vec3<S> {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Vec3<S> {
        Vec3(Vector::<S::f32x4>::new([x, y, z, 0.0]))
    }

    // Replacements for the old `const ZERO`/`const X`/... constants. Not
    // const because `Vector::new` isn't const fn.
    pub const ZERO: Self = {
        // SAFETY: we rely on `Vector::<S::f32x4>::ZERO` being a const from
        // `NumericVector`. `Vec3` wraps it directly.
        Vec3(<Vector<S::f32x4> as NumericVector>::ZERO)
    };

    #[inline(always)]
    pub fn x_axis() -> Vec3<S> {
        Vec3::new(1.0, 0.0, 0.0)
    }
    #[inline(always)]
    pub fn y_axis() -> Vec3<S> {
        Vec3::new(0.0, 1.0, 0.0)
    }
    #[inline(always)]
    pub fn z_axis() -> Vec3<S> {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[inline(always)]
    pub fn from_axis(axis: Axis) -> Vec3<S> {
        match axis {
            Axis::X => Vec3::x_axis(),
            Axis::Y => Vec3::y_axis(),
            Axis::Z => Vec3::z_axis(),
        }
    }
    #[inline(always)]
    pub fn is_finite(&self) -> bool {
        let nan = <Vector<S::f32x4> as FloatVector>::is_nan(self.0);
        let inf = <Vector<S::f32x4> as FloatVector>::is_infinite(self.0);
        !(nan.any() || inf.any())
    }
}

impl<S: Simd> Vec3<S> {
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
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        [self.x(), self.y(), self.z(), self.w()]
    }
}

// LinAlg3 ops (dot3, cross3) need S::f32x4: LinAlg3Register.
impl<S: Simd> Vec3<S>
where
    S::f32x4: LinAlg3Register,
{
    #[inline(always)]
    pub fn cross(&self, other: Vec3<S>) -> Self {
        // DOP=false: simpler formula. Anticommutativity (a×b + b×a == 0) holds
        // bit-exactly because every product cancels with its counterpart in
        // the swapped call. DOP=true's correction term breaks that — its err
        // captures rounding noise that doesn't cancel across calls.
        Vec3(self.0.cross3::<false>(other.0))
    }

    #[inline(always)]
    pub fn norm_squared(&self) -> f32 {
        self.0.dot3(self.0)
    }

    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalized(&self) -> Self {
        Vec3(self.0 / Vector::<S::f32x4>::splat(self.norm()))
    }
}

impl<S: Simd> Mul for Vec3<S>
where
    S::f32x4: LinAlg3Register,
{
    type Output = f32;
    /// dot product (3D, ignoring lane 4)
    #[inline(always)]
    fn mul(self, other: Vec3<S>) -> f32 {
        self.0.dot3(other.0)
    }
}

impl<S: Simd> MulAssign for Vec3<S> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Vec3<S>) {
        self.0 = self.0 * other.0;
    }
}

impl<S: Simd> Mul<f32> for Vec3<S> {
    type Output = Vec3<S>;
    #[inline(always)]
    fn mul(self, other: f32) -> Vec3<S> {
        Vec3(self.0 * Vector::<S::f32x4>::splat(other))
    }
}

impl<S: Simd> Mul<Vec3<S>> for f32 {
    type Output = Vec3<S>;
    #[inline(always)]
    fn mul(self, other: Vec3<S>) -> Vec3<S> {
        other * self
    }
}

impl<S: Simd> Div<f32> for Vec3<S> {
    type Output = Vec3<S>;
    #[inline(always)]
    fn div(self, other: f32) -> Vec3<S> {
        Vec3(self.0 / Vector::<S::f32x4>::splat(other))
    }
}

impl<S: Simd> Add for Vec3<S> {
    type Output = Vec3<S>;
    #[inline(always)]
    fn add(self, other: Vec3<S>) -> Vec3<S> {
        Vec3(self.0 + other.0)
    }
}

impl<S: Simd> Neg for Vec3<S> {
    type Output = Vec3<S>;
    #[inline(always)]
    fn neg(self) -> Vec3<S> {
        Vec3(-self.0)
    }
}

impl<S: Simd> Sub for Vec3<S> {
    type Output = Vec3<S>;
    #[inline(always)]
    fn sub(self, other: Vec3<S>) -> Vec3<S> {
        Vec3(self.0 - other.0)
    }
}

impl<S: Simd> From<f32> for Vec3<S> {
    #[inline(always)]
    fn from(s: f32) -> Vec3<S> {
        // splat s into the 3 lanes, lane 4 = 0
        Vec3::new(s, s, s)
    }
}

impl<S: Simd> From<Vec3<S>> for Vector<S::f32x4> {
    #[inline(always)]
    fn from(v: Vec3<S>) -> Vector<S::f32x4> {
        v.0
    }
}

impl<S: Simd> From<[f32; 3]> for Vec3<S> {
    #[inline(always)]
    fn from(other: [f32; 3]) -> Vec3<S> {
        Vec3::new(other[0], other[1], other[2])
    }
}

impl<S: Simd> From<[f32; 4]> for Vec3<S> {
    #[inline(always)]
    fn from(other: [f32; 4]) -> Vec3<S> {
        Vec3(Vector::<S::f32x4>::new(other))
    }
}

impl<S: Simd> From<Vector<S::f32x4>> for Vec3<S> {
    #[inline(always)]
    fn from(v: Vector<S::f32x4>) -> Vec3<S> {
        Vec3(v)
    }
}

// Avoid unused-PhantomData warnings if S becomes unused in any future variant.
#[allow(dead_code)]
#[inline(always)]
fn _phantom_s_marker<S: Simd>() -> PhantomData<S> {
    PhantomData
}

// Cross-type conversion lives in `point.rs` (impl From<Point3<S>> for Vec3<S>)
// to keep the orphan rule on the right side of the wrapping type.

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Tests use the scalar backend's f32x4 (Vector<ArrayRegister<f32, 4>>),
    // which is portable and exercises the LinAlg3/Float code paths without
    // requiring x86-specific intrinsics.
    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;

    #[test]
    fn test_vec() {
        let v = V3::new(100.0, 0.2, 1.0);
        assert!(v.norm() > 100.0);
        assert!(v.norm_squared() > 10000.0);
        assert!(v.normalized().norm() - 1.0 < 0.000001);
    }

    fn arb_vec3() -> impl Strategy<Value = V3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| V3::new(x, y, z))
    }

    fn arb_nonzero_vec3() -> impl Strategy<Value = V3> {
        arb_vec3().prop_filter("nonzero", |v| v.norm() > 1e-6)
    }

    #[test]
    fn test_zero_identity() {
        let v = V3::new(3.0, -1.0, 7.0);
        let result = v + V3::ZERO;
        assert_eq!(result.x(), v.x());
        assert_eq!(result.y(), v.y());
        assert_eq!(result.z(), v.z());
    }

    #[test]
    fn test_w_is_zero() {
        let v = V3::new(1.0, 2.0, 3.0);
        assert_eq!(v.w(), 0.0);
    }

    #[test]
    fn test_from_axis() {
        assert_eq!(V3::from_axis(Axis::X), V3::x_axis());
        assert_eq!(V3::from_axis(Axis::Y), V3::y_axis());
        assert_eq!(V3::from_axis(Axis::Z), V3::z_axis());
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
            let v = V3::new(x, y, z);
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
