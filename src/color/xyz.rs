use std::ops::{Add, AddAssign, Div, DivAssign, Mul};

use thermite::Vector;
use thermite::simd::Simd;
use thermite::vector::{GenericVector, NumericVector};

/// CIE XYZ tristimulus values stored in lanes 0..3 of an `f32x4`, lane 4 = 0.

pub struct XYZColor<S: Simd>(pub Vector<S::f32x4>);
impl<S: Simd> Copy for XYZColor<S> {}
impl<S: Simd> Clone for XYZColor<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> std::fmt::Debug for XYZColor<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("XYZColor")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl<S: Simd> XYZColor<S> {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> XYZColor<S> {
        XYZColor(Vector::<S::f32x4>::new([x, y, z, 0.0]))
    }
    #[inline(always)]
    pub fn from_raw(v: Vector<S::f32x4>) -> XYZColor<S> {
        XYZColor(v)
    }
    #[inline(always)]
    pub fn black() -> XYZColor<S> {
        XYZColor(<Vector<S::f32x4> as NumericVector>::ZERO)
    }
    #[inline(always)]
    pub fn zero() -> XYZColor<S> {
        Self::black()
    }
}

impl<S: Simd> XYZColor<S> {
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
}

impl<S: Simd> Mul<f32> for XYZColor<S> {
    type Output = XYZColor<S>;
    #[inline(always)]
    fn mul(self, other: f32) -> XYZColor<S> {
        XYZColor::from_raw(self.0 * Vector::<S::f32x4>::splat(other))
    }
}

impl<S: Simd> Mul<XYZColor<S>> for f32 {
    type Output = XYZColor<S>;
    #[inline(always)]
    fn mul(self, other: XYZColor<S>) -> XYZColor<S> {
        other * self
    }
}

impl<S: Simd> Div<f32> for XYZColor<S> {
    type Output = XYZColor<S>;
    #[inline(always)]
    fn div(self, other: f32) -> XYZColor<S> {
        XYZColor::from_raw(self.0 / Vector::<S::f32x4>::splat(other))
    }
}

impl<S: Simd> DivAssign<f32> for XYZColor<S> {
    #[inline(always)]
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / Vector::<S::f32x4>::splat(other);
    }
}

impl<S: Simd> Add for XYZColor<S> {
    type Output = XYZColor<S>;
    #[inline(always)]
    fn add(self, other: XYZColor<S>) -> XYZColor<S> {
        XYZColor::from_raw(self.0 + other.0)
    }
}

impl<S: Simd> AddAssign for XYZColor<S> {
    #[inline(always)]
    fn add_assign(&mut self, other: XYZColor<S>) {
        self.0 = self.0 + other.0;
    }
}

impl<S: Simd> Into<Vector<S::f32x4>> for XYZColor<S> {
    #[inline(always)]
    fn into(self) -> Vector<S::f32x4> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type C = XYZColor<TestS>;

    fn arb_color() -> impl Strategy<Value = C> {
        (0.0f32..10.0, 0.0f32..10.0, 0.0f32..10.0).prop_map(|(x, y, z)| C::new(x, y, z))
    }

    #[test]
    fn test_black_is_zero() {
        let b = C::black();
        assert_eq!(b.x(), 0.0);
        assert_eq!(b.y(), 0.0);
        assert_eq!(b.z(), 0.0);
    }

    #[test]
    fn test_black_add_identity() {
        let c = C::new(1.0, 2.0, 3.0);
        let result = c + C::black();
        assert_eq!(result.x(), c.x());
        assert_eq!(result.y(), c.y());
        assert_eq!(result.z(), c.z());
    }

    #[test]
    fn test_debug_and_clone() {
        let c = C::new(1.0, 2.0, 3.0);
        let s = format!("{:?}", c);
        assert!(s.contains("XYZColor"));
        let cloned = c.clone();
        assert_eq!(cloned.x(), c.x());
        assert_eq!(cloned.y(), c.y());
        assert_eq!(cloned.z(), c.z());
    }

    #[test]
    fn test_zero_is_black() {
        let z = C::zero();
        assert_eq!(z.x(), 0.0);
        assert_eq!(z.y(), 0.0);
        assert_eq!(z.z(), 0.0);
    }

    #[test]
    fn test_into_vector() {
        let c = C::new(1.0, 2.0, 3.0);
        let raw: Vector<<TestS as Simd>::f32x4> = c.into();
        assert_eq!(raw.extract::<0>(), 1.0);
        assert_eq!(raw.extract::<1>(), 2.0);
        assert_eq!(raw.extract::<2>(), 3.0);
    }

    proptest! {
        #[test]
        fn component_access(x in 0.0f32..10.0, y in 0.0f32..10.0, z in 0.0f32..10.0) {
            let c = C::new(x, y, z);
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
