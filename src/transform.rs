use crate::prelude::*;
use thermite::register::{LinAlg3Register, LinAlg4Register};
use thermite::simd::Simd;
use thermite::vector::LinAlg4Vector;

/// 4x4 matrix stored as 4 column vectors (column-major, matches nalgebra's
/// memory layout). Mat-vec and mat-mat operations dispatch to thermite's
/// `LinAlg4Vector::mat4_*` primitives.
pub struct Matrix4x4<S: Simd>(pub [Vector<S::f32x4>; 4]);

impl<S: Simd> Copy for Matrix4x4<S> {}
impl<S: Simd> Clone for Matrix4x4<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Matrix4x4<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<S: Simd> std::fmt::Debug for Matrix4x4<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Matrix4x4").field(&self.as_array()).finish()
    }
}

impl<S: Simd> Matrix4x4<S> {
    #[inline(always)]
    pub fn identity() -> Matrix4x4<S> {
        Matrix4x4([
            Vector::<S::f32x4>::new([1.0, 0.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 1.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 1.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    /// Column-major flat layout: `out[col * 4 + row] = m[row][col]`.
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 16] {
        let mut out = [0.0_f32; 16];
        for c in 0..4 {
            let col = self.0[c];
            out[c * 4 + 0] = col.extract::<0>();
            out[c * 4 + 1] = col.extract::<1>();
            out[c * 4 + 2] = col.extract::<2>();
            out[c * 4 + 3] = col.extract::<3>();
        }
        out
    }

    #[inline(always)]
    pub fn from_array(values: [f32; 16]) -> Self {
        Matrix4x4([
            Vector::<S::f32x4>::new([values[0], values[1], values[2], values[3]]),
            Vector::<S::f32x4>::new([values[4], values[5], values[6], values[7]]),
            Vector::<S::f32x4>::new([values[8], values[9], values[10], values[11]]),
            Vector::<S::f32x4>::new([values[12], values[13], values[14], values[15]]),
        ])
    }
}

impl<S: Simd> Matrix4x4<S>
where
    S::f32x4: LinAlg4Register,
{
    #[inline(always)]
    pub fn transpose(&self) -> Matrix4x4<S> {
        Matrix4x4(<Vector<S::f32x4> as LinAlg4Vector>::mat4_transpose(&self.0))
    }
}

impl<S: Simd> Matrix4x4<S> {
    /// General 4x4 inverse via the cofactor / adjugate closed form. Returns
    /// `None` if the matrix is singular (`|det| < f32::EPSILON`). Runs at
    /// construction time (once per transform), so a scalar formulation is fine.
    ///
    /// This stays scalar deliberately. A SIMD reformulation was profiled and
    /// lost: Lengyel's cross/dot block form regressed (the per-column dot
    /// products are horizontal reductions, plus scalar `extract`/insert
    /// round-trips to read the bottom row and rebuild the 4th column), and the
    /// shuffle-based GLM form is unreachable in fast form — thermite's only
    /// backend-portable shuffle (`swizzle!`/`SwizzleRegister`) lowers to a
    /// `permutevar`+`blend` sequence on x86 rather than an immediate `vshufps`,
    /// and the immediate `ShuffleRegister` path isn't implemented for the
    /// scalar (`ArrayRegister`) test backend. For one lone 4x4 the cross-lane
    /// data movement costs more than straight-line scalar FMAs save, so the
    /// compiler-optimized scalar version below is the fastest portable option.
    /// See `benches/math_benches.rs::mat4_try_inverse`.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<Matrix4x4<S>> {
        // Row-major flat view: `m[row * 4 + col]`. (as_array() is column-major,
        // so transpose the indexing here.) Cofactor formula is written against
        // row-major below for readability, then re-emitted column-major.
        let a = self.as_array(); // column-major: a[col*4 + row]
        let m = |row: usize, col: usize| a[col * 4 + row];

        let m00 = m(0, 0);
        let m01 = m(0, 1);
        let m02 = m(0, 2);
        let m03 = m(0, 3);
        let m10 = m(1, 0);
        let m11 = m(1, 1);
        let m12 = m(1, 2);
        let m13 = m(1, 3);
        let m20 = m(2, 0);
        let m21 = m(2, 1);
        let m22 = m(2, 2);
        let m23 = m(2, 3);
        let m30 = m(3, 0);
        let m31 = m(3, 1);
        let m32 = m(3, 2);
        let m33 = m(3, 3);

        // 2x2 sub-determinants of the bottom two rows (s) and top two rows (c).
        let s0 = m00 * m11 - m10 * m01;
        let s1 = m00 * m12 - m10 * m02;
        let s2 = m00 * m13 - m10 * m03;
        let s3 = m01 * m12 - m11 * m02;
        let s4 = m01 * m13 - m11 * m03;
        let s5 = m02 * m13 - m12 * m03;

        let c5 = m22 * m33 - m32 * m23;
        let c4 = m21 * m33 - m31 * m23;
        let c3 = m21 * m32 - m31 * m22;
        let c2 = m20 * m33 - m30 * m23;
        let c1 = m20 * m32 - m30 * m22;
        let c0 = m20 * m31 - m30 * m21;

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;

        // Inverse entries in row-major `r[row][col]`.
        let r00 = (m11 * c5 - m12 * c4 + m13 * c3) * inv_det;
        let r01 = (-m01 * c5 + m02 * c4 - m03 * c3) * inv_det;
        let r02 = (m31 * s5 - m32 * s4 + m33 * s3) * inv_det;
        let r03 = (-m21 * s5 + m22 * s4 - m23 * s3) * inv_det;

        let r10 = (-m10 * c5 + m12 * c2 - m13 * c1) * inv_det;
        let r11 = (m00 * c5 - m02 * c2 + m03 * c1) * inv_det;
        let r12 = (-m30 * s5 + m32 * s2 - m33 * s1) * inv_det;
        let r13 = (m20 * s5 - m22 * s2 + m23 * s1) * inv_det;

        let r20 = (m10 * c4 - m11 * c2 + m13 * c0) * inv_det;
        let r21 = (-m00 * c4 + m01 * c2 - m03 * c0) * inv_det;
        let r22 = (m30 * s4 - m31 * s2 + m33 * s0) * inv_det;
        let r23 = (-m20 * s4 + m21 * s2 - m23 * s0) * inv_det;

        let r30 = (-m10 * c3 + m11 * c1 - m12 * c0) * inv_det;
        let r31 = (m00 * c3 - m01 * c1 + m02 * c0) * inv_det;
        let r32 = (-m30 * s3 + m31 * s1 - m32 * s0) * inv_det;
        let r33 = (m20 * s3 - m21 * s1 + m22 * s0) * inv_det;

        // Re-emit column-major: each column is one f32x4.
        Some(Matrix4x4([
            Vector::<S::f32x4>::new([r00, r10, r20, r30]),
            Vector::<S::f32x4>::new([r01, r11, r21, r31]),
            Vector::<S::f32x4>::new([r02, r12, r22, r32]),
            Vector::<S::f32x4>::new([r03, r13, r23, r33]),
        ]))
    }
}

impl<S: Simd> Mul<Vec3<S>> for Matrix4x4<S>
where
    S::f32x4: LinAlg4Register,
{
    type Output = Vec3<S>;
    #[inline(always)]
    fn mul(self, rhs: Vec3<S>) -> Self::Output {
        // Vec3 has w=0, so mat4_vec3_product is the appropriate primitive —
        // it skips the translation column and avoids the homogenization step.
        // COLUMN_MAJOR=true because our `self.0` stores columns directly.
        Vec3(rhs.0.mat4_vec3_product::<true>(&self.0))
    }
}

impl<S: Simd> Mul<Point3<S>> for Matrix4x4<S>
where
    S::f32x4: LinAlg4Register,
{
    type Output = Point3<S>;
    #[inline(always)]
    fn mul(self, rhs: Point3<S>) -> Self::Output {
        // Point3 has w=1, so mat4_vec4_product applies the full 4x4 (including
        // the translation column). `normalize()` divides by the resulting w to
        // bring w back to 1 (no-op for affine transforms).
        Point3(rhs.0.mat4_vec4_product::<true>(&self.0)).normalize()
    }
}

impl<S: Simd> Mul<Ray<S>> for Matrix4x4<S>
where
    S::f32x4: LinAlg3Register + LinAlg4Register,
{
    type Output = Ray<S>;
    #[inline(always)]
    fn mul(self, rhs: Ray<S>) -> Self::Output {
        Ray {
            origin: self * rhs.origin,
            direction: (self * rhs.direction).normalized(),
            ..rhs
        }
    }
}

impl<S: Simd> Mul for Matrix4x4<S>
where
    S::f32x4: LinAlg4Register,
{
    type Output = Matrix4x4<S>;
    #[inline(always)]
    fn mul(self, rhs: Matrix4x4<S>) -> Self::Output {
        Matrix4x4(<Vector<S::f32x4> as LinAlg4Vector>::mat4_product::<true>(
            &self.0, &rhs.0,
        ))
    }
}

pub struct Transform3<S: Simd> {
    pub forward: Matrix4x4<S>,
    pub reverse: Matrix4x4<S>,
}

impl<S: Simd> Copy for Transform3<S> {}
impl<S: Simd> Clone for Transform3<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Transform3<S> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.forward == other.forward && self.reverse == other.reverse
    }
}
impl<S: Simd> std::fmt::Debug for Transform3<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transform3")
            .field("forward", &self.forward)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<S: Simd> Transform3<S>
where
    S::f32x4: LinAlg4Register,
{
    #[inline(always)]
    pub fn new() -> Self {
        Transform3 {
            forward: Matrix4x4::identity(),
            reverse: Matrix4x4::identity(),
        }
    }
    /// Build a transform from an arbitrary forward matrix, computing the
    /// reverse via a general 4x4 inverse. Returns `None` if `forward` is
    /// singular.
    #[inline(always)]
    pub fn new_from_matrix(forward: Matrix4x4<S>) -> Option<Self> {
        forward.try_inverse().map(|reverse| Transform3 { forward, reverse })
    }

    #[inline(always)]
    pub fn inverse(self) -> Transform3<S> {
        Transform3::new_from_raw(self.reverse, self.forward)
    }

    #[inline(always)]
    pub fn from_translation(shift: Vec3<S>) -> Self {
        let (tx, ty, tz) = (shift.x(), shift.y(), shift.z());
        // Affine translation: inverse is translation by the negated shift.
        // Columns 0..2 are the identity basis; column 3 holds the translation.
        let forward = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([1.0, 0.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 1.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 1.0, 0.0]),
            Vector::<S::f32x4>::new([tx, ty, tz, 1.0]),
        ]);
        let reverse = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([1.0, 0.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 1.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 1.0, 0.0]),
            Vector::<S::f32x4>::new([-tx, -ty, -tz, 1.0]),
        ]);
        Transform3::new_from_raw(forward, reverse)
    }

    #[inline(always)]
    pub fn from_scale(scale: Vec3<S>) -> Self {
        let (sx, sy, sz) = (scale.x(), scale.y(), scale.z());
        // Diagonal scale: inverse is the reciprocal scale.
        let forward = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([sx, 0.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, sy, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, sz, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ]);
        let reverse = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([1.0 / sx, 0.0, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 1.0 / sy, 0.0, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 1.0 / sz, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ]);
        Transform3::new_from_raw(forward, reverse)
    }

    /// Rotation about a (unit-length) `axis` by `radians`, via Rodrigues'
    /// formula. The inverse of an orthonormal rotation is its transpose, so the
    /// reverse matrix is built directly without a general inversion.
    #[inline(always)]
    pub fn from_axis_angle(axis: Vec3<S>, radians: f32) -> Self {
        let (x, y, z) = (axis.x(), axis.y(), axis.z());
        let c = radians.cos();
        let s = radians.sin();
        let t = 1.0 - c;

        // 3x3 rotation entries r[row][col] (Rodrigues).
        let r00 = t * x * x + c;
        let r01 = t * x * y - s * z;
        let r02 = t * x * z + s * y;
        let r10 = t * x * y + s * z;
        let r11 = t * y * y + c;
        let r12 = t * y * z - s * x;
        let r20 = t * x * z - s * y;
        let r21 = t * y * z + s * x;
        let r22 = t * z * z + c;

        // Forward: columns are (r[0][col], r[1][col], r[2][col], 0).
        let forward = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([r00, r10, r20, 0.0]),
            Vector::<S::f32x4>::new([r01, r11, r21, 0.0]),
            Vector::<S::f32x4>::new([r02, r12, r22, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ]);
        // Reverse is the transpose: columns are the rows of `forward`.
        let reverse = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([r00, r01, r02, 0.0]),
            Vector::<S::f32x4>::new([r10, r11, r12, 0.0]),
            Vector::<S::f32x4>::new([r20, r21, r22, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ]);
        Transform3::new_from_raw(forward, reverse)
    }

    #[inline(always)]
    pub fn from_stack(
        scale: Option<Transform3<S>>,
        rotate: Option<Transform3<S>>,
        translate: Option<Transform3<S>>,
    ) -> Transform3<S> {
        let mut stack = Transform3::new();
        if let Some(scale) = scale {
            stack = scale * stack;
        }
        if let Some(rotate) = rotate {
            stack = rotate * stack;
        }
        if let Some(translate) = translate {
            stack = translate * stack;
        }
        stack
    }

    #[inline(always)]
    pub fn new_from_raw(forward: Matrix4x4<S>, reverse: Matrix4x4<S>) -> Self {
        Transform3 { forward, reverse }
    }

    #[inline(always)]
    pub fn from_vector_stack(
        v0: Vector<S::f32x4>,
        v1: Vector<S::f32x4>,
        v2: Vector<S::f32x4>,
    ) -> Self {
        // Original `m`: columns (v0, v1, v2, e3) where e3 = (0,0,0,1). Each of
        // v0/v1/v2 is a tangent-frame basis vector in lanes 0..2 with lane 3=0.
        // `forward` is `m.transpose()`; `reverse` is `m` (since the inverse of
        // an orthonormal frame is its transpose).
        let extract3 = |v: Vector<S::f32x4>| (v.extract::<0>(), v.extract::<1>(), v.extract::<2>());
        let (m11, m12, m13) = extract3(v0);
        let (m21, m22, m23) = extract3(v1);
        let (m31, m32, m33) = extract3(v2);

        let m = Matrix4x4::<S>([
            Vector::<S::f32x4>::new([m11, m12, m13, 0.0]),
            Vector::<S::f32x4>::new([m21, m22, m23, 0.0]),
            Vector::<S::f32x4>::new([m31, m32, m33, 0.0]),
            Vector::<S::f32x4>::new([0.0, 0.0, 0.0, 1.0]),
        ]);
        Transform3::new_from_raw(m.transpose(), m)
    }

    #[inline(always)]
    pub fn axis_transform(&self) -> (Vec3<S>, Vec3<S>, Vec3<S>) {
        (
            self.to_world(Vec3::x_axis()),
            self.to_world(Vec3::y_axis()),
            self.to_world(Vec3::z_axis()),
        )
    }

    #[inline(always)]
    pub fn to_local<T>(&self, value: T) -> <Matrix4x4<S> as Mul<T>>::Output
    where
        Matrix4x4<S>: Mul<T>,
    {
        self.reverse * value
    }
    #[inline(always)]
    pub fn to_world<T>(&self, value: T) -> <Matrix4x4<S> as Mul<T>>::Output
    where
        Matrix4x4<S>: Mul<T>,
    {
        self.forward * value
    }
}

impl<S: Simd> From<TangentFrame<S>> for Transform3<S>
where
    S::f32x4: LinAlg3Register + LinAlg4Register,
{
    #[inline(always)]
    fn from(value: TangentFrame<S>) -> Self {
        Transform3::from_vector_stack(value.tangent.0, value.bitangent.0, value.normal.0)
    }
}

impl<S: Simd> Mul<Transform3<S>> for Transform3<S>
where
    S::f32x4: LinAlg4Register,
{
    type Output = Transform3<S>;
    #[inline(always)]
    fn mul(self, rhs: Transform3<S>) -> Self::Output {
        Transform3::new_from_raw(rhs.forward * self.forward, self.reverse * rhs.reverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;
    type P3 = Point3<TestS>;
    type T3 = Transform3<TestS>;
    type M4 = Matrix4x4<TestS>;

    fn arb_vec3() -> impl Strategy<Value = V3> {
        (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0).prop_map(|(x, y, z)| V3::new(x, y, z))
    }

    fn arb_unit_vec3() -> impl Strategy<Value = V3> {
        arb_vec3()
            .prop_filter("nonzero", |v| v.norm() > 0.01)
            .prop_map(|v| v.normalized())
    }

    fn arb_point3() -> impl Strategy<Value = P3> {
        (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0).prop_map(|(x, y, z)| P3::new(x, y, z))
    }

    fn arb_rotation() -> impl Strategy<Value = T3> {
        (arb_unit_vec3(), -PI..PI).prop_map(|(axis, angle)| T3::from_axis_angle(axis, angle))
    }

    fn arb_translation() -> impl Strategy<Value = T3> {
        arb_vec3().prop_map(|v| T3::from_translation(v))
    }

    fn arb_nonzero_scale() -> impl Strategy<Value = T3> {
        (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0)
            .prop_map(|(x, y, z)| T3::from_scale(V3::new(x, y, z)))
    }

    proptest! {
        #[test]
        fn identity_transform_is_noop_vec(v in arb_vec3()) {
            let t = T3::new();
            let result = t.to_world(v);
            let diff = (result - v).norm();
            prop_assert!(diff < 1e-6, "identity transform moved vec by {}", diff);
        }

        #[test]
        fn identity_transform_is_noop_point(p in arb_point3()) {
            let t = T3::new();
            let result = t.to_world(p);
            let diff = (result - p).norm();
            prop_assert!(diff < 1e-6, "identity transform moved point by {}", diff);
        }

        #[test]
        fn rotation_preserves_vec_length(r in arb_rotation(), v in arb_vec3()) {
            let rotated = r.to_world(v);
            let orig_norm = v.norm();
            let new_norm = rotated.norm();
            prop_assert!(
                (orig_norm - new_norm).abs() < 1e-3,
                "rotation changed norm: {} -> {}", orig_norm, new_norm
            );
        }

        #[test]
        fn roundtrip_vec_rotation(r in arb_rotation(), v in arb_vec3()) {
            let roundtrip = r.to_local(r.to_world(v));
            let diff = (roundtrip - v).norm();
            prop_assert!(diff < 1e-2, "rotation roundtrip error={}", diff);
        }

        #[test]
        fn roundtrip_point_translation(t in arb_translation(), p in arb_point3()) {
            let roundtrip = t.to_local(t.to_world(p));
            let diff = (roundtrip - p).norm();
            prop_assert!(diff < 1e-2, "translation roundtrip error={}", diff);
        }

        #[test]
        fn roundtrip_vec_scale(s in arb_nonzero_scale(), v in arb_vec3()) {
            let roundtrip = s.to_local(s.to_world(v));
            let diff = (roundtrip - v).norm();
            prop_assert!(diff < 1e-2, "scale roundtrip error={}", diff);
        }

        #[test]
        fn inverse_inverse_is_identity(r in arb_rotation(), v in arb_vec3()) {
            let inv_inv = r.inverse().inverse();
            let a = r.to_world(v);
            let b = inv_inv.to_world(v);
            let diff = (a - b).norm();
            prop_assert!(diff < 1e-2, "double inverse error={}", diff);
        }

        #[test]
        fn translation_does_not_affect_vectors(t in arb_translation(), v in arb_vec3()) {
            let result = t.to_world(v);
            let diff = (result - v).norm();
            prop_assert!(diff < 1e-4, "translation moved vector by {}", diff);
        }

        #[test]
        fn from_stack_matches_manual_composition(
            s in arb_nonzero_scale(),
            r in arb_rotation(),
            t in arb_translation(),
            v in arb_vec3()
        ) {
            let stacked = T3::from_stack(Some(s), Some(r), Some(t));
            let manual = t * r * s;
            let a = stacked.to_world(v);
            let b = manual.to_world(v);
            let diff = (a - b).norm();
            prop_assert!(diff < 1e-2, "stack vs manual diff={}", diff);
        }
    }

    #[test]
    fn test_transform() {
        let transform_translate = T3::from_translation(V3::new(1.0, 2.0, 0.0));
        let transform_rotate = T3::from_axis_angle(V3::z_axis(), PI / 4.0);
        let transform_scale = T3::from_scale(V3::new(2.0, 2.0, 2.0));

        let test_vec = V3::new(1.0, 1.0, 1.0);

        let translated_vec = transform_translate.to_world(test_vec);
        assert!(
            (translated_vec - test_vec).norm() < 1e-6,
            "translation should not affect vectors"
        );

        let rotated_vec = transform_rotate.to_world(test_vec);
        crate::assert_approx_eq(rotated_vec.norm(), test_vec.norm(), 1e-5);

        let scaled_vec = transform_scale.to_world(test_vec);
        crate::assert_approx_eq(scaled_vec.norm(), test_vec.norm() * 2.0, 1e-5);

        let test_point = P3::origin() + test_vec;

        let translated_point = transform_translate.to_world(test_point);
        crate::assert_approx_eq(translated_point.x(), test_point.x() + 1.0, 1e-5);
        crate::assert_approx_eq(translated_point.y(), test_point.y() + 2.0, 1e-5);

        let rotated_point = transform_rotate.to_world(test_point);
        crate::assert_approx_eq(
            (rotated_point - P3::origin()).norm(),
            (test_point - P3::origin()).norm(),
            1e-5,
        );
    }

    #[test]
    fn test_round_trip_error() {
        let transform_translate = T3::from_translation(V3::new(1.0, 2.0, 0.0));
        let transform_rotate = T3::from_axis_angle(V3::z_axis(), PI / 4.0);
        let transform_scale = T3::from_scale(V3::new(2.0, 3.0, 4.0));

        let trs = transform_translate * transform_rotate * transform_scale;
        let trs2 = T3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );
        let rs = transform_rotate * transform_scale;
        let tr = transform_translate * transform_rotate;
        let ts = transform_translate * transform_scale;

        let test_vec = V3::new(1.0, 1.0, 0.0).normalized();

        let eval_round_trip_error_vec = |transform: T3, input: V3| {
            (transform.to_local(transform.to_world(input)) - input).norm()
        };
        let eval_round_trip_error_point = |transform: T3, input: P3| {
            (transform.to_local(transform.to_world(input)) - input).norm()
        };

        let tolerance = 1e-5;
        for (name, transform) in [
            ("trs", trs),
            ("trs2", trs2),
            ("rs", rs),
            ("tr", tr),
            ("ts", ts),
        ] {
            let err = eval_round_trip_error_vec(transform, test_vec);
            assert!(err < tolerance, "vec round-trip error for {} = {}", name, err);
        }

        let test_point = P3::origin() + test_vec;
        for (name, transform) in [
            ("trs", trs),
            ("trs2", trs2),
            ("rs", rs),
            ("tr", tr),
            ("ts", ts),
        ] {
            let err = eval_round_trip_error_point(transform, test_point);
            assert!(
                err < tolerance,
                "point round-trip error for {} = {}",
                name,
                err
            );
        }
    }

    #[test]
    fn test_transform_combination() {
        let transform_translate = T3::from_translation(V3::new(1.0, 1.0, 0.0));
        let transform_rotate = T3::from_axis_angle(V3::z_axis(), PI / 4.0);
        let transform_scale = T3::from_scale(V3::new(2.0, 3.0, 4.0));

        let combination_trs = transform_translate * transform_rotate * transform_scale;
        let combination_trs_2 = T3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );
        let Transform3 { forward, reverse: _ } = combination_trs_2.clone();
        let redone = T3::new_from_matrix(forward).unwrap();

        let test_vec = V3::new(1.0, 1.0, 0.0).normalized();

        let v1 = combination_trs.to_world(test_vec);
        let v2 = combination_trs_2.to_world(test_vec);
        assert!(
            (v1 - v2).norm() < 1e-5,
            "from_stack vs mul mismatch for vec: {:?} vs {:?}",
            v1,
            v2
        );

        let v3 = redone.to_world(test_vec);
        assert!(
            (v1 - v3).norm() < 1e-5,
            "redone vs original mismatch for vec: {:?} vs {:?}",
            v1,
            v3
        );

        let test_point = P3::origin() + test_vec;

        let p1 = combination_trs.to_world(test_point);
        let p2 = combination_trs_2.to_world(test_point);
        assert!(
            (p1 - p2).norm() < 1e-5,
            "from_stack vs mul mismatch for point: {:?} vs {:?}",
            p1,
            p2
        );

        let p3 = redone.to_world(test_point);
        assert!(
            (p1 - p3).norm() < 1e-5,
            "redone vs original mismatch for point: {:?} vs {:?}",
            p1,
            p3
        );
    }

    #[test]
    fn test_reverse_transform_combination() {
        let transform_translate = T3::from_translation(V3::new(1.0, 1.0, 0.0));
        let transform_rotate = T3::from_axis_angle(V3::z_axis(), PI / 4.0);
        let transform_scale = T3::from_scale(V3::new(2.0, 3.0, 4.0));

        let combination_trs = transform_translate * transform_rotate * transform_scale;
        let combination_trs_2 = T3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );

        let test_vec = V3::new(1.0, 1.0, 0.0).normalized();

        let v1 = combination_trs.to_local(test_vec);
        let v2 = combination_trs_2.to_local(test_vec);
        assert!(
            (v1 - v2).norm() < 1e-5,
            "to_local mismatch for vec: {:?} vs {:?}",
            v1,
            v2
        );

        let test_point = P3::origin() + test_vec;

        let p1 = combination_trs.to_local(test_point);
        let p2 = combination_trs_2.to_local(test_point);
        assert!(
            (p1 - p2).norm() < 1e-5,
            "to_local mismatch for point: {:?} vs {:?}",
            p1,
            p2
        );
    }

    #[test]
    fn test_translate() {
        // Column-major flat layout: identity basis in columns 0..2, translation
        // (1, 2, 3) in column 3.
        #[rustfmt::skip]
        let matrix: M4 = Matrix4x4::from_array([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 2.0, 3.0, 1.0,
        ]);
        let simd_vec = V3::new(1.0, 2.0, 3.0);
        let simd_point = P3::new(1.0, 2.0, 3.0);

        let transform = T3::from_translation(V3::new(1.0, 2.0, 3.0));

        let result_vec = transform.to_world(simd_vec);
        assert!(
            (result_vec - simd_vec).norm() < 1e-6,
            "translation affected vector"
        );

        let mat_vec = matrix * simd_vec;
        assert!(
            (mat_vec - simd_vec).norm() < 1e-6,
            "matrix translation affected vector"
        );

        let result_point = transform.to_world(simd_point);
        crate::assert_approx_eq(result_point.x(), 2.0, 1e-5);
        crate::assert_approx_eq(result_point.y(), 4.0, 1e-5);
        crate::assert_approx_eq(result_point.z(), 6.0, 1e-5);

        let mat_point = matrix * simd_point;
        assert!(
            (result_point - mat_point).norm() < 1e-6,
            "transform vs matrix mismatch for point"
        );

        let round_trip = transform.to_local(result_point);
        assert!(
            (round_trip - simd_point).norm() < 1e-5,
            "point round-trip failed"
        );
    }
}
