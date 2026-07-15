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

    /// General 4x4 inverse via thermite's `LinAlg4Vector::mat4_inverse_inplace`
    /// (the vectorized GLM shuffle form). Returns `None` if the matrix is
    /// singular (`|det| < f32::EPSILON`). Runs at construction time (once per
    /// transform).
    ///
    /// This previously used a hand-rolled scalar cofactor/adjugate routine
    /// because thermite's shuffle-based inverse wasn't reachable in fast form
    /// (no immediate `ShuffleRegister` path on the scalar backend, and the
    /// portable `swizzle!` lowered to `permutevar`+`blend` on x86). Thermite's
    /// LinAlg rework added that primitive across all backends, so we defer to
    /// it. `mat4_inverse_inplace` hands back the determinant; we keep our own
    /// epsilon tolerance here rather than thermite's exact-zero test (its
    /// `mat4_inverse` only rejects an exactly-zero determinant and returns a
    /// finite-but-unreliable result for ill-conditioned matrices).
    /// See `benches/math_benches.rs::mat4_try_inverse`.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<Matrix4x4<S>> {
        let mut m = self.0;
        let det = <Vector<S::f32x4> as LinAlg4Vector>::mat4_inverse_inplace(&mut m);
        if det.abs() < f32::EPSILON {
            return None;
        }
        Some(Matrix4x4(m))
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
        //
        // `zero4` is load-bearing, not hygiene. mat4_vec3_product computes
        // `m[0]*x + m[1]*y + m[2]*z` over the *full 4-lane* columns; it only skips column 3.
        // Lane 4 of the result is therefore zero only when the matrix's bottom row is
        // [0,0,0,1]. The inverse-transpose used to transform normals
        // (`transform.reverse.transpose() * normal`) moves the translation *into* that row, so
        // the product picks it up as `w = t·v` and `normalized()` — which divides by the 3D
        // norm — rescales it rather than clearing it. That leaked a w onto every transformed
        // normal (a camera at look_from=[-5,0,0] gave lens_normal.w = 5.0), which then rode
        // into `Point3 + Vec3` offsets and produced points with w != 1.
        Vec3(rhs.0.mat4_vec3_product::<true>(&self.0).zero4())
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

    /// Standard SRT stack: scale is applied first (about the local origin),
    /// then rotation, then translation — `to_world = T·R·S`.
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
    /// Standard composition: `(a * b).to_world(x) == a.to_world(b.to_world(x))`
    /// (rhs applied first), matching Matrix4x4 multiplication. The inverse of
    /// `A·B` is `B⁻¹·A⁻¹`, hence the swapped reverse product.
    #[inline(always)]
    fn mul(self, rhs: Transform3<S>) -> Self::Output {
        Transform3::new_from_raw(self.forward * rhs.forward, rhs.reverse * self.reverse)
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

    /// A transformed `Vec3` must keep w == 0, *including* through the inverse-transpose used to
    /// transform normals.
    ///
    /// Regression: `mat4_vec3_product` computes `m[0]*x + m[1]*y + m[2]*z` over the full 4-lane
    /// columns, skipping only column 3. Lane 4 of the result is zero only when the matrix's
    /// bottom row is [0,0,0,1] — and `reverse.transpose()` moves the translation *into* that row,
    /// so the product picked it up as `w = t·v`. `normalized()` divides by the 3D norm, which
    /// rescaled the bogus w rather than clearing it.
    ///
    /// Measured before the fix: translation (5,0,0) with normal (1,0,0) gave w = 5.0 (the
    /// rust_pathtracer camera at look_from=[-5,0,0]). That rode into `point + normal * offset`,
    /// producing points with w = 1.005, which tripped a debug_assert in sphere intersection —
    /// and once that was papered over with `Point3::normalize`, it silently relocated every
    /// light-tracing shadow-ray origin, so `veach_v` reported occluded on all 1.4M connections
    /// and light tracing rendered pure black.
    #[test]
    fn transformed_normal_keeps_w_zero_through_inverse_transpose() {
        // the exact configuration from the light-tracing failure
        let transform = T3::from_translation(V3::new(-5.0, 0.0, 0.0));
        let normal = V3::new(1.0, 0.0, 0.0);

        let transformed = (transform.reverse.transpose() * normal).normalized();
        let w = transformed.0.extract::<3>();
        assert!(
            w.abs() < 1e-6,
            "inverse-transpose of a translated frame leaked w = {w} onto a normal; a Vec3 must \
             always have w = 0 or it corrupts every `Point3 + Vec3` offset downstream"
        );

        let forward_w = transform.to_world(normal).0.extract::<3>();
        assert!(
            forward_w.abs() < 1e-6,
            "forward transform leaked w = {forward_w} onto a normal"
        );
    }

    proptest! {
        /// The w=0 invariant must survive an arbitrary rotate+translate frame, in both
        /// directions and through both transposes.
        #[test]
        fn transformed_vec3_always_has_w_zero(
            v in arb_unit_vec3(),
            shift in arb_vec3(),
            axis in arb_unit_vec3(),
            angle in -PI..PI,
        ) {
            let transform = T3::from_axis_angle(axis, angle) * T3::from_translation(shift);
            for out in [
                transform.to_world(v),
                transform.to_local(v),
                (transform.reverse.transpose() * v).normalized(),
                (transform.forward.transpose() * v).normalized(),
            ] {
                let w = out.0.extract::<3>();
                prop_assert!(w.abs() < 1e-5, "transformed Vec3 gained w = {}", w);
            }
        }
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

    // Pins the composition ORDER, which the round-trip/comparison tests above
    // cannot see (they compare `from_stack` against the same `t*r*s` product, and
    // Vec3 probes are translation-blind). `Mul` follows the standard convention
    // (a*b)(x) = a(b(x)), so `t*r*s` and from_stack(s,r,t) both mean: scale
    // first, then rotate, then translate (SRT). Point3 probe catches a flipped
    // Mul: the pre-fix backwards composition gave (0,2,20) here.
    #[test]
    fn from_stack_applies_scale_rotate_translate_in_order() {
        let stacked = T3::from_stack(
            Some(T3::from_scale(V3::new(2.0, 2.0, 2.0))),
            Some(T3::from_axis_angle(V3::z_axis(), PI / 2.0)),
            Some(T3::from_translation(V3::new(0.0, 0.0, 10.0))),
        );
        // (1,0,0) --scale 2--> (2,0,0) --rotate z90--> (0,2,0) --translate--> (0,2,10)
        let p = stacked.to_world(P3::new(1.0, 0.0, 0.0));
        assert!(
            (p - P3::new(0.0, 2.0, 10.0)).norm() < 1e-4,
            "from_stack must be SRT: expected (0,2,10), got {:?}",
            p
        );
        // origin must land exactly at the translation (scale/rotation must not touch it)
        let c = stacked.to_world(P3::origin());
        assert!(
            (c - P3::new(0.0, 0.0, 10.0)).norm() < 1e-4,
            "origin must map to the translation: got {:?}",
            c
        );
        // standard Mul convention directly: (t * s)(origin) = t(s(origin)) = (0,0,10)
        let ts = T3::from_translation(V3::new(0.0, 0.0, 10.0)) * T3::from_scale(V3::new(2.0, 2.0, 2.0));
        let q = ts.to_world(P3::origin());
        assert!(
            (q - P3::new(0.0, 0.0, 10.0)).norm() < 1e-4,
            "(a*b)(x) must equal a(b(x)): got {:?}",
            q
        );
    }

    #[test]
    fn matrix_debug_clone_eq() {
        let m = M4::identity();
        let s = format!("{:?}", m);
        assert!(s.contains("Matrix4x4"));
        let c = m.clone();
        assert_eq!(c, m); // PartialEq
        assert_ne!(m, M4::from_array([2.0; 16]));
    }

    #[test]
    fn matrix_transpose_involution() {
        let m = M4::from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        assert_eq!(m.transpose().transpose(), m);
        // transpose swaps off-diagonal column/row entries
        let mt = m.transpose();
        assert_eq!(mt.as_array()[1], m.as_array()[4]);
    }

    #[test]
    fn matrix_try_inverse_singular_is_none() {
        // a zero matrix is singular -> no inverse
        assert!(M4::from_array([0.0; 16]).try_inverse().is_none());
        // a rank-deficient matrix (two identical columns) is also singular
        let mut vals = [0.0f32; 16];
        vals[0] = 1.0;
        vals[5] = 1.0;
        vals[10] = 1.0;
        // leave the 4th column all-zero -> det 0
        assert!(M4::from_array(vals).try_inverse().is_none());
    }

    /// Exploratory analysis (run with `--nocapture`) of how accurate the
    /// inverse stays as the determinant collapses toward zero, to sanity-check
    /// the `|det| < f32::EPSILON` cutoff in `try_inverse`.
    ///
    /// We build a near-singular family `M(delta)` whose 4th row is a fixed
    /// linear combination of the first three rows plus `delta` times an
    /// independent direction. As `delta -> 0` the rows become linearly
    /// dependent: `det(M) ~ delta` and the condition number grows like
    /// `1/delta`, so the inverse loses ~`log10(1/delta)` decimal digits. We
    /// measure that loss directly as the reconstruction error
    /// `max|M * M^-1 - I|`.
    ///
    /// Findings (scalar backend, see the printed table):
    /// - Reconstruction error scales like `~1e-5 / |det|` — roughly two orders
    ///   of magnitude *worse* than the naive `eps_f32 / |det|` (~1.2e-7/|det|)
    ///   you'd predict from a single rounding, because the cofactor expansion
    ///   chains ~16 FMAs plus the `1/det` division before any cancellation.
    /// - Consequently the inverse is already numerically worthless well above
    ///   f32::EPSILON: at `|det| ~= 7e-5` the error is ~6e-2, and by
    ///   `|det| ~= 7e-6` it is O(1), yet `try_inverse` still returns `Some`.
    /// - In this family `|det|` never actually lands inside `(0, EPSILON)`: it
    ///   drops from ~7.6e-6 straight to an exact `0.0` once the cofactor sum
    ///   cancels completely. So the `|det| < f32::EPSILON` gate here only ever
    ///   rejects *exactly* singular matrices — it is doing nothing to screen out
    ///   the merely ill-conditioned ones above it.
    /// - Conclusion: a *fixed* determinant tolerance can't separate "good" from
    ///   "garbage", because accuracy depends on `eps/|det|` (the condition
    ///   number), not on `|det|` alone — and the safe `|det|` threshold would
    ///   depend on the matrix's scale anyway. Raising the cutoff to, say, 1e-4
    ///   would reject the worthless inverses in this family but is arbitrary and
    ///   scale-dependent. If callers need a trustworthiness guarantee, gate on
    ///   the reconstruction error (or a condition estimate), not a larger
    ///   `|det|`. f32::EPSILON stays a reasonable "is it (near) exactly
    ///   singular" guard, which is all `try_inverse` promises.
    #[test]
    fn matrix_try_inverse_tolerance_analysis() {
        // Build a matrix from row-major rows (from_array expects column-major).
        fn from_rows(rows: [[f32; 4]; 4]) -> M4 {
            let mut cols = [0.0f32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    cols[c * 4 + r] = rows[r][c];
                }
            }
            M4::from_array(cols)
        }

        // Three fixed independent rows, plus a 4th that is their sum (-> exactly
        // singular) perturbed by `delta` along an independent direction.
        let r0 = [2.0f32, 1.0, 1.0, 1.0];
        let r1 = [1.0f32, 3.0, 1.0, 1.0];
        let r2 = [1.0f32, 1.0, 4.0, 1.0];
        let indep = [1.0f32, 1.0, 1.0, 5.0];

        // max |M * M^-1 - I| over all 16 entries.
        fn reconstruction_error(m: M4, inv: M4) -> f32 {
            let prod = (m * inv).as_array();
            let id = M4::identity().as_array();
            prod.iter()
                .zip(id.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        }

        println!(
            "\n   delta        |det|         recon_err     eps/|det|    result"
        );
        println!(
            "  ------------------------------------------------------------------"
        );
        for k in 0..11 {
            let delta = 10.0f32.powi(-k);
            let r3 = std::array::from_fn(|i| r0[i] + r1[i] + r2[i] + delta * indep[i]);
            let m = from_rows([r0, r1, r2, r3]);

            // The determinant thermite computed (mat4_det shares the inverse
            // path's cofactor form) for the table.
            let det_val = <Vector<<TestS as Simd>::f32x4> as LinAlg4Vector>::mat4_det(&m.0);

            match m.try_inverse() {
                Some(inv) => {
                    let err = reconstruction_error(m, inv);
                    println!(
                        "  {:>9.1e}   {:>11.4e}   {:>11.4e}   {:>9.2e}   Some(err={:.2e})",
                        delta,
                        det_val.abs(),
                        err,
                        f32::EPSILON / det_val.abs(),
                        err
                    );
                }
                None => {
                    println!(
                        "  {:>9.1e}   {:>11.4e}   {:>11}   {:>9.2e}   None (rejected)",
                        delta,
                        det_val.abs(),
                        "-",
                        f32::EPSILON / det_val.abs().max(f32::MIN_POSITIVE)
                    );
                }
            }
        }

        // Sanity guards that lock in the qualitative findings above:
        // 1. A comfortably-conditioned member inverts accurately.
        let delta = 1.0f32;
        let r3 = std::array::from_fn(|i| r0[i] + r1[i] + r2[i] + delta * indep[i]);
        let m = from_rows([r0, r1, r2, r3]);
        let inv = m.try_inverse().expect("delta=1 is well-conditioned");
        assert!(
            reconstruction_error(m, inv) < 1e-4,
            "well-conditioned reconstruction should be tight"
        );

        // 2. The exactly-singular limit (delta = 0) is rejected.
        let r3 = std::array::from_fn(|i| r0[i] + r1[i] + r2[i]);
        let m = from_rows([r0, r1, r2, r3]);
        assert!(
            m.try_inverse().is_none(),
            "exactly-singular matrix must be rejected"
        );
    }

    #[test]
    fn matrix_times_ray() {
        let t = T3::from_translation(V3::new(1.0, 2.0, 3.0));
        let ray = Ray::<TestS>::new(P3::origin(), V3::x_axis());
        let moved = t.forward * ray;
        // translation shifts the origin but leaves a (normalized) direction along x
        let diff = (moved.origin - P3::new(1.0, 2.0, 3.0)).norm();
        assert!(diff < 1e-5, "ray origin {:?}", moved.origin);
        assert!((moved.direction.norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn transform_debug_and_eq() {
        let t = T3::from_translation(V3::new(1.0, 0.0, 0.0));
        let s = format!("{:?}", t);
        assert!(s.contains("Transform3"));
        assert_eq!(t, t.clone());
        assert_ne!(t, T3::new());
    }

    #[test]
    fn from_vector_stack_and_axis_transform() {
        // an orthonormal frame (identity basis) -> identity-like transform
        let tf = TangentFrame::<TestS>::new(V3::x_axis(), V3::y_axis(), V3::z_axis());
        let t: T3 = tf.into(); // From<TangentFrame> -> from_vector_stack
        let (ax, ay, az) = t.axis_transform();
        assert!((ax - V3::x_axis()).norm() < 1e-5);
        assert!((ay - V3::y_axis()).norm() < 1e-5);
        assert!((az - V3::z_axis()).norm() < 1e-5);
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
