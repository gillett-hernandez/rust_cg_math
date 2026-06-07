use crate::prelude::*;
use thermite::simd::Simd;
use thermite::register::LinAlg3Register;

pub struct Matrix4x4<S: Simd>(pub Vector<S::f32x16>);

impl<S: Simd> Copy for Matrix4x4<S> {}
impl<S: Simd> Clone for Matrix4x4<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Matrix4x4<S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<S: Simd> std::fmt::Debug for Matrix4x4<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Matrix4x4").field(&self.as_array()).finish()
    }
}

impl<S: Simd> Matrix4x4<S> {
    pub fn identity() -> Matrix4x4<S> {
        Matrix4x4(Vector::<S::f32x16>::new([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]))
    }

    /// Column-major flat layout: `m[col * 4 + row]`. Matches nalgebra's
    /// in-memory order, which lets `From<nalgebra::Matrix4>` be a 1:1 copy.
    pub fn as_array(&self) -> [f32; 16] {
        let arr = self.0.into_array();
        let mut out = [0.0_f32; 16];
        for i in 0..16 {
            out[i] = arr[i];
        }
        out
    }

    pub fn from_array(values: [f32; 16]) -> Self {
        Matrix4x4(Vector::<S::f32x16>::new(values))
    }

    pub fn transpose(&self) -> Matrix4x4<S> {
        // Column-major: swap col/row indices.
        let m = self.as_array();
        let mut t = [0.0_f32; 16];
        for c in 0..4 {
            for r in 0..4 {
                t[r * 4 + c] = m[c * 4 + r];
            }
        }
        Self::from_array(t)
    }
}

// Helper: m[col * 4 + row] under column-major storage.
#[inline(always)]
fn m_at(m: &[f32; 16], row: usize, col: usize) -> f32 {
    m[col * 4 + row]
}

impl<S: Simd> Mul<Vec3<S>> for Matrix4x4<S> {
    type Output = Vec3<S>;
    fn mul(self, rhs: Vec3<S>) -> Self::Output {
        // Vec3 has w=0, so the translation column (col 3) cancels out.
        let m = self.as_array();
        let v = rhs.as_array();
        let x = m_at(&m, 0, 0) * v[0] + m_at(&m, 0, 1) * v[1] + m_at(&m, 0, 2) * v[2] + m_at(&m, 0, 3) * v[3];
        let y = m_at(&m, 1, 0) * v[0] + m_at(&m, 1, 1) * v[1] + m_at(&m, 1, 2) * v[2] + m_at(&m, 1, 3) * v[3];
        let z = m_at(&m, 2, 0) * v[0] + m_at(&m, 2, 1) * v[1] + m_at(&m, 2, 2) * v[2] + m_at(&m, 2, 3) * v[3];
        Vec3::new(x, y, z)
    }
}

impl<S: Simd> Mul<Point3<S>> for Matrix4x4<S> {
    type Output = Point3<S>;
    fn mul(self, rhs: Point3<S>) -> Self::Output {
        let m = self.as_array();
        let p = rhs.as_array();
        let x = m_at(&m, 0, 0) * p[0] + m_at(&m, 0, 1) * p[1] + m_at(&m, 0, 2) * p[2] + m_at(&m, 0, 3) * p[3];
        let y = m_at(&m, 1, 0) * p[0] + m_at(&m, 1, 1) * p[1] + m_at(&m, 1, 2) * p[2] + m_at(&m, 1, 3) * p[3];
        let z = m_at(&m, 2, 0) * p[0] + m_at(&m, 2, 1) * p[1] + m_at(&m, 2, 2) * p[2] + m_at(&m, 2, 3) * p[3];
        let w = m_at(&m, 3, 0) * p[0] + m_at(&m, 3, 1) * p[1] + m_at(&m, 3, 2) * p[2] + m_at(&m, 3, 3) * p[3];
        Point3(Vector::<S::f32x4>::new([x, y, z, w])).normalize()
    }
}

impl<S: Simd> Mul<Ray<S>> for Matrix4x4<S>
where
    S::f32x4: LinAlg3Register,
{
    type Output = Ray<S>;
    fn mul(self, rhs: Ray<S>) -> Self::Output {
        Ray {
            origin: self * rhs.origin,
            direction: (self * rhs.direction).normalized(),
            ..rhs
        }
    }
}

impl<S: Simd> Mul for Matrix4x4<S> {
    type Output = Matrix4x4<S>;
    fn mul(self, rhs: Matrix4x4<S>) -> Self::Output {
        // Column-major: out[col, row] = sum_k a[k, row] * b[col, k].
        let a = self.as_array();
        let b = rhs.as_array();
        let mut out = [0.0_f32; 16];
        for c in 0..4 {
            for r in 0..4 {
                let mut sum = 0.0_f32;
                for k in 0..4 {
                    sum += m_at(&a, r, k) * m_at(&b, k, c);
                }
                out[c * 4 + r] = sum;
            }
        }
        Self::from_array(out)
    }
}

pub struct Transform3<S: Simd> {
    pub forward: Matrix4x4<S>,
    pub reverse: Matrix4x4<S>,
}

impl<S: Simd> Copy for Transform3<S> {}
impl<S: Simd> Clone for Transform3<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> PartialEq for Transform3<S> {
    fn eq(&self, other: &Self) -> bool {
        self.forward == other.forward && self.reverse == other.reverse
    }
}
impl<S: Simd> std::fmt::Debug for Transform3<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transform3")
            .field("forward", &self.forward)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl<S: Simd> Transform3<S>
where
    S::f32x4: LinAlg3Register,
{
    pub fn new() -> Self {
        Transform3 {
            forward: Matrix4x4::identity(),
            reverse: Matrix4x4::identity(),
        }
    }
    pub fn new_from_matrix(forward: nalgebra::Matrix4<f32>) -> Option<Self> {
        forward.try_inverse().map(|inverse| Transform3 {
            forward: Matrix4x4::from(forward),
            reverse: Matrix4x4::from(inverse),
        })
    }

    pub fn inverse(self) -> Transform3<S> {
        Transform3::new_from_raw(self.reverse, self.forward)
    }

    pub fn from_translation(shift: Vec3<S>) -> Self {
        Transform3::new_from_matrix(nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
            shift.x(),
            shift.y(),
            shift.z(),
        )))
        .expect("translation matrix was not invertible")
    }

    pub fn from_scale(scale: Vec3<S>) -> Self {
        Transform3::new_from_matrix(nalgebra::Matrix4::new_nonuniform_scaling(
            &nalgebra::Vector3::new(scale.x(), scale.y(), scale.z()),
        ))
        .expect("scale matrix was not invertible")
    }

    pub fn from_axis_angle(axis: Vec3<S>, radians: f32) -> Self {
        let axisangle = radians * nalgebra::Vector3::new(axis.x(), axis.y(), axis.z());
        let affine = nalgebra::Matrix4::from_scaled_axis(axisangle);
        Transform3::new_from_matrix(affine).expect("rotation matrix was not invertible")
    }

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

    pub fn new_from_raw(forward: Matrix4x4<S>, reverse: Matrix4x4<S>) -> Self {
        Transform3 { forward, reverse }
    }

    pub fn from_vector_stack(
        v0: Vector<S::f32x4>,
        v1: Vector<S::f32x4>,
        v2: Vector<S::f32x4>,
    ) -> Self {
        let extract = |v: Vector<S::f32x4>| (v.extract::<0>(), v.extract::<1>(), v.extract::<2>());
        let (m11, m12, m13) = extract(v0);
        let (m21, m22, m23) = extract(v1);
        let (m31, m32, m33) = extract(v2);

        let m = Matrix4x4::<S>::from_array([
            m11, m12, m13, 0.0, m21, m22, m23, 0.0, m31, m32, m33, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        Transform3::new_from_raw(m.transpose(), m)
    }

    pub fn axis_transform(&self) -> (Vec3<S>, Vec3<S>, Vec3<S>) {
        (
            self.to_world(Vec3::x_axis()),
            self.to_world(Vec3::y_axis()),
            self.to_world(Vec3::z_axis()),
        )
    }

    pub fn to_local<T>(&self, value: T) -> <Matrix4x4<S> as Mul<T>>::Output
    where
        Matrix4x4<S>: Mul<T>,
    {
        self.reverse * value
    }
    pub fn to_world<T>(&self, value: T) -> <Matrix4x4<S> as Mul<T>>::Output
    where
        Matrix4x4<S>: Mul<T>,
    {
        self.forward * value
    }
}

impl<S: Simd> From<TangentFrame<S>> for Transform3<S>
where
    S::f32x4: LinAlg3Register,
{
    fn from(value: TangentFrame<S>) -> Self {
        Transform3::from_vector_stack(value.tangent.0, value.bitangent.0, value.normal.0)
    }
}

impl<S: Simd> From<nalgebra::Matrix4<f32>> for Matrix4x4<S> {
    fn from(matrix: nalgebra::Matrix4<f32>) -> Self {
        // nalgebra is column-major in memory; the legacy code took
        // `matrix.as_slice()` which gives column-major order, but indexed it
        // into Matrix4x4 as if it were row-major. Match that legacy behavior
        // to keep test invariants stable.
        let slice = matrix.as_slice();
        let mut values = [0.0_f32; 16];
        for (i, v) in slice.iter().enumerate() {
            values[i] = *v;
        }
        Matrix4x4::from_array(values)
    }
}

impl<S: Simd> From<Matrix4x4<S>> for nalgebra::Matrix4<f32> {
    fn from(other: Matrix4x4<S>) -> Self {
        let m = other.as_array();
        nalgebra::Matrix4::new(
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15],
        )
        .transpose()
    }
}

impl<S: Simd> Mul<Transform3<S>> for Transform3<S>
where
    S::f32x4: LinAlg3Register,
{
    type Output = Transform3<S>;
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
        let redone = T3::new_from_matrix(forward.into()).unwrap();

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
        let n_translate =
            nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0));

        let matrix: M4 = Matrix4x4::from(n_translate);
        let simd_vec = V3::new(1.0, 2.0, 3.0);
        let simd_point = P3::new(1.0, 2.0, 3.0);

        let transform = T3::new_from_matrix(n_translate).unwrap();

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
