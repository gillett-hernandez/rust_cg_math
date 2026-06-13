use crate::prelude::*;
use thermite::simd::Simd;
use thermite::register::LinAlg3Register;

// also known as an orthonormal basis.
pub struct TangentFrame<S: Simd> {
    pub tangent: Vec3<S>,
    pub bitangent: Vec3<S>,
    pub normal: Vec3<S>,
}

impl<S: Simd> Copy for TangentFrame<S> {}
impl<S: Simd> Clone for TangentFrame<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> std::fmt::Debug for TangentFrame<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TangentFrame")
            .field("tangent", &self.tangent)
            .field("bitangent", &self.bitangent)
            .field("normal", &self.normal)
            .finish()
    }
}

impl<S: Simd> TangentFrame<S>
where
    S::f32x4: LinAlg3Register,
{
    #[inline(always)]
    pub fn new(tangent: Vec3<S>, bitangent: Vec3<S>, normal: Vec3<S>) -> Self {
        debug_assert!(
            (tangent * bitangent).abs() < 0.000001,
            "tbit:{:?} * {:?} was != 0",
            tangent,
            bitangent
        );
        debug_assert!(
            (tangent * normal).abs() < 0.000001,
            "tn: {:?} * {:?} was != 0",
            tangent,
            normal
        );
        debug_assert!(
            (bitangent * normal).abs() < 0.000001,
            "bitn:{:?} * {:?} was != 0",
            bitangent,
            normal
        );
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: bitangent.normalized(),
            normal: normal.normalized(),
        }
    }
    #[inline(always)]
    pub fn from_tangent_and_normal(tangent: Vec3<S>, normal: Vec3<S>) -> Self {
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: tangent.normalized().cross(normal.normalized()).normalized(),
            normal: normal.normalized(),
        }
    }

    #[inline(always)]
    pub fn from_normal(normal: Vec3<S>) -> Self {
        let [x, y, z, _]: [f32; 4] = normal.as_array();
        let sign = (1.0 as f32).copysign(z);
        let a = -1.0 / (sign + z);
        let b = x * y * a;
        TangentFrame {
            tangent: Vec3::new(1.0 + sign * x * x * a, sign * b, -sign * x),
            bitangent: Vec3::new(b, sign + y * y * a, -y),
            normal,
        }
    }

    #[inline(always)]
    pub fn to_world(&self, v: &Vec3<S>) -> Vec3<S> {
        self.tangent * v.x() + self.bitangent * v.y() + self.normal * v.z()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec3<S>) -> Vec3<S> {
        Vec3::new(
            self.tangent * (*v),
            self.bitangent * (*v),
            self.normal * (*v),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;
    type TF = TangentFrame<TestS>;

    fn arb_unit_vec3() -> impl Strategy<Value = V3> {
        (-1.0f32..1.0, -1.0f32..1.0, -1.0f32..1.0)
            .prop_filter("nonzero", |(x, y, z)| x * x + y * y + z * z > 1e-4)
            .prop_map(|(x, y, z)| V3::new(x, y, z).normalized())
    }

    fn arb_vec3() -> impl Strategy<Value = V3> {
        (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0).prop_map(|(x, y, z)| V3::new(x, y, z))
    }

    #[test]
    fn test_debug_and_clone() {
        let frame = TF::new(V3::x_axis(), V3::y_axis(), V3::z_axis());
        let s = format!("{:?}", frame);
        assert!(s.contains("TangentFrame"));
        let c = frame.clone();
        assert_eq!((c.tangent - frame.tangent).norm(), 0.0);
        assert_eq!((c.normal - frame.normal).norm(), 0.0);
    }

    #[test]
    fn test_new_normalizes_axis_basis() {
        // feed a (scaled) orthogonal basis; new() normalizes each vector.
        let frame = TF::new(
            V3::new(2.0, 0.0, 0.0),
            V3::new(0.0, 3.0, 0.0),
            V3::new(0.0, 0.0, 4.0),
        );
        assert!((frame.tangent.norm() - 1.0).abs() < 1e-6);
        assert!((frame.bitangent.norm() - 1.0).abs() < 1e-6);
        assert!((frame.normal.norm() - 1.0).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn from_tangent_and_normal_is_orthonormal(n in arb_unit_vec3()) {
            // pick a tangent that isn't parallel to the normal
            let seed = if n.x().abs() < 0.9 { V3::x_axis() } else { V3::y_axis() };
            let tangent = (seed - n * (seed * n)).normalized();
            let frame = TF::from_tangent_and_normal(tangent, n);
            prop_assert!((frame.tangent.norm() - 1.0).abs() < 1e-4);
            prop_assert!((frame.bitangent.norm() - 1.0).abs() < 1e-4);
            prop_assert!((frame.normal.norm() - 1.0).abs() < 1e-4);
            prop_assert!((frame.bitangent * frame.normal).abs() < 1e-3, "bitangent not ⊥ normal");
        }

        #[test]
        fn from_normal_produces_orthonormal_basis(n in arb_unit_vec3()) {
            let frame = TF::from_normal(n);

            let t_norm = frame.tangent.norm();
            let b_norm = frame.bitangent.norm();
            let n_norm = frame.normal.norm();
            prop_assert!((t_norm - 1.0).abs() < 1e-4, "tangent norm={}", t_norm);
            prop_assert!((b_norm - 1.0).abs() < 1e-4, "bitangent norm={}", b_norm);
            prop_assert!((n_norm - 1.0).abs() < 1e-4, "normal norm={}", n_norm);

            let tb = (frame.tangent * frame.bitangent).abs();
            let tn = (frame.tangent * frame.normal).abs();
            let bn = (frame.bitangent * frame.normal).abs();
            prop_assert!(tb < 1e-4, "t.b={}", tb);
            prop_assert!(tn < 1e-4, "t.n={}", tn);
            prop_assert!(bn < 1e-4, "b.n={}", bn);
        }

        #[test]
        fn to_world_to_local_roundtrip(n in arb_unit_vec3(), v in arb_vec3()) {
            let frame = TF::from_normal(n);
            let world = frame.to_world(&v);
            let back = frame.to_local(&world);
            let diff = (back - v).norm();
            prop_assert!(diff < 1e-2, "roundtrip error={}", diff);
        }

        #[test]
        fn to_local_to_world_roundtrip(n in arb_unit_vec3(), v in arb_vec3()) {
            let frame = TF::from_normal(n);
            let local = frame.to_local(&v);
            let back = frame.to_world(&local);
            let diff = (back - v).norm();
            prop_assert!(diff < 1e-2, "roundtrip error={}", diff);
        }

        #[test]
        fn normal_maps_to_z_in_local(n in arb_unit_vec3()) {
            let frame = TF::from_normal(n);
            let local_n = frame.to_local(&n);
            let expected = V3::z_axis();
            let diff = (local_n - expected).norm();
            prop_assert!(diff < 1e-3, "normal in local={:?}, expected Z", local_n);
        }

        #[test]
        fn z_maps_to_normal_in_world(n in arb_unit_vec3()) {
            let frame = TF::from_normal(n);
            let world_z = frame.to_world(&V3::z_axis());
            let diff = (world_z - n).norm();
            prop_assert!(diff < 1e-3, "Z in world={:?}, expected {:?}", world_z, n);
        }
    }
}
