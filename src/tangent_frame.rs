use crate::prelude::*;

// also known as an orthonormal basis.
#[derive(Copy, Clone, Debug)]
pub struct TangentFrame {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
}

impl TangentFrame {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
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
    pub fn from_tangent_and_normal(tangent: Vec3, normal: Vec3) -> Self {
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: tangent.normalized().cross(normal.normalized()).normalized(),
            normal: normal.normalized(),
        }
    }

    pub fn from_normal(normal: Vec3) -> Self {
        // let n2 = Vec3::from_raw(normal.0 * normal.0);
        // let (x, y, z) = (normal.x(), normal.y(), normal.z());
        let [x, y, z, _]: [f32; 4] = normal.0.into();
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
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.tangent * v.x() + self.bitangent * v.y() + self.normal * v.z()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
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

    fn arb_unit_vec3() -> impl Strategy<Value = Vec3> {
        (-1.0f32..1.0, -1.0f32..1.0, -1.0f32..1.0)
            .prop_filter("nonzero", |(x, y, z)| x * x + y * y + z * z > 1e-4)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z).normalized())
    }

    fn arb_vec3() -> impl Strategy<Value = Vec3> {
        (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    proptest! {
        #[test]
        fn from_normal_produces_orthonormal_basis(n in arb_unit_vec3()) {
            let frame = TangentFrame::from_normal(n);

            // all unit length
            let t_norm = frame.tangent.norm();
            let b_norm = frame.bitangent.norm();
            let n_norm = frame.normal.norm();
            prop_assert!((t_norm - 1.0).abs() < 1e-4, "tangent norm={}", t_norm);
            prop_assert!((b_norm - 1.0).abs() < 1e-4, "bitangent norm={}", b_norm);
            prop_assert!((n_norm - 1.0).abs() < 1e-4, "normal norm={}", n_norm);

            // mutually orthogonal
            let tb = (frame.tangent * frame.bitangent).abs();
            let tn = (frame.tangent * frame.normal).abs();
            let bn = (frame.bitangent * frame.normal).abs();
            prop_assert!(tb < 1e-4, "t.b={}", tb);
            prop_assert!(tn < 1e-4, "t.n={}", tn);
            prop_assert!(bn < 1e-4, "b.n={}", bn);
        }

        #[test]
        fn to_world_to_local_roundtrip(n in arb_unit_vec3(), v in arb_vec3()) {
            let frame = TangentFrame::from_normal(n);
            let world = frame.to_world(&v);
            let back = frame.to_local(&world);
            let diff = (back - v).norm();
            prop_assert!(diff < 1e-2, "roundtrip error={}", diff);
        }

        #[test]
        fn to_local_to_world_roundtrip(n in arb_unit_vec3(), v in arb_vec3()) {
            let frame = TangentFrame::from_normal(n);
            let local = frame.to_local(&v);
            let back = frame.to_world(&local);
            let diff = (back - v).norm();
            prop_assert!(diff < 1e-2, "roundtrip error={}", diff);
        }

        #[test]
        fn normal_maps_to_z_in_local(n in arb_unit_vec3()) {
            let frame = TangentFrame::from_normal(n);
            let local_n = frame.to_local(&n);
            let expected = Vec3::Z;
            let diff = (local_n - expected).norm();
            prop_assert!(diff < 1e-3, "normal in local={:?}, expected Z", local_n);
        }

        #[test]
        fn z_maps_to_normal_in_world(n in arb_unit_vec3()) {
            let frame = TangentFrame::from_normal(n);
            let world_z = frame.to_world(&Vec3::Z);
            let diff = (world_z - n).norm();
            prop_assert!(diff < 1e-3, "Z in world={:?}, expected {:?}", world_z, n);
        }
    }
}
