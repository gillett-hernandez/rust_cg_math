use crate::prelude::*;

use std::ops::IndexMut;
use std::simd::{f32x16, simd_swizzle};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Matrix4x4(f32x16);

impl Matrix4x4 {
    const I: Matrix4x4 = Matrix4x4(f32x16::from_array([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]));
    pub fn transpose(&self) -> Matrix4x4 {
        Matrix4x4(simd_swizzle!(
            self.0,
            [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        ))
    }
}

impl Mul<Vec3> for Matrix4x4 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        // only apply scale and rotation
        let [v0, v1, v2, v3]: [f32; 4] = rhs.0.into();

        // let column0: f32x4 = shuffle!(self.0, [0, 4, 8, 12]);
        // let column1: f32x4 = shuffle!(self.0, [1, 5, 9, 13]);
        // let column2: f32x4 = shuffle!(self.0, [2, 6, 10, 14]);
        // let column3: f32x4 = shuffle!(self.0, [3, 7, 11, 15]);

        let row1: f32x4 = simd_swizzle!(self.0, [0, 1, 2, 3]);
        let row2: f32x4 = simd_swizzle!(self.0, [4, 5, 6, 7]);
        let row3: f32x4 = simd_swizzle!(self.0, [8, 9, 10, 11]);
        let row4: f32x4 = simd_swizzle!(self.0, [12, 13, 14, 15]);

        let result = row1 * f32x4::splat(v0)
            + row2 * f32x4::splat(v1)
            + row3 * f32x4::splat(v2)
            + row4 * f32x4::splat(v3);

        result.into()
    }
}

impl Mul<Point3> for Matrix4x4 {
    type Output = Point3;
    fn mul(self, rhs: Point3) -> Self::Output {
        // only apply scale and rotation
        let [v0, v1, v2, v3]: [f32; 4] = rhs.0.into();

        // let column1: f32x4 = shuffle!(self.0, [0, 4, 8, 12]);
        // let column2: f32x4 = shuffle!(self.0, [1, 5, 9, 13]);
        // let column3: f32x4 = shuffle!(self.0, [2, 6, 10, 14]);
        // let column4: f32x4 = shuffle!(self.0, [3, 7, 11, 15]);
        let row1: f32x4 = simd_swizzle!(self.0, [0, 1, 2, 3]);
        let row2: f32x4 = simd_swizzle!(self.0, [4, 5, 6, 7]);
        let row3: f32x4 = simd_swizzle!(self.0, [8, 9, 10, 11]);
        let row4: f32x4 = simd_swizzle!(self.0, [12, 13, 14, 15]);

        let result = row1 * f32x4::splat(v0)
            + row2 * f32x4::splat(v1)
            + row3 * f32x4::splat(v2)
            + row4 * f32x4::splat(v3);

        Point3(result).normalize()
    }
}

impl Mul<Ray> for Matrix4x4 {
    type Output = Ray;
    fn mul(self, rhs: Ray) -> Self::Output {
        Ray {
            origin: (self * rhs.origin),
            direction: (self * rhs.direction).normalized(),
            ..rhs
        }
    }
}
impl Mul for Matrix4x4 {
    type Output = Matrix4x4;
    fn mul(self, rhs: Matrix4x4) -> Self::Output {
        // should probably just use nalgebra's matmul rather than my own

        let a_row1: f32x4 = simd_swizzle!(self.0, [0, 1, 2, 3]);
        let a_row2: f32x4 = simd_swizzle!(self.0, [4, 5, 6, 7]);
        let a_row3: f32x4 = simd_swizzle!(self.0, [8, 9, 10, 11]);
        let a_row4: f32x4 = simd_swizzle!(self.0, [12, 13, 14, 15]);

        let b_column1: f32x4 = simd_swizzle!(rhs.0, [0, 4, 8, 12]);
        let b_column2: f32x4 = simd_swizzle!(rhs.0, [1, 5, 9, 13]);
        let b_column3: f32x4 = simd_swizzle!(rhs.0, [2, 6, 10, 14]);
        let b_column4: f32x4 = simd_swizzle!(rhs.0, [3, 7, 11, 15]);

        let m11 = (a_row1 * b_column1).reduce_sum();
        let m12 = (a_row1 * b_column2).reduce_sum();
        let m13 = (a_row1 * b_column3).reduce_sum();
        let m14 = (a_row1 * b_column4).reduce_sum();

        let m21 = (a_row2 * b_column1).reduce_sum();
        let m22 = (a_row2 * b_column2).reduce_sum();
        let m23 = (a_row2 * b_column3).reduce_sum();
        let m24 = (a_row2 * b_column4).reduce_sum();

        let m31 = (a_row3 * b_column1).reduce_sum();
        let m32 = (a_row3 * b_column2).reduce_sum();
        let m33 = (a_row3 * b_column3).reduce_sum();
        let m34 = (a_row3 * b_column4).reduce_sum();

        let m41 = (a_row4 * b_column1).reduce_sum();
        let m42 = (a_row4 * b_column2).reduce_sum();
        let m43 = (a_row4 * b_column3).reduce_sum();
        let m44 = (a_row4 * b_column4).reduce_sum();

        Matrix4x4 {
            0: f32x16::from_array([
                m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44,
            ]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3 {
    pub forward: Matrix4x4,
    pub reverse: Matrix4x4,
}

impl Transform3 {
    pub fn new() -> Self {
        Transform3 {
            forward: Matrix4x4::I,
            reverse: Matrix4x4::I,
        }
    }
    pub fn new_from_matrix(forward: nalgebra::Matrix4<f32>) -> Option<Self> {
        forward.try_inverse().map(|inverse| Transform3 {
            forward: Matrix4x4::from(forward),
            reverse: Matrix4x4::from(inverse),
        })
    }

    pub fn inverse(self) -> Transform3 {
        // returns a transform3 that when multiplied with another Transform3, Vec3 or Point3,
        // applies the inverse transform of self
        Transform3::new_from_raw(self.reverse, self.forward)
    }

    pub fn from_translation(shift: Vec3) -> Self {
        Transform3::new_from_matrix(nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
            shift.x(),
            shift.y(),
            shift.z(),
        )))
        .expect("somehow, translate matrix was not invertible")
    }

    pub fn from_scale(scale: Vec3) -> Self {
        Transform3::new_from_matrix(nalgebra::Matrix4::new_nonuniform_scaling(
            &nalgebra::Vector3::new(scale.x(), scale.y(), scale.z()),
        ))
        .expect("somehow, scale matrix was not invertible")
    }

    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Self {
        // TODO: change this to the code at https://www.iquilezles.org/www/articles/noacos/noacos.htm

        let axisangle = radians * nalgebra::Vector3::new(axis.x(), axis.y(), axis.z());

        let affine = nalgebra::Matrix4::from_scaled_axis(axisangle);
        Transform3::new_from_matrix(affine).expect("somehow, rotation matrix was not invertible")
    }

    // pub fn rotation(quaternion: f32x4) -> Self {
    //     let quat = nalgebra::Quaternion::new()

    //     let affine = nalgebra::Matrix4::from_scaled_axis(axisangle);
    //     Transform3::new_from_matrix(affine)
    // }

    pub fn from_stack(
        scale: Option<Transform3>,
        rotate: Option<Transform3>,
        translate: Option<Transform3>,
    ) -> Transform3 {
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

    pub fn new_from_raw(forward: Matrix4x4, reverse: Matrix4x4) -> Self {
        Transform3 { forward, reverse }
    }

    // assumes vector stack is a tangent frame

    // to world is equivalent to
    // [ Tx Bx Nx        [ vx
    //   Ty By Ny    *     vy     =
    //   Tz Bz Nz ]        vz ]

    // to local is equivalent to
    // [ Tx Ty Tz        [ vx
    //   Bx By Bz    *     vy     =   [Tx * vx + Ty * vy + Tz * vz, ...]
    //   Nx Ny Nz ]        vz ]

    pub fn from_vector_stack(v0: f32x4, v1: f32x4, v2: f32x4) -> Self {
        let [m11, m12, m13, _]: [f32; 4] = v0.into();
        let [m21, m22, m23, _]: [f32; 4] = v1.into();
        let [m31, m32, m33, _]: [f32; 4] = v2.into();

        let m = Matrix4x4(f32x16::from_array([
            m11, m12, m13, 0.0, m21, m22, m23, 0.0, m31, m32, m33, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]));
        Transform3::new_from_raw(m.transpose(), m)
    }

    pub fn axis_transform(&self) -> (Vec3, Vec3, Vec3) {
        (
            self.to_world(Vec3::X),
            self.to_world(Vec3::Y),
            self.to_world(Vec3::Z),
        )
    }

    pub fn to_local<T>(&self, value: T) -> <Matrix4x4 as Mul<T>>::Output
    where
        Matrix4x4: Mul<T>,
    {
        self.reverse * value
    }
    pub fn to_world<T>(&self, value: T) -> <Matrix4x4 as Mul<T>>::Output
    where
        Matrix4x4: Mul<T>,
    {
        self.forward * value
    }
}

impl From<TangentFrame> for Transform3 {
    fn from(value: TangentFrame) -> Self {
        value.tangent;
        value.bitangent;
        value.normal;
        Transform3::from_vector_stack(value.tangent.0, value.bitangent.0, value.normal.0)
    }
}

impl From<nalgebra::Matrix4<f32>> for Matrix4x4 {
    fn from(matrix: nalgebra::Matrix4<f32>) -> Self {
        let vec: Vec<f32> = matrix.as_slice().to_owned();
        let mut elements: f32x16 = f32x16::splat(0.0);
        for (i, v) in vec.iter().enumerate() {
            *elements.index_mut(i) = *v;
        }
        Matrix4x4(elements)
    }
}

impl From<Matrix4x4> for nalgebra::Matrix4<f32> {
    fn from(other: Matrix4x4) -> Self {
        let [m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44]: [f32;
            16] = other.0.into();
        nalgebra::Matrix4::new(
            m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44,
        )
        .transpose()
    }
}

impl Mul<Transform3> for Transform3 {
    type Output = Transform3;
    fn mul(self, rhs: Transform3) -> Self::Output {
        Transform3::new_from_raw(rhs.forward * self.forward, self.reverse * rhs.reverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_transform() {
        let transform_translate = Transform3::from_translation(Vec3::new(1.0, 2.0, 0.0));
        let transform_rotate = Transform3::from_axis_angle(Vec3::Z, PI / 4.0);
        let transform_scale = Transform3::from_scale(Vec3::new(2.0, 2.0, 2.0));

        let test_vec = Vec3::new(1.0, 1.0, 1.0);
        println!("testing vec {:?}", test_vec);

        println!("{:?}", transform_translate.to_world(test_vec));
        println!("{:?}", transform_rotate.to_world(test_vec));
        println!("{:?}", transform_scale.to_world(test_vec));

        let test_point = Point3::ORIGIN + test_vec;
        println!("testing point {:?}", test_point);

        println!("{:?}", transform_translate.to_world(test_point));
        println!("{:?}", transform_rotate.to_world(test_point));
        println!("{:?}", transform_scale.to_world(test_point));
    }

    #[test]
    fn test_round_trip_error() {
        let transform_translate = Transform3::from_translation(Vec3::new(1.0, 2.0, 0.0));
        let transform_rotate = Transform3::from_axis_angle(Vec3::Z, PI / 4.0);
        // let _transform_scale_uniform = Transform3::scale(Vec3::new(2.0, 2.0, 2.0));
        let transform_scale = Transform3::from_scale(Vec3::new(2.0, 3.0, 4.0));

        let trs = transform_translate * transform_rotate * transform_scale;
        let trs2 = Transform3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );
        let rs = transform_rotate * transform_scale;
        let tr = transform_translate * transform_rotate;
        let ts = transform_translate * transform_scale;

        let test_vec = Vec3::new(1.0, 1.0, 0.0).normalized();
        println!("testing vec {:?}", test_vec);

        let eval_round_trip_error_vec = |transform: Transform3, input: Vec3| {
            (transform.to_local(transform.to_world(input)) - input).norm()
        };
        let eval_round_trip_error_point = |transform: Transform3, input: Point3| {
            (transform.to_local(transform.to_world(input)) - input).norm()
        };

        println!("vec trs, {:?}", eval_round_trip_error_vec(trs, test_vec));
        println!("vec trs2, {:?}", eval_round_trip_error_vec(trs2, test_vec));
        println!("vec  rs, {:?}", eval_round_trip_error_vec(rs, test_vec));
        println!("vec  tr, {:?}", eval_round_trip_error_vec(tr, test_vec));
        println!("vec  ts, {:?}", eval_round_trip_error_vec(ts, test_vec));

        let test_point = Point3::ORIGIN + test_vec;
        println!("testing point {:?}", test_point);

        println!(
            "point trs, {:?}",
            eval_round_trip_error_point(trs, test_point)
        );
        println!(
            "point trs2, {:?}",
            eval_round_trip_error_point(trs2, test_point)
        );
        println!(
            "point  rs, {:?}",
            eval_round_trip_error_point(rs, test_point)
        );
        println!(
            "point  tr, {:?}",
            eval_round_trip_error_point(tr, test_point)
        );
        println!(
            "point  ts, {:?}",
            eval_round_trip_error_point(ts, test_point)
        );
    }

    #[test]
    fn test_transform_combination() {
        let transform_translate = Transform3::from_translation(Vec3::new(1.0, 1.0, 0.0));
        let transform_rotate = Transform3::from_axis_angle(Vec3::Z, PI / 4.0);
        let transform_scale = Transform3::from_scale(Vec3::new(2.0, 3.0, 4.0));

        let combination_trs = transform_translate * transform_rotate * transform_scale;
        let combination_trs_2 = Transform3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );
        let Transform3 {
            forward,
            reverse: _,
        } = combination_trs_2.clone();
        let redone = Transform3::new_from_matrix(forward.into()).unwrap();
        let combination_rs = transform_rotate * transform_scale;
        let combination_tr = transform_translate * transform_rotate;
        let combination_ts = transform_translate * transform_scale;

        let test_vec = Vec3::new(1.0, 1.0, 0.0).normalized();
        println!("testing vec {:?}", test_vec);

        println!("vec trs, {:?}", combination_trs.to_world(test_vec));
        println!("vec trs_2, {:?}", combination_trs_2.to_world(test_vec));
        println!("vec redone, {:?}", redone.to_world(test_vec));
        println!("vec  rs, {:?}", combination_rs.to_world(test_vec));
        println!("vec  tr, {:?}", combination_tr.to_world(test_vec));
        println!("vec  ts, {:?}", combination_ts.to_world(test_vec));

        let test_point = Point3::ORIGIN + test_vec;
        println!("testing point {:?}", test_point);

        println!("point trs, {:?}", combination_trs.to_world(test_point));
        println!("point trs_2, {:?}", combination_trs_2.to_world(test_point));
        println!("point redone, {:?}", redone.to_world(test_point));
        println!("point  rs, {:?}", combination_rs.to_world(test_point));
        println!("point  tr, {:?}", combination_tr.to_world(test_point));
        println!("point  ts, {:?}", combination_ts.to_world(test_point));
    }

    #[test]
    fn test_reverse_transform_combination() {
        let transform_translate = Transform3::from_translation(Vec3::new(1.0, 1.0, 0.0));
        let transform_rotate = Transform3::from_axis_angle(Vec3::Z, PI / 4.0);
        let transform_scale = Transform3::from_scale(Vec3::new(2.0, 3.0, 4.0));

        let combination_trs = transform_translate * transform_rotate * transform_scale;
        let combination_trs_2 = Transform3::from_stack(
            Some(transform_scale),
            Some(transform_rotate),
            Some(transform_translate),
        );
        let combination_rs = transform_rotate * transform_scale;
        let combination_tr = transform_translate * transform_rotate;
        let combination_ts = transform_translate * transform_scale;

        let test_vec = Vec3::new(1.0, 1.0, 0.0).normalized();
        println!("testing vec {:?}", test_vec);

        println!("vec trs, {:?}", combination_trs.to_local(test_vec));
        println!("vec trs_2, {:?}", combination_trs_2.to_local(test_vec));
        println!("vec  rs, {:?}", combination_rs.to_local(test_vec));
        println!("vec  tr, {:?}", combination_tr.to_local(test_vec));
        println!("vec  ts, {:?}", combination_ts.to_local(test_vec));

        let test_point = Point3::ORIGIN + test_vec;
        println!("testing point {:?}", test_point);

        println!("point trs, {:?}", combination_trs.to_local(test_point));
        println!("point trs_2, {:?}", combination_trs_2.to_local(test_point));
        println!("point  rs, {:?}", combination_rs.to_local(test_point));
        println!("point  tr, {:?}", combination_tr.to_local(test_point));
        println!("point  ts, {:?}", combination_ts.to_local(test_point));
    }

    #[test]
    fn test_translate() {
        let n_translate =
            nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0));

        let matrix = Matrix4x4::from(n_translate);
        let point = nalgebra::Vector4::new(1.0, 2.0, 3.0, 1.0);
        let simd_vec = Vec3(f32x4::from_array([1.0, 2.0, 3.0, 0.0]));
        let simd_point = Point3(f32x4::from_array([1.0, 2.0, 3.0, 1.0]));

        let transform = Transform3::new_from_matrix(n_translate).unwrap();
        let result1 = n_translate * point;
        let result2 = matrix * simd_vec;
        let result3 = matrix * simd_point;
        let result4 = transform.to_world(simd_vec);
        let result5 = transform.to_local(simd_vec);
        let result6 = transform.to_world(simd_point);
        let result7 = transform.to_local(simd_point);
        println!(
            "{:?} {:?} {:?} {:?} {:?}",
            result1, result2, result3, result4, result5
        );
        println!("{:?} {:?}", result6, result7);
    }
}
