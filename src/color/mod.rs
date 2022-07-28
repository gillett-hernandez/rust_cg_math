// use packed_simd::f32x4;

use nalgebra::{Matrix3, Vector3};

mod rgb;
mod xyz;
pub use rgb::RGBColor;
pub use xyz::XYZColor;

// using matrix data from https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_RGB_color_space

impl From<RGBColor> for XYZColor {
    fn from(rgb: RGBColor) -> Self {
        let rgb_to_xyz: Matrix3<f32> = Matrix3::new(
            0.490, 0.310, 0.200, 0.17697, 0.8124, 0.01063, 0.0, 0.01, 0.99,
        );
        let [a, b, c, _]: [f32; 4] = rgb.0.into();
        let intermediate = rgb_to_xyz * Vector3::new(a, b, c) / 0.17697;
        XYZColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}

// using matrix data from above, but inverted.
// implies that RGBColor is in the CIE RGB color space. keep this in mind if you ever use it.

impl From<XYZColor> for RGBColor {
    fn from(xyz: XYZColor) -> Self {
        let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
            0.41847, -0.15866, -0.082835, -0.091169, 0.25243, 0.015708, 0.00092090, -0.0025498,
            0.17860,
        );
        let [a, b, c, _]: [f32; 4] = xyz.0.into();
        let intermediate = xyz_to_rgb * Vector3::new(a, b, c);
        RGBColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_back_and_forth_xyz() {
        let color = XYZColor::new(1.0, 1.0, 1.0);
        let rgb_color = RGBColor::from(color);
        let xyz_color = XYZColor::from(rgb_color);
        // println!("{:?} {:?}", color, xyz_color);
        assert!((xyz_color.0 - color.0).abs().sum() < 0.0001);
    }
    #[test]
    fn test_back_and_forth_rgb() {
        let color = RGBColor::new(1.0, 1.0, 1.0);
        let xyz_color = XYZColor::from(color);
        let rgb_color = RGBColor::from(xyz_color);
        // println!("{:?} {:?}", color, xyz_color);
        assert!((rgb_color.0 - color.0).abs().sum() < 0.0001);
    }

    #[test]
    fn test_matrices() {
        let m0 = Matrix3::new(
            0.490, 0.310, 0.200, 0.17697, 0.8124, 0.01063, 0.0, 0.01, 0.99,
        );
        let m1 = Matrix3::new(
            0.41847, -0.15866, -0.082835, -0.091169, 0.25243, 0.015708, 0.00092090, -0.0025498,
            0.17860,
        );
        println!("{:?}", m1 * m0);
    }
}
