use crate::color::XYZColor;
use crate::misc::gaussian;
use crate::*;

mod hw;
mod sw;

pub use hw::{HeroEnergy, HeroWavelength};
pub use sw::{SingleEnergy, SingleWavelength};

use packed_simd::f32x4;

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

pub fn x_bar_f32x4(angstroms: f32x4) -> f32x4 {
    gaussian_f32x4(angstroms, 1.056, 5998.0, 379.0, 310.0)
        + gaussian_f32x4(angstroms, 0.362, 4420.0, 160.0, 267.0)
        + gaussian_f32x4(angstroms, -0.065, 5011.0, 204.0, 262.0)
}

pub fn y_bar_f32x4(angstroms: f32x4) -> f32x4 {
    gaussian_f32x4(angstroms, 0.821, 5688.0, 469.0, 405.0)
        + gaussian_f32x4(angstroms, 0.286, 5309.0, 163.0, 311.0)
}

pub fn z_bar_f32x4(angstroms: f32x4) -> f32x4 {
    gaussian_f32x4(angstroms, 1.217, 4370.0, 118.0, 360.0)
        + gaussian_f32x4(angstroms, 0.681, 4590.0, 260.0, 138.0)
}

impl From<SingleWavelength> for XYZColor {
    fn from(swss: SingleWavelength) -> Self {
        // convert to Angstroms. 10 Angstroms == 1nm
        let angstroms = swss.lambda * 10.0;

        XYZColor::new(
            swss.energy.0 * x_bar(angstroms),
            swss.energy.0 * y_bar(angstroms),
            swss.energy.0 * z_bar(angstroms),
        )
    }
}

impl From<HeroWavelength> for XYZColor {
    fn from(hwss: HeroWavelength) -> Self {
        let angstroms = hwss.lambda * 10.0;

        XYZColor::new(
            (x_bar_f32x4(angstroms) * hwss.energy.0).sum(),
            (y_bar_f32x4(angstroms) * hwss.energy.0).sum(),
            (z_bar_f32x4(angstroms) * hwss.energy.0).sum(),
        )
    }
}
