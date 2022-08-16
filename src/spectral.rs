use crate::prelude::*;

use packed_simd::f32x4;

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

pub type SingleWavelength = WavelengthEnergy<f32, f32>;
pub type HeroWavelength = WavelengthEnergy<f32x4, f32x4>;

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

// traits

pub trait WavelengthEnergyTrait<L: Field, E: Field> {
    fn new(lambda: L, energy: E) -> WavelengthEnergy<L, E> {
        WavelengthEnergy { lambda, energy }
    }
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<L, E>;
}

// does a WavelengthEnergy with L != E make any sense?
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WavelengthEnergy<L: Field, E: Field> {
    pub lambda: L,
    pub energy: E,
}

impl<L: Field, E: Field> WavelengthEnergy<L, E> {
    pub fn replace_energy(self, e: E) -> Self {
        Self { energy: e, ..self }
    }
}

impl From<WavelengthEnergy<f32, f32>> for XYZColor {
    fn from(we: WavelengthEnergy<f32, f32>) -> Self {
        let angstroms = we.lambda * 10.0;
        XYZColor::new(
            we.energy * x_bar(angstroms),
            we.energy * y_bar(angstroms),
            we.energy * z_bar(angstroms),
        )
    }
}

impl From<WavelengthEnergy<f32x4, f32x4>> for XYZColor {
    fn from(we: WavelengthEnergy<f32x4, f32x4>) -> Self {
        let angstroms = we.lambda * 10.0;
        XYZColor::new(
            (we.energy * x_bar_f32x4(angstroms)).sum(),
            (we.energy * y_bar_f32x4(angstroms)).sum(),
            (we.energy * z_bar_f32x4(angstroms)).sum(),
        )
    }
}

impl WavelengthEnergyTrait<f32, f32> for WavelengthEnergy<f32, f32> {
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<f32, f32> {
        WavelengthEnergy {
            lambda: bounds.lower + sample * bounds.span(),
            energy: 0.0,
        }
    }
}

impl WavelengthEnergyTrait<f32x4, f32x4> for WavelengthEnergy<f32x4, f32x4> {
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<f32x4, f32x4> {
        let hero = sample * bounds.span();
        let delta = bounds.span() / 4.0;
        let mult = f32x4::new(0.0, 1.0, 2.0, 3.0);
        let wavelengths = bounds.lower + (hero + mult * delta);
        let sub: f32x4 = wavelengths
            .gt(f32x4::splat(bounds.upper))
            .select(f32x4::splat(bounds.span()), f32x4::splat(0.0));
        HeroWavelength::new(wavelengths - sub, f32x4::splat(0.0))
    }
}
