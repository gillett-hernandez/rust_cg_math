use crate::color::XYZColor;
use crate::misc::gaussian;
use crate::*;

use ordered_float::OrderedFloat;
use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SingleEnergy(pub f32);

impl SingleEnergy {
    pub fn new(energy: f32) -> Self {
        SingleEnergy { 0: energy }
    }
    pub const ZERO: SingleEnergy = SingleEnergy { 0: 0.0 };
    pub const ONE: SingleEnergy = SingleEnergy { 0: 1.0 };
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}
impl Add for SingleEnergy {
    type Output = SingleEnergy;
    fn add(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 + rhs.0)
    }
}
impl AddAssign for SingleEnergy {
    fn add_assign(&mut self, rhs: SingleEnergy) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 * rhs)
    }
}
impl Mul<SingleEnergy> for f32 {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self * rhs.0)
    }
}

impl Mul for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 * rhs.0)
    }
}

impl MulAssign for SingleEnergy {
    fn mul_assign(&mut self, other: SingleEnergy) {
        self.0 = self.0 * other.0
    }
}

impl MulAssign<f32> for SingleEnergy {
    fn mul_assign(&mut self, other: f32) {
        self.0 = self.0 * other
    }
}

impl Div<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn div(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 / rhs)
    }
}

impl From<f32> for SingleEnergy {
    fn from(value: f32) -> Self {
        SingleEnergy::new(value)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SingleWavelength {
    pub lambda: f32,
    pub energy: SingleEnergy,
}

impl SingleWavelength {
    pub const fn new(lambda: f32, energy: SingleEnergy) -> SingleWavelength {
        SingleWavelength { lambda, energy }
    }

    pub fn new_from_range(x: f32, bounds: Bounds1D) -> Self {
        SingleWavelength::new(bounds.lower + x * bounds.span(), SingleEnergy::ZERO)
    }

    pub fn with_energy(&self, energy: SingleEnergy) -> Self {
        SingleWavelength::new(self.lambda, energy)
    }

    pub fn replace_energy(&self, energy: f32) -> Self {
        self.with_energy(SingleEnergy::new(energy))
    }

    pub const BLACK: SingleWavelength = SingleWavelength::new(0.0, SingleEnergy::ZERO);
}

impl Mul<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy * other)
    }
}

impl Mul<SingleWavelength> for f32 {
    type Output = SingleWavelength;
    fn mul(self, other: SingleWavelength) -> SingleWavelength {
        other.with_energy(self * other.energy)
    }
}

impl Mul<XYZColor> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, _xyz: XYZColor) -> SingleWavelength {
        // let lambda = other.wavelength;
        // let other_as_color: XYZColor = other.into();
        // other_as_color gives us the x y and z values for other
        // self.with_energy(self.energy * xyz.y())
        unimplemented!()
    }
}

impl Div<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn div(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy / other)
    }
}

impl DivAssign<f32> for SingleWavelength {
    fn div_assign(&mut self, other: f32) {
        self.energy = self.energy / other;
    }
}

impl Mul<SingleEnergy> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        self.with_energy(self.energy * rhs)
    }
}

impl Mul<SingleWavelength> for SingleEnergy {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleWavelength) -> Self::Output {
        rhs.with_energy(self * rhs.energy)
    }
}

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

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Op {
    Add,
    Mul,
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum InterpolationMode {
    Linear,
    Nearest,
    Cubic,
}

#[derive(Debug, Clone)]
pub enum SPD {
    Linear {
        signal: Vec<f32>,
        bounds: Bounds1D,
        mode: InterpolationMode,
    },
    Tabulated {
        signal: Vec<(f32, f32)>,
        mode: InterpolationMode,
    },
    Polynomial {
        xoffset: f32,
        coefficients: [f32; 8],
    },
    Cauchy {
        a: f32,
        b: f32,
    },
    Exponential {
        signal: Vec<(f32, f32, f32, f32)>,
    },
    InverseExponential {
        signal: Vec<(f32, f32, f32, f32)>,
    },
    Blackbody {
        temperature: f32,
        boost: f32,
    },
    Machine {
        seed: f32,
        list: Vec<(Op, SPD)>,
    },
}

impl Default for SPD {
    fn default() -> Self {
        SPD::Linear {
            signal: vec![0.0],
            bounds: EXTENDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CDF {
    pub pdf: SPD,
    pub cdf: SPD,
    pub cdf_integral: f32,
}

pub trait SpectralPowerDistributionFunction {
    fn evaluate(&self, lambda: f32) -> f32;
    fn evaluate_power(&self, lambda: f32) -> f32;
    // note: sample power
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (SingleWavelength, PDF);
    fn evaluate_integral(&self, integration_bounds: Bounds1D, step_size: f32) -> f32;
    fn convert_to_xyz(&self, integration_bounds: Bounds1D, step_size: f32) -> XYZColor {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum: XYZColor = XYZColor::ZERO;
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let angstroms = lambda * 10.0;
            let val = self.evaluate_power(lambda);
            sum.0 += f32x4::new(
                val * x_bar(angstroms),
                val * y_bar(angstroms),
                val * z_bar(angstroms),
                0.0,
            ) * step_size;
        }
        sum
    }
}

impl SpectralPowerDistributionFunction for SPD {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        match &self {
            SPD::Linear {
                signal,
                bounds,
                mode,
            } => {
                if !bounds.contains(&lambda) {
                    return 0.0;
                }
                let step_size = bounds.span() / (signal.len() as f32);
                let index = ((lambda - bounds.lower) / step_size) as usize;
                let left = signal[index];
                let right = if index + 1 < signal.len() {
                    signal[index + 1]
                } else {
                    return signal[index];
                };
                let t = (lambda - (bounds.lower + index as f32 * step_size)) / step_size;
                // println!("t is {}", t);
                match mode {
                    InterpolationMode::Linear => (1.0 - t) * left + t * right,
                    InterpolationMode::Nearest => {
                        if t < 0.5 {
                            left
                        } else {
                            right
                        }
                    }
                    InterpolationMode::Cubic => {
                        let t2 = 2.0 * t;
                        let one_sub_t = 1.0 - t;
                        let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                        let h01 = t * t * (3.0 - t2);
                        h00 * left + h01 * right
                    }
                }
            }
            SPD::Polynomial {
                xoffset,
                coefficients,
            } => {
                let mut val = 0.0;
                let tmp_lambda = lambda - xoffset;
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += coef * tmp_lambda.powi(i as i32);
                }
                val
            }
            SPD::Tabulated { signal, mode } => {
                // let result = signal.binary_search_by_key(lambda, |&(a, b)| a);
                let index = match signal
                    .binary_search_by_key(&OrderedFloat::<f32>(lambda), |&(a, _b)| {
                        OrderedFloat::<f32>(a)
                    }) {
                    Err(index) if index > 0 => index,
                    Ok(index) | Err(index) => index,
                };
                if index == signal.len() {
                    let left = signal[index - 1];
                    return left.1;
                }
                let right = signal[index];
                let t;
                if index == 0 {
                    return right.1;
                }
                let left = signal[index - 1];
                t = (lambda - left.0) / (right.0 - left.0);

                match mode {
                    InterpolationMode::Linear => (1.0 - t) * left.1 + t * right.1,
                    InterpolationMode::Nearest => {
                        if t < 0.5 {
                            left.1
                        } else {
                            right.1
                        }
                    }
                    InterpolationMode::Cubic => {
                        let t2 = 2.0 * t;
                        let one_sub_t = 1.0 - t;
                        let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                        let h01 = t * t * (3.0 - t2);
                        h00 * left.1 + h01 * right.1
                    }
                }
            }
            SPD::Cauchy { a, b } => *a + *b / (lambda * lambda),
            SPD::Exponential { signal } => {
                let mut val = 0.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val += gaussianf32(lambda, multiplier, offset, sigma1, sigma2);
                }
                val
            }
            SPD::InverseExponential { signal } => {
                let mut val = 1.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val -= gaussianf32(lambda, multiplier, offset, sigma1, sigma2);
                }
                val.max(0.0)
            }
            SPD::Machine { seed, list } => {
                let mut val = *seed;
                for (op, spd) in list {
                    let eval = spd.evaluate_power(lambda);
                    val = match op {
                        Op::Add => val + eval,
                        Op::Mul => val * eval,
                    };
                }
                val.max(0.0)
            }
            SPD::Blackbody { temperature, boost } => {
                if *boost == 0.0 {
                    blackbody(*temperature, lambda)
                } else {
                    boost * blackbody(*temperature, lambda)
                        / blackbody(*temperature, max_blackbody_lambda(*temperature))
                }
            }
        }
    }
    fn evaluate(&self, lambda: f32) -> f32 {
        // use the same curves as power distributions for reflectance functions, but cap it to 1.0 so no energy is ever added
        self.evaluate_power(lambda).min(1.0)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (SingleWavelength, PDF) {
        match &self {
            _ => {
                let ws = SingleWavelength::new_from_range(sample.x, wavelength_range);
                (
                    ws.replace_energy(self.evaluate_power(ws.lambda)),
                    PDF::from(1.0 / wavelength_range.span()), // uniform distribution
                )
            }
        }
    }
    fn evaluate_integral(&self, integration_bounds: Bounds1D, step_size: f32) -> f32 {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum = 0.0;
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let val = self.evaluate_power(lambda);
            sum += val * step_size;
        }
        sum
    }
}

impl SpectralPowerDistributionFunction for CDF {
    fn evaluate(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda)
    }
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.pdf.evaluate_power(lambda)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (SingleWavelength, PDF) {
        match &self.cdf {
            SPD::Linear {
                signal,
                bounds,
                mode,
            } => {
                let restricted_bounds = bounds.intersection(wavelength_range);
                // remap sample.x to lie between the values that correspond to restricted_bounds.lower and restricted_bounds.upper
                let lower_cdf_value =
                    self.cdf.evaluate_power(restricted_bounds.lower - 0.0001) / self.cdf_integral;
                let upper_cdf_value =
                    self.cdf.evaluate_power(restricted_bounds.upper - 0.0001) / self.cdf_integral;
                sample.x = lower_cdf_value + sample.x * (upper_cdf_value - lower_cdf_value);
                // println!(
                //     "remapped sample value to be {} which is between {} and {}",
                //     sample.x, lower_cdf_value, upper_cdf_value
                // );
                let maybe_index = signal
                    .binary_search_by_key(&OrderedFloat::<f32>(sample.x), |&a| {
                        OrderedFloat::<f32>(a / self.cdf_integral)
                    });
                let lambda = match maybe_index {
                    Ok(index) | Err(index) => {
                        if index == 0 {
                            // index is at end, so return lambda that corresponds to index
                            bounds.lower
                        } else {
                            let left = bounds.lower
                                + (index as f32 - 1.0) * (bounds.upper - bounds.lower)
                                    / (signal.len() as f32);
                            let right = bounds.lower
                                + (index as f32) * (bounds.upper - bounds.lower)
                                    / (signal.len() as f32);
                            let v0 = signal[index - 1] / self.cdf_integral;
                            let v1 = signal[index] / self.cdf_integral;
                            let t = (sample.x - v0) / (v1 - v0);
                            assert!(0.0 <= t && t <= 1.0, "{}, {}, {}, {}", t, sample.x, v0, v1);
                            match mode {
                                InterpolationMode::Linear => (1.0 - t) * left + t * right,
                                InterpolationMode::Nearest => {
                                    if t < 0.5 {
                                        left
                                    } else {
                                        right
                                    }
                                }
                                InterpolationMode::Cubic => {
                                    let t2 = 2.0 * t;
                                    let one_sub_t = 1.0 - t;
                                    let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                                    let h01 = t * t * (3.0 - t2);
                                    h00 * left + h01 * right
                                }
                            }
                            .clamp(bounds.lower, bounds.upper)
                        }
                    }
                };
                // println!("lambda was {}", lambda);
                let power = self.pdf.evaluate_power(lambda);

                // println!("power was {}", power);
                (
                    SingleWavelength::new(lambda, power.into()),
                    PDF::from(power / self.cdf_integral),
                )
            }
            _ => self.cdf.sample_power_and_pdf(wavelength_range, sample),
        }
    }
    fn evaluate_integral(&self, integration_bounds: Bounds1D, step_size: f32) -> f32 {
        self.pdf.evaluate_integral(integration_bounds, step_size)
    }
}

impl From<SPD> for CDF {
    fn from(curve: SPD) -> Self {
        match &curve {
            SPD::Linear {
                signal,
                bounds,
                mode,
            } => {
                let mut cdf_signal = signal.clone();
                let mut s = 0.0;
                let step_size = bounds.span() / (signal.len() as f32);
                for (i, v) in signal.iter().enumerate() {
                    s += v * step_size;
                    cdf_signal[i] = s;
                }
                // println!("integral is {}, step_size was {}", s, step_size);
                CDF {
                    pdf: curve.clone(),
                    cdf: SPD::Linear {
                        signal: cdf_signal,
                        bounds: *bounds,
                        mode: *mode,
                    },
                    cdf_integral: s,
                }
            }
            _ => {
                let mut cdf_signal = Vec::new();
                let mut s = 0.0;
                let num_samples = 400;
                let step_size = BOUNDED_VISIBLE_RANGE.span() / (num_samples as f32);
                for i in 0..num_samples {
                    let lambda = BOUNDED_VISIBLE_RANGE.lower + (i as f32) * step_size;
                    s += curve.evaluate_power(lambda);
                    // println!("lambda: {}, s: {}", lambda, s);
                    cdf_signal.push(s);
                }
                // println!("cdf integral was {}", s);
                CDF {
                    pdf: curve.clone(),
                    cdf: SPD::Linear {
                        signal: cdf_signal,
                        bounds: BOUNDED_VISIBLE_RANGE,
                        mode: InterpolationMode::Cubic,
                    },
                    cdf_integral: s,
                }
                // CDF {
                // pdf: curve.clone(),
                // cdf: curve.clone(),
                // cdf_integral: curve.evaluate_integral(EXTENDED_VISIBLE_RANGE, 1.0),,
            }
        }
    }
}

// pub trait RereflectanceFunction {
//     fn evaluate(&self, lambda: f32, energy: f32) -> (f32, f32);
// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cdf1() {
        let cdf: CDF = SPD::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .into();

        let mut s = 0.0;
        for _ in 0..100 {
            let sampled =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf2() {
        let cdf: CDF = SPD::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        }
        .into();

        let mut s = 0.0;
        for _ in 0..100 {
            let sampled =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf3() {
        // test sampling according to the CDF with narrowed bounds wrt the original signal bounds
        let cdf: CDF = SPD::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .into();

        let narrowed_bounds = Bounds1D::new(500.0, 600.0);
        let mut s = 0.0;
        for _ in 0..100 {
            let sampled = cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf4() {
        // test sampling according to the CDF with narrowed bounds in general
        let cdf: CDF = SPD::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        }
        .into();

        let narrowed_bounds = Bounds1D::new(500.0, 600.0);
        let mut s = 0.0;
        for _ in 0..100 {
            let sampled = cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
        }
        println!("{}", s);
    }
}
