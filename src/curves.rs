use crate::prelude::*;

use crate::spectral::{x_bar, y_bar, z_bar};

#[cfg(feature = "deepsize")]
use deepsize::DeepSizeOf;
use ordered_float::OrderedFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::simd::num::SimdUint;
use std::simd::usizex4;

const ONE_SUB_EPSILON: f32 = 1.0 - std::f32::EPSILON;

// structs

#[derive(Debug, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub enum Op {
    Add,
    Mul,
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub enum InterpolationMode {
    Linear,
    Nearest,
    Cubic,
}

pub trait SpectralPowerDistributionFunction<T: Field> {
    // range: [0, infinty)
    fn evaluate_power(&self, lambda: T) -> T;
    // range: [0, 1]
    fn evaluate_clamped(&self, lambda: T) -> T;

    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (WavelengthEnergy<T, T>, PDF<T, Uniform01>);
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub enum Curve {
    /// f(x) = C
    /// no variation across the domain
    Const(f32),
    /// Linearly spaced samples, across a specific domain. Also takes an interpolation mode so that various curve shapes can be represented
    Linear {
        signal: Vec<f32>,
        bounds: Bounds1D,
        mode: InterpolationMode,
    },
    /// Tabulated samples, each item in signal represents an (x,y) pair. This vec is assumed to be sorted.
    Tabulated {
        signal: Vec<(f32, f32)>,
        mode: InterpolationMode,
    },
    /// An 8th degree polynomial, with the const offset term stored in `domain_range_mapping`.
    /// A good value for the x_offset and x_scale for an input x value of light wavelength in the visible range in nanometers
    /// is roughly 600 offset, 200 scale. thus, 400 is mapped to -1 and 800 is mapped to 1
    Polynomial {
        /// packed as x_offset, x_scale, y_offset, y_scale
        domain_range_mapping: [f32; 4],
        coefficients: [f32; 8],
    },
    /// [Cauchy's equation](https://en.wikipedia.org/wiki/Cauchy%27s_equation)
    Cauchy { a: f32, b: f32 },
    /// Each entry of signal is (offset, sigma1, sigma2, mult) which represents a nonsymmetric bell curve
    /// centered at `offset`, with `sigma1` as the left std deviation, `sigma2` as the right standard deviation, and `mult` as the multiplier
    /// in pseudocode, f(x) = 1.0 - sum_i^n bell_curve[i].eval(x)
    Exponential { signal: Vec<(f32, f32, f32, f32)> },
    /// Each entry of signal is (offset, sigma1, sigma2, mult) which represents a nonsymmetric bell curve
    /// centered at `offset`, with `sigma1` as the left std deviation, `sigma2` as the right standard deviation, and `mult` as the multiplier
    /// however each bell curve is actually subtracted from a baseline of 1.0,
    /// in pseudocode, f(x) = 1.0 - sum_i^n bell_curve[i].eval(x)
    InverseExponential { signal: Vec<(f32, f32, f32, f32)> },
    /// Represents a blackbody curve at a specific `temperature`, boosted by `boost`. if `boost` is 1.0, the curve is normalized to be 1.0 at the peak energy emitting wavelength in nm.
    Blackbody { temperature: f32, boost: f32 },
    /// Represents a ordered list of operations applied to a seed value,
    /// with Op being either Add or Mul of some other `Curve`,
    /// where Op::Mul is elementwise multiplication and Op::Add is elementwise addition
    /// Note that any of the member `Curve`s can themselves be another Machine,
    Machine { seed: f32, list: Vec<(Op, Curve)> },
}

impl Default for Curve {
    fn default() -> Self {
        Curve::Const(0.0)
    }
}

// impl Div<f32> for SPD {
//     type Output = SPD;
//     fn div(self, rhs: f32) -> Self::Output {
//         match self {
//             SPD::Const(c) => SPD::Const(c / rhs),
//             SPD::Blackbody { temperature, boost } => SPD::Blackbody {
//                 temperature,
//                 boost: boost / rhs,
//             },
//             _ => SPD::Machine {
//                 seed: rhs.recip(),
//                 list: vec![(Op::Mul, &self)],
//             },
//         }
//     }
// }

impl Curve {
    pub fn y_bar() -> Curve {
        Curve::Exponential {
            signal: vec![(568.0, 46.9, 40.5, 0.821), (530.9, 16.3, 31.1, 0.286)],
        }
    }

    pub fn from_function<F>(
        mut func: F,
        samples: usize,
        domain: Bounds1D,
        mode: InterpolationMode,
    ) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        let lower = domain.lower;
        let bin_size = domain.span() / samples as f32;
        let mut values = Vec::new();
        for i in 0..samples {
            let pt = (i as f32 + 0.5) * bin_size + lower;
            let value = func(pt);
            values.push(value);
        }
        Curve::Linear {
            signal: values,
            bounds: domain,
            mode,
        }
    }

    pub fn evaluate(&self, x: f32) -> f32 {
        match &self {
            Curve::Const(v) => v.max(0.0),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                if x <= bounds.lower {
                    return *signal.first().unwrap();
                } else if x >= bounds.upper {
                    return *signal.last().unwrap();
                }
                let step_size = bounds.span() / (signal.len() as f32);
                let index = ((x - bounds.lower) / step_size) as usize;
                let left = signal[index];
                let right = if index + 1 < signal.len() {
                    signal[index + 1]
                } else {
                    return signal[index];
                };
                let t = (x - (bounds.lower + index as f32 * step_size)) / step_size;
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
            Curve::Polynomial {
                domain_range_mapping,
                coefficients,
            } => {
                let [x0, xs, y0, ys]: [f32; 4] = (*domain_range_mapping).into();
                debug_assert!(xs > 0.0);
                let mut val = y0;
                let x = (x - x0) / xs;
                // y offset takes care of the constant (x^0) term
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += ys * coef * x.powi(i as i32 + 1);
                }
                val.max(0.0)
            }
            Curve::Tabulated { signal, mode } => {
                // let result = signal.binary_search_by_key(lambda, |&(a, b)| a);
                let index = match signal
                    .binary_search_by_key(&OrderedFloat::<f32>(x), |&(a, _b)| {
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
                t = (x - left.0) / (right.0 - left.0);

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
            Curve::Cauchy { a, b } => *a + *b / (x * x),
            Curve::Exponential { signal } => {
                let mut val = 0.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val += gaussianf32(x, multiplier, offset, sigma1, sigma2);
                }
                val
            }
            Curve::InverseExponential { signal } => {
                let mut val = 1.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val -= gaussianf32(x, multiplier, offset, sigma1, sigma2);
                }
                val.max(0.0)
            }
            Curve::Machine { seed, list } => {
                let mut val = *seed;
                for (op, spd) in list {
                    let eval = spd.evaluate(x);
                    val = match op {
                        Op::Add => val + eval,
                        Op::Mul => val * eval,
                    };
                }
                val.max(0.0)
            }
            Curve::Blackbody { temperature, boost } => {
                if *boost == 0.0 {
                    blackbody(*temperature, x)
                } else {
                    boost * blackbody(*temperature, x)
                        / blackbody(*temperature, max_blackbody_lambda(*temperature))
                }
            }
        }
    }

    pub fn to_cdf(&self, bounds: Bounds1D, resolution: usize) -> CurveWithCDF {
        // resolution is ignored if Curve variant is `Linear`
        match &self {
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                // converting linear curve to CDF, easy enough since you have the raw signal
                let mut cdf_signal = signal.clone();
                let mut s = 0.0;
                let step_size = bounds.span() / (signal.len() as f32);
                for (i, v) in signal.iter().enumerate() {
                    cdf_signal[i] = s;
                    s += v * step_size;
                }
                cdf_signal.push(s);

                // divide each entry in the cdf by the integral so that it ends at 1.0
                cdf_signal.iter_mut().for_each(|e| *e /= s);
                // println!("integral is {}, step_size was {}", s, step_size);
                CurveWithCDF {
                    pdf: self.clone(),
                    cdf: Curve::Linear {
                        signal: cdf_signal,
                        bounds: *bounds,
                        mode: *mode,
                    },
                    pdf_integral: s,
                }
            }
            _ => {
                // converting arbitrary curve to CDF, need to sample to compute the integral.
                // TODO: convert riemann sum to trapezoidal rule or something more accurate.
                let mut cdf_signal = Vec::new();
                let mut s = 0.0;
                let step_size = bounds.span() / (resolution as f32);
                for i in 0..resolution {
                    let lambda = bounds.lower + (i as f32) * step_size;
                    s += self.evaluate(lambda);
                    cdf_signal.push(s);
                }

                cdf_signal.iter_mut().for_each(|e| *e /= s);

                CurveWithCDF {
                    pdf: self.clone(),
                    cdf: Curve::Linear {
                        signal: cdf_signal,
                        mode: InterpolationMode::Cubic,
                        bounds,
                    },
                    pdf_integral: s,
                }
            }
        }
    }

    pub fn evaluate_integral(
        &self,
        integration_bounds: Bounds1D,
        samples: usize,
        clamped: bool,
    ) -> f32 {
        // trapezoidal rule
        let step_size = integration_bounds.span() / samples as f32;
        let mut sum = 0.0;
        let mut last_f = if clamped {
            self.evaluate(integration_bounds.lower)
                .clamp(0.0, 1.0 - std::f32::EPSILON)
        } else {
            self.evaluate(integration_bounds.lower)
        };
        for i in 1..=samples {
            let x = integration_bounds.lower + (i as f32) * step_size;
            let f_x = if clamped {
                self.evaluate(x).clamp(0.0, 1.0 - std::f32::EPSILON)
            } else {
                self.evaluate(x)
            };
            sum += step_size * (last_f.min(f_x) + 0.5 * (last_f - f_x).abs());
            last_f = f_x;
        }
        sum
    }
    pub fn convert_to_xyz(
        &self,
        integration_bounds: Bounds1D,
        step_size: f32,
        clamped: bool,
    ) -> XYZColor {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum: XYZColor = XYZColor::ZERO;
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let angstroms = lambda * 10.0;
            let val = if clamped {
                self.evaluate_clamped(lambda)
            } else {
                self.evaluate_power(lambda)
            };
            sum.0 += f32x4::from_array([
                val * x_bar(angstroms),
                val * y_bar(angstroms),
                val * z_bar(angstroms),
                0.0,
            ]) * f32x4::splat(step_size);
        }
        sum
    }
}

impl SpectralPowerDistributionFunction<f32> for Curve {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.evaluate(lambda).max(0.0)
    }
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.evaluate(lambda).clamp(0.0, ONE_SUB_EPSILON)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (SingleWavelength, PDF<f32, Uniform01>) {
        match &self {
            _ => {
                let ws = SingleWavelength::new_from_range(sample.x, wavelength_range);
                (
                    ws.replace_energy(self.evaluate(ws.lambda)),
                    PDF::new(1.0 / wavelength_range.span()), // uniform distribution
                )
            }
        }
    }
}

#[cfg(feature = "simdfloat_patch")]
impl SpectralPowerDistributionFunction<f32x4> for Curve {
    fn evaluate_power(&self, lambda: f32x4) -> f32x4 {
        match &self {
            Curve::Const(v) => f32x4::splat(v.max(0.0)),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                let splatted_step_size = f32x4::splat(bounds.span() / (signal.len() as f32));
                let index =
                    ((lambda - f32x4::splat(bounds.lower)) / splatted_step_size).cast::<usize>();

                let left = f32x4::gather_or_default(&signal, index);

                let shifted = index + usizex4::splat(1);
                let right =
                    f32x4::gather_or(&signal, shifted, f32x4::splat(*signal.last().unwrap()));

                let t = (lambda
                    - (f32x4::splat(bounds.lower) + index.cast::<f32>() * splatted_step_size))
                    / splatted_step_size;
                // println!("t is {}", t);
                match mode {
                    InterpolationMode::Linear => (f32x4::splat(1.0) - t) * left + t * right,
                    InterpolationMode::Nearest => t.simd_lt(f32x4::splat(0.5)).select(left, right),
                    InterpolationMode::Cubic => {
                        let t2 = f32x4::splat(2.0) * t;
                        let one_sub_t = f32x4::splat(1.0) - t;
                        let h00 = (f32x4::splat(1.0) + t2) * one_sub_t * one_sub_t;
                        let h01 = t * t * (f32x4::splat(3.0) - t2);
                        h00 * left + h01 * right
                    }
                }
            }

            Curve::Polynomial {
                domain_range_mapping,
                coefficients,
            } => {
                let [x0, xs, y0, ys]: [f32; 4] = (*domain_range_mapping).into();
                debug_assert!(xs > 0.0);

                let x = (lambda - f32x4::splat(x0)) / f32x4::splat(xs);

                // TODO: optimize this more
                let mut sum = f32x4::splat(y0);

                // y offset takes care of the constant (x^0) term, so start with x rather than 1
                let mut xpow = x;

                for i in 0..8 {
                    sum += f32x4::splat(coefficients[i]) * xpow;
                    xpow *= x;
                }

                sum.max(f32x4::ZERO) * f32x4::splat(ys)
            }
            Curve::Cauchy { a, b } => f32x4::splat(*a) + f32x4::splat(*b) / (lambda * lambda),
            Curve::Exponential { signal } => {
                let mut val = f32x4::splat(0.0);
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val += gaussian_f32x4(lambda, multiplier, offset, sigma1, sigma2);
                }
                val
            }
            Curve::InverseExponential { signal } => {
                let mut val = f32x4::splat(1.0);
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val -= gaussian_f32x4(lambda, multiplier, offset, sigma1, sigma2);
                }
                val.max(f32x4::splat(0.0))
            }

            Curve::Blackbody { temperature, boost } => {
                let bbd = blackbody_f32x4(*temperature, lambda);
                if *boost == 0.0 {
                    bbd
                } else {
                    // renormalize blackbody spectra so that it's all between 0 and 1, then multiply by boost.
                    f32x4::splat(*boost) * bbd
                        / f32x4::splat(blackbody(*temperature, max_blackbody_lambda(*temperature)))
                }
            }
            _ => f32x4::from_array([
                self.evaluate(lambda[0]),
                self.evaluate(lambda[1]),
                self.evaluate(lambda[2]),
                self.evaluate(lambda[3]),
            ]),
        }
    }

    fn evaluate_clamped(&self, lambda: f32x4) -> f32x4 {
        self.evaluate_power(lambda)
            .simd_clamp(f32x4::ZERO, f32x4::ONE)
    }

    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (HeroWavelength, PDF<f32x4, Uniform01>) {
        match &self {
            _ => {
                let ws = HeroWavelength::new_from_range(sample.x, wavelength_range);
                (
                    ws.replace_energy(self.evaluate_power(ws.lambda)),
                    PDF::new(f32x4::splat(1.0 / wavelength_range.span())), // uniform distribution
                )
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub struct CurveWithCDF {
    // pdf range is [0, infinity), though actual infinite values are not handled yet, and if they were it would be through special handling as dirac delta distributions
    pub pdf: Curve,
    // cdf ranges from 0 to 1
    pub cdf: Curve,
    // store pdf integral so that we don't have to normalize the `pdf` curve beforehand. instead, all samplings of the pdf when taken through the cdf should be normalized by dividing by pdf_integral.
    pub pdf_integral: f32,
}

impl SpectralPowerDistributionFunction<f32> for CurveWithCDF {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda)
    }
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.pdf.evaluate_clamped(lambda)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (SingleWavelength, PDF<f32, Uniform01>) {
        match &self.cdf {
            Curve::Const(v) => (
                SingleWavelength::new(wavelength_range.sample(sample.x), (*v).into()),
                (1.0 / self.pdf_integral).into(),
            ),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                let restricted_bounds = bounds.intersection(wavelength_range);
                // remap sample.x to lie between the values that correspond to restricted_bounds.lower and restricted_bounds.upper
                let lower_cdf_value = self.cdf.evaluate(restricted_bounds.lower);
                let upper_cdf_value = self.cdf.evaluate(restricted_bounds.upper);
                sample.x = lower_cdf_value + sample.x * (upper_cdf_value - lower_cdf_value);
                // println!("{:?}", self.cdf);
                // println!(
                //     "remapped sample value to be {:?} which is between {:?} and {:?}",
                //     sample.x, lower_cdf_value, upper_cdf_value
                // );
                let maybe_index = signal
                    .binary_search_by_key(&OrderedFloat::<f32>(sample.x), |&a| {
                        OrderedFloat::<f32>(a)
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
                            let v0 = signal[index - 1];
                            let v1 = signal[index];
                            let t = if v0 != v1 {
                                (sample.x - v0) / (v1 - v0)
                            } else {
                                0.0
                            };

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
                let power = self.pdf.evaluate(lambda);

                // println!("power was {}", power);
                (
                    SingleWavelength::new(lambda, power.into()),
                    PDF::from(power / self.pdf_integral),
                )
            }
            // should this be self.pdf.sample_power_and_pdf?
            _ => self.cdf.sample_power_and_pdf(wavelength_range, sample),
        }
    }
}

// TODO: figure out how to use SMIS/CMIS for these sample functions, especially with CurveWithCDF

#[cfg(feature = "simdfloat_patch")]
impl SpectralPowerDistributionFunction<f32x4> for CurveWithCDF {
    fn evaluate_power(&self, lambda: f32x4) -> f32x4 {
        self.pdf.evaluate_power(lambda)
    }
    fn evaluate_clamped(&self, lambda: f32x4) -> f32x4 {
        self.pdf.evaluate_clamped(lambda)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (HeroWavelength, PDF<f32x4, Uniform01>) {
        match &self.cdf {
            Curve::Const(v) => (
                HeroWavelength::new_from_range(sample.x, wavelength_range)
                    .replace_energy(f32x4::splat(*v)),
                f32x4::splat(1.0 / self.pdf_integral).into(),
            ),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                let restricted_bounds = bounds.intersection(wavelength_range);
                // remap sample.x to lie between the values that correspond to restricted_bounds.lower and restricted_bounds.upper
                let lower_cdf_value = self.cdf.evaluate(restricted_bounds.lower);
                let upper_cdf_value = self.cdf.evaluate(restricted_bounds.upper);
                sample.x = lower_cdf_value + sample.x * (upper_cdf_value - lower_cdf_value);
                // println!("{:?}", self.cdf);
                // println!(
                //     "remapped sample value to be {:?} which is between {:?} and {:?}",
                //     sample.x, lower_cdf_value, upper_cdf_value
                // );
                let maybe_index = signal
                    .binary_search_by_key(&OrderedFloat::<f32>(sample.x), |&a| {
                        OrderedFloat::<f32>(a)
                    });
                let hero_lambda = match maybe_index {
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
                            let v0 = signal[index - 1];
                            let v1 = signal[index];
                            let t = if v0 != v1 {
                                (sample.x - v0) / (v1 - v0)
                            } else {
                                0.0
                            };

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
                let correlated_sample_x = (hero_lambda - bounds.lower) / bounds.span();
                let out_we = HeroWavelength::new_from_range(correlated_sample_x, *bounds);
                let power: f32x4 = self.pdf.evaluate_power(out_we.lambda);

                // println!("power was {}", power);
                (
                    out_we.replace_energy(power),
                    f32x4::splat(power[0] / self.pdf_integral).into(),
                )
            }
            // should this be self.pdf.sample_power_and_pdf?
            _ => self.cdf.sample_power_and_pdf(wavelength_range, sample),
        }
    }
}

// TODO: impl SPDF<f32x4> for CurveWithCDF and Curve
/*

fn sample_power_and_pdf(
    &self,
    wavelength_range: Bounds1D,
    sample: Sample1D,
) -> (HeroWavelength, PDFx4) {
    // let hero = HeroWavelength::new_from_range(sample.x, wavelength_range);
    let (sw, pdf) = self.sample_power_and_pdf(wavelength_range, sample);
    let transformed_sample =
        Sample1D::new((sw.lambda - wavelength_range.lower) / wavelength_range.span());
    let mut hw = HeroWavelength::new_from_range(transformed_sample.x, wavelength_range);
    hw.energy.0 = hw.energy.0.replace(0, sw.energy.0);

    // replace other energies with spectra evaluated at lambda
    for i in 1..4 {
        hw.energy.0 = hw
            .energy
            .0
            .replace(i, self.pdf.evaluate(hw.lambda.extract(i)));
    }
    // TODO: reconsider what the pdf of the other wavelengths should be in this case.
    (hw, PDFx4::from(f32x4::splat(pdf.0)))
} */

#[cfg(test)]
mod test {
    use crate::{sample::*, spectral::BOUNDED_VISIBLE_RANGE};

    use super::*;

    #[test]
    fn test_y_bar_spd() {
        let spd = Curve::y_bar();
        assert!(spd.evaluate(550.0) == 0.99955124);
    }

    #[test]
    fn test_curve_const() {
        let test_curve = Curve::Const(0.5);
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }
    #[test]
    fn test_curve_tabulated() {
        let test_curve = Curve::Tabulated {
            signal: vec![],
            mode: InterpolationMode::Linear,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }
    #[test]
    fn test_curve_linear() {
        let test_curve = Curve::Linear {
            signal: vec![],
            bounds: Bounds1D::new(400.0, 800.0),
            mode: InterpolationMode::Linear,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }
    #[test]
    fn test_curve_cauchy() {
        let test_curve = Curve::Cauchy { a: 1.4, b: 2400.0 };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 100, false);
        assert_eq!(integral / 400.0, 1.4075);
    }
    #[test]
    fn test_curve_blackbody() {
        let test_curve = Curve::Blackbody {
            temperature: 5400.0,
            boost: 1.0,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }
    #[test]
    fn test_curve_exponential() {
        let test_curve = Curve::Exponential { signal: todo!() };
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }
    #[test]
    fn test_curve_inverse_exponential() {
        let test_curve = Curve::InverseExponential { signal: todo!() };
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_eq!(integral, 50.0);
    }

    #[test]
    #[cfg(feature = "simdfloat_patch")]
    fn test_curve_polynomial() {
        let curve = Curve::Polynomial {
            domain_range_mapping: [600.0, 200.0, 0.5, 0.06],
            coefficients: [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        };

        let result = curve.evaluate_power(f32x4::from_array([450.0, 550.0, 650.0, 750.0]));
        println!("{:?}", result);
    }
    #[test]
    fn test_curve_machine() {}

    #[test]
    fn test_cdf1() {
        let cdf: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let mut s = 0.0;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf2() {
        let cdf: CurveWithCDF = Curve::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let mut s = 0.0;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf3() {
        // test sampling according to the CDF with narrowed bounds wrt the original signal bounds
        let cdf: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let narrowed_bounds = Bounds1D::new(500.0, 600.0);
        let mut s = 0.0;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf4() {
        // test sampling according to the CDF with narrowed bounds in general
        let narrowed_bounds = Bounds1D::new(500.0, 600.0);

        let cdf: CurveWithCDF = Curve::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        }
        .to_cdf(narrowed_bounds, 100);

        let mut s = 0.0;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{}", s);
    }

    #[test]
    fn test_cdf_addition() {
        let cdf1: CurveWithCDF = Curve::Exponential {
            signal: vec![(400.0, 100.0, 100.0, 0.9), (600.0, 100.0, 100.0, 1.0)],
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        for i in 0..100 {
            let lambda = BOUNDED_VISIBLE_RANGE.lerp(i as f32 / 100.0);
            println!(
                "{}, {}, {}",
                lambda,
                cdf1.pdf.evaluate(lambda),
                cdf1.cdf.evaluate(lambda)
            );
        }
        println!();
        let cdf2: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        for i in 0..100 {
            let lambda = BOUNDED_VISIBLE_RANGE.lerp(i as f32 / 100.0);
            println!(
                "{}, {}, {}",
                lambda,
                cdf2.pdf.evaluate(lambda),
                cdf2.cdf.evaluate(lambda)
            );
        }
        println!();
        let integral1 = cdf1.pdf_integral;
        let integral2 = cdf2.pdf_integral;

        let combined_spd = Curve::Machine {
            seed: 0.0,
            list: vec![(Op::Add, cdf1.pdf), (Op::Add, cdf2.pdf)],
        };

        let combined_cdf_curve = Curve::Machine {
            seed: 0.0,
            list: vec![(Op::Add, cdf1.cdf), (Op::Add, cdf2.cdf)],
        };

        // let combined_spd
        let combined_cdf = CurveWithCDF {
            pdf: combined_spd,
            cdf: combined_cdf_curve,
            pdf_integral: integral1 + integral2,
        };
        for i in 0..100 {
            let lambda = BOUNDED_VISIBLE_RANGE.lerp(i as f32 / 100.0);
            println!(
                "{}, {}, {}",
                lambda,
                combined_cdf.pdf.evaluate(lambda),
                combined_cdf.cdf.evaluate(lambda)
            );
        }

        let mut s = 0.0;
        for _ in 0..1000 {
            let (we, pdf): (_, PDF<f32, _>) = combined_cdf
                .sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("\n\n{} {}", s / 1000.0, combined_cdf.pdf_integral);
    }

    #[test]
    fn test_from_func() {
        let bounds = Bounds1D::new(0.0, 1.0);
        let curve = Curve::from_function(|x| x * x, 100, bounds, InterpolationMode::Cubic);

        let true_integral = |x: f32| x * x * x / 3.0;
        println!(
            "{}, {}",
            true_integral(1.0) - true_integral(0.0),
            curve.evaluate_integral(bounds, 100, false)
        );
    }

    #[test]
    fn test_cdf_from_func() {
        let bounds = Bounds1D::new(0.0, 1.0);
        let curve = Curve::from_function(|x| x * x, 100, bounds, InterpolationMode::Cubic);

        let true_integral = |x: f32| x * x * x / 3.0;
        let true_integral = true_integral(1.0) - true_integral(0.0);
        let cdf = curve.to_cdf(bounds, 100);

        println!("pdf integral is {}", cdf.pdf_integral);

        let samples = 100;
        let mut estimate = 0.0;
        let mut variance_pt_1 = 0.0;
        let mut sampler = StratifiedSampler::new(20, 10, 10);

        let mut min_sample_x = 1.0;
        for _ in 0..samples {
            let sample = if true {
                Sample1D::new_random_sample()
            } else {
                sampler.draw_1d()
            };
            let (v, pdf) = cdf.sample_power_and_pdf(bounds, sample);
            if v.lambda < min_sample_x {
                min_sample_x = v.lambda;
            }
            estimate += v.energy / *pdf / samples as f32;
            println!("{}, {}, {:?}", v.lambda, v.energy, pdf);
            variance_pt_1 += (v.energy / *pdf).powi(2) / samples as f32;
        }
        println!("estimate = {}, true_integral = {}", estimate, true_integral);
        println!(
            "variance = {:?}",
            (variance_pt_1 - estimate.powi(2)) / (samples - 1) as f32
        );
        println!("lowest sample is {}", min_sample_x);
    }

    #[test]
    #[cfg(feature = "simdfloat_patch")]
    fn test_cdf_sample_hwss() {
        let cdf: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let mut s = f32x4::ZERO;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32x4, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{:?}", s);
    }
}
