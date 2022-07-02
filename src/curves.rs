use crate::{
    blackbody, blackbody_f32x4, gaussian_f32x4, gaussianf32, max_blackbody_lambda, Bounds1D, PDFx4,
    Sample1D, PDF,
};

use crate::spectral::{HeroWavelength, SingleWavelength, SpectralPowerDistributionFunction};

use ordered_float::OrderedFloat;
use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

const ONE_SUB_EPSILON: f32 = 1.0 - std::f32::EPSILON;

// structs

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
pub enum Curve {
    Const(f32),
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
    // offset, sigma1, sigma2, mult
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
        list: Vec<(Op, Curve)>,
    },
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

    pub fn from_function<F>(mut func: F, samples: usize, domain: Bounds1D) -> Self
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
            mode: InterpolationMode::Cubic,
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
                xoffset,
                coefficients,
            } => {
                let mut val = 0.0;
                let tmp_lambda = x - xoffset;
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += coef * tmp_lambda.powi(i as i32);
                }
                val
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

    #[allow(dead_code)]
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
}

impl SpectralPowerDistributionFunction for Curve {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.evaluate(lambda)
    }
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.evaluate(lambda).min(ONE_SUB_EPSILON)
    }
    fn evaluate_power_hero(&self, lambda: f32x4) -> f32x4 {
        match &self {
            Curve::Const(v) => f32x4::splat(v.max(0.0)),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                let step_size = bounds.span() / (signal.len() as f32);
                let index = (lambda - bounds.lower) / step_size;
                let left = f32x4::new(
                    signal[index.extract(0) as usize],
                    signal[index.extract(1) as usize],
                    signal[index.extract(2) as usize],
                    signal[index.extract(3) as usize],
                );
                let squeeze = |idx: usize| {
                    if idx + 1 < signal.len() {
                        signal[idx + 1]
                    } else {
                        signal[idx]
                    }
                };
                let right = f32x4::new(
                    squeeze(index.extract(0) as usize),
                    squeeze(index.extract(1) as usize),
                    squeeze(index.extract(2) as usize),
                    squeeze(index.extract(3) as usize),
                );
                let t = (lambda - (bounds.lower + index * step_size)) / step_size;
                // println!("t is {}", t);
                match mode {
                    InterpolationMode::Linear => (1.0 - t) * left + t * right,
                    InterpolationMode::Nearest => t.lt(f32x4::splat(0.5)).select(left, right),
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
                xoffset,
                coefficients,
            } => {
                let mut val = f32x4::splat(0.0);
                let tmp_lambda = lambda - *xoffset;
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += coef * tmp_lambda.powf(f32x4::splat(i as f32));
                }
                val
            }

            Curve::Cauchy { a, b } => *a + *b / (lambda * lambda),
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
                    *boost * bbd / blackbody(*temperature, max_blackbody_lambda(*temperature))
                }
            }
            _ => f32x4::new(
                self.evaluate(lambda.extract(0)),
                self.evaluate(lambda.extract(1)),
                self.evaluate(lambda.extract(2)),
                self.evaluate(lambda.extract(3)),
            ),
        }
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
                    ws.replace_energy(self.evaluate(ws.lambda)),
                    PDF::from(1.0 / wavelength_range.span()), // uniform distribution
                )
            }
        }
    }
    fn sample_power_and_pdf_hero(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (HeroWavelength, PDFx4) {
        // since this is being called on an SPD arbitrary, there's no tabulated CDF data to indicate where peaks are in this distribution.
        // could sample using stochastic MIS or something to adjust the pdf, but for now use uniform sampling.
        match &self {
            _ => {
                let ws = HeroWavelength::new_from_range(sample.x, wavelength_range);
                (
                    ws.replace_energy(self.evaluate_power_hero(ws.lambda)),
                    PDFx4::from(f32x4::splat(1.0 / wavelength_range.span())), // uniform distribution
                )
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CurveWithCDF {
    // pdf range is [0, infinity), though actual infinite values are not handled yet, and if they were it would be through special handling as dirac delta distributions
    pub pdf: Curve,
    // cdf ranges from 0 to 1
    pub cdf: Curve,
    // store pdf integral so that we don't have to normalize the `pdf` curve beforehand. instead, all samplings of the pdf when taken through the cdf should be normalized by dividing by pdf_integral.
    pub pdf_integral: f32,
}

impl SpectralPowerDistributionFunction for CurveWithCDF {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda)
    }
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda).min(ONE_SUB_EPSILON)
    }
    fn evaluate_power_hero(&self, lambda: f32x4) -> f32x4 {
        self.pdf.evaluate_power_hero(lambda)
    }
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (SingleWavelength, PDF) {
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
    fn sample_power_and_pdf_hero(
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
    }
}

#[cfg(test)]
mod test {
    use crate::{spectral::BOUNDED_VISIBLE_RANGE, Sample1D, Sampler, StratifiedSampler};

    use super::*;

    #[test]
    fn test_y_bar_spd() {
        let spd = Curve::y_bar();
        assert!(spd.evaluate(550.0) == 0.99955124);
    }
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
            let sampled =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
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
            let sampled =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
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
            let sampled = cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
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
            let sampled = cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
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
            let sampled = combined_cdf
                .sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += sampled.0.energy.0 / (sampled.1).0;
        }
        println!("\n\n{} {}", s / 1000.0, combined_cdf.pdf_integral);
    }

    #[test]
    fn test_from_func() {
        let bounds = Bounds1D::new(0.0, 1.0);
        let curve = Curve::from_function(|x| x * x, 100, bounds);

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
        let curve = Curve::from_function(|x| x * x, 100, bounds);

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
            estimate += v.energy.0 / pdf.0 / samples as f32;
            println!("{}, {}, {}", v.lambda, v.energy.0, pdf.0);
            variance_pt_1 += (v.energy.0 / pdf.0).powi(2) / samples as f32;
        }
        println!("estimate = {}, true_integral = {}", estimate, true_integral);
        println!(
            "variance = {:?}",
            (variance_pt_1 - estimate.powi(2)) / (samples - 1) as f32
        );
        println!("lowest sample is {}", min_sample_x);
    }
}
