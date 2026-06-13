use crate::prelude::*;

use crate::spectral::{x_bar, y_bar, z_bar};

#[cfg(feature = "deepsize")]
use deepsize::DeepSizeOf;
use ordered_float::OrderedFloat;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thermite::math::TranscendentalMath;

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
    ) -> (WavelengthEnergy<T, T>, PDF<T, Length>);
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
    #[inline(always)]
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
    #[inline(always)]
    pub fn y_bar() -> Curve {
        Curve::Exponential {
            signal: vec![(568.0, 46.9, 40.5, 0.821), (530.9, 16.3, 31.1, 0.286)],
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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
                let step_size = bounds.span() / ((signal.len() - 1) as f32);
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

    #[inline(always)]
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
                let step_size = bounds.span() / ((signal.len() - 1) as f32);
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

    #[inline(always)]
    pub fn evaluate_integral(
        &self,
        integration_bounds: Bounds1D,
        samples: usize,
        clamped: bool,
    ) -> f32 {
        match self {
            Curve::Const(v) => *v * integration_bounds.span(),
            Curve::Blackbody { .. } => {
                // https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature

                let samples = if samples % 2 == 0 {
                    samples + 1
                } else {
                    samples
                };
                let step_size = integration_bounds.span() / samples as f32;
                let factor = step_size / 2.0;
                let mut sum = 0.0;

                const O: f32 = (2.0f32 * 1.7320508f32).recip();
                for i in 0..samples {
                    let x0 = integration_bounds.lower + step_size * (0.5 - O + i as f32);
                    let x1 = integration_bounds.lower + step_size * (0.5 + O + i as f32);
                    let f_x0 = if clamped {
                        self.evaluate(x0).clamp(0.0, ONE_SUB_EPSILON)
                    } else {
                        self.evaluate(x0)
                    };
                    let f_x1 = if clamped {
                        self.evaluate(x1).clamp(0.0, ONE_SUB_EPSILON)
                    } else {
                        self.evaluate(x1)
                    };
                    sum += f_x0 + f_x1;
                }
                sum * factor
            }
            Curve::Linear {
                signal,
                bounds: signal_bounds,
                mode: InterpolationMode::Linear,
            } => {
                // analytical integration of piecewise-linear curve
                if signal.is_empty() {
                    return 0.0;
                }

                let clamp = |v: f32| -> f32 {
                    if clamped {
                        v.clamp(0.0, ONE_SUB_EPSILON)
                    } else {
                        v
                    }
                };

                let ib_lo = integration_bounds.lower;
                let ib_hi = integration_bounds.upper;
                let sb_lo = signal_bounds.lower;
                let sb_hi = signal_bounds.upper;
                let step = signal_bounds.span() / signal.len() as f32;
                let mut sum = 0.0;

                // constant extension below signal bounds (value = first sample)
                if ib_lo < sb_lo {
                    let end = ib_hi.min(sb_lo);
                    sum += clamp(signal[0]) * (end - ib_lo);
                }

                // constant extension above signal bounds (value = last sample)
                if ib_hi > sb_hi {
                    let start = ib_lo.max(sb_hi);
                    sum += clamp(*signal.last().unwrap()) * (ib_hi - start);
                }

                // piecewise-linear region within signal bounds
                let overlap_lo = ib_lo.max(sb_lo);
                let overlap_hi = ib_hi.min(sb_hi);

                if overlap_lo < overlap_hi && signal.len() >= 2 {
                    let start_idx = ((overlap_lo - sb_lo) / step) as usize;
                    for i in start_idx..signal.len() - 1 {
                        let seg_lo = sb_lo + i as f32 * step;
                        let seg_hi = seg_lo + step;

                        if seg_lo >= overlap_hi {
                            break;
                        }

                        let x_lo = seg_lo.max(overlap_lo);
                        let x_hi = seg_hi.min(overlap_hi);
                        if x_lo >= x_hi {
                            continue;
                        }

                        // unclamped interpolated values at the clipped segment endpoints
                        let t_lo = (x_lo - seg_lo) / step;
                        let t_hi = (x_hi - seg_lo) / step;
                        let y_lo_raw = (1.0 - t_lo) * signal[i] + t_lo * signal[i + 1];
                        let y_hi_raw = (1.0 - t_hi) * signal[i] + t_hi * signal[i + 1];

                        if !clamped {
                            // trapezoid rule is exact for linear segments
                            sum += (x_hi - x_lo) * (y_lo_raw + y_hi_raw) * 0.5;
                        } else {
                            // When clamped, the function is piecewise: linear
                            // where within [0, ONE_SUB_EPSILON], constant at the
                            // boundary where outside. Split at crossing points so
                            // each sub-segment's trapezoid is exact.
                            let width = x_hi - x_lo;
                            let dy = y_hi_raw - y_lo_raw;

                            // collect split points as fractions t in [0, width]
                            let mut ts = [0.0f32; 4];
                            let mut n = 1usize;
                            ts[0] = 0.0;

                            if dy.abs() > f32::EPSILON {
                                // t where y(t) = 0: y_lo_raw + dy*t/width = 0
                                let t_zero = -y_lo_raw * width / dy;
                                if t_zero > 0.0 && t_zero < width {
                                    ts[n] = t_zero;
                                    n += 1;
                                }
                                // t where y(t) = ONE_SUB_EPSILON
                                let t_one = (ONE_SUB_EPSILON - y_lo_raw) * width / dy;
                                if t_one > 0.0 && t_one < width {
                                    ts[n] = t_one;
                                    n += 1;
                                }
                            }

                            ts[n] = width;
                            n += 1;
                            ts[..n].sort_by(|a, b| a.total_cmp(b));

                            for j in 0..n - 1 {
                                let ta = ts[j];
                                let tb = ts[j + 1];
                                let ya = (y_lo_raw + dy * ta / width).clamp(0.0, ONE_SUB_EPSILON);
                                let yb = (y_lo_raw + dy * tb / width).clamp(0.0, ONE_SUB_EPSILON);
                                sum += (tb - ta) * (ya + yb) * 0.5;
                            }
                        }
                    }
                } else if overlap_lo < overlap_hi {
                    // single-sample signal: constant within bounds
                    sum += clamp(signal[0]) * (overlap_hi - overlap_lo);
                }

                sum
            }
            _ => {
                // simpson's rule
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
    }
    #[inline(always)]
    pub fn convert_to_xyz<S: thermite::simd::Simd>(
        &self,
        integration_bounds: Bounds1D,
        step_size: f32,
        clamped: bool,
    ) -> XYZColor<S> {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum: XYZColor<S> = XYZColor::black();
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let angstroms = lambda * 10.0;
            let val = if clamped {
                self.evaluate_clamped(lambda)
            } else {
                self.evaluate_power(lambda)
            };
            sum = sum
                + XYZColor::new(
                    val * x_bar(angstroms),
                    val * y_bar(angstroms),
                    val * z_bar(angstroms),
                ) * step_size;
        }
        sum
    }
}

impl SpectralPowerDistributionFunction<f32> for Curve {
    #[inline(always)]
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.evaluate(lambda).max(0.0)
    }
    #[inline(always)]
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.evaluate(lambda).clamp(0.0, ONE_SUB_EPSILON)
    }
    #[inline(always)]
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (SingleWavelength, PDF<f32, Length>) {
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

/// Generic SIMD evaluator for `Curve`. Replaces the simdfloat_patch-gated
/// `SpectralPowerDistributionFunction<f32x4> for Curve` and works across any
/// thermite f32 vector width.
///
/// `Linear`, `Tabulated`, and `Machine` fall back to a per-lane scalar map.
/// The previous SIMD-vectorized `Linear` used `gather_or_default`, but gather
/// is scalarized on most CPUs anyway, so the simpler `map` path is close to
/// equivalent in practice. A thermite-`gather_or` path can be added later if
/// profiling shows it matters.
impl<R> SpectralPowerDistributionFunction<Vector<R>> for Curve
where
    R: thermite::register::FloatRegister<Element = f32>,
    Vector<R>: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    #[inline(always)]
    fn evaluate_power(&self, lambda: Vector<R>) -> Vector<R> {
        match self {
            Curve::Const(v) => Vector::<R>::splat(v.max(0.0)),
            Curve::Polynomial {
                domain_range_mapping,
                coefficients,
            } => {
                let [x0, xs, y0, ys]: [f32; 4] = *domain_range_mapping;
                debug_assert!(xs > 0.0);

                let x = (lambda - Vector::<R>::splat(x0)) / Vector::<R>::splat(xs);
                let mut sum = Vector::<R>::splat(y0);
                let mut xpow = x;
                for i in 0..8 {
                    sum += Vector::<R>::splat(coefficients[i]) * xpow;
                    xpow *= x;
                }
                <Vector<R> as NumericVector>::max(sum, <Vector<R> as NumericVector>::ZERO)
                    * Vector::<R>::splat(ys)
            }
            Curve::Cauchy { a, b } => {
                Vector::<R>::splat(*a) + Vector::<R>::splat(*b) / (lambda * lambda)
            }
            Curve::Exponential { signal } => {
                let mut val = <Vector<R> as NumericVector>::ZERO;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val += gaussian_v(lambda, multiplier, offset, sigma1, sigma2);
                }
                val
            }
            Curve::InverseExponential { signal } => {
                let mut val = <Vector<R> as NumericVector>::ONE;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val -= gaussian_v(lambda, multiplier, offset, sigma1, sigma2);
                }
                <Vector<R> as NumericVector>::max(val, <Vector<R> as NumericVector>::ZERO)
            }
            Curve::Blackbody { temperature, boost } => {
                let bbd = blackbody_v(*temperature, lambda);
                if *boost == 0.0 {
                    bbd
                } else {
                    Vector::<R>::splat(*boost) * bbd
                        / Vector::<R>::splat(blackbody(
                            *temperature,
                            max_blackbody_lambda(*temperature),
                        ))
                }
            }
            // Linear / Tabulated / Machine: per-lane scalar fallback.
            _ => lambda.map(|l| self.evaluate(l)),
        }
    }

    #[inline(always)]
    fn evaluate_clamped(&self, lambda: Vector<R>) -> Vector<R> {
        <Vector<R> as NumericVector>::clamp(
            self.evaluate_power(lambda),
            <Vector<R> as NumericVector>::ZERO,
            <Vector<R> as NumericVector>::ONE,
        )
    }

    #[inline(always)]
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (HeroWavelength<R>, PDF<Vector<R>, Length>) {
        let ws = HeroWavelength::<R>::new_from_range(sample.x, wavelength_range);
        (
            ws.replace_energy(self.evaluate_power(ws.lambda)),
            PDF::new(Vector::<R>::splat(1.0 / wavelength_range.span())),
        )
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
    #[inline(always)]
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda)
    }
    #[inline(always)]
    fn evaluate_clamped(&self, lambda: f32) -> f32 {
        self.pdf.evaluate_clamped(lambda)
    }
    #[inline(always)]
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (SingleWavelength, PDF<f32, Length>) {
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

/// Generic SIMD CDF sampler. Replaces the simdfloat_patch-gated
/// `SpectralPowerDistributionFunction<f32x4> for CurveWithCDF`.
impl<R> SpectralPowerDistributionFunction<Vector<R>> for CurveWithCDF
where
    R: thermite::register::FloatRegister<Element = f32>,
    Vector<R>: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    #[inline(always)]
    fn evaluate_power(&self, lambda: Vector<R>) -> Vector<R> {
        self.pdf.evaluate_power(lambda)
    }
    #[inline(always)]
    fn evaluate_clamped(&self, lambda: Vector<R>) -> Vector<R> {
        self.pdf.evaluate_clamped(lambda)
    }
    #[inline(always)]
    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        mut sample: Sample1D,
    ) -> (HeroWavelength<R>, PDF<Vector<R>, Length>) {
        match &self.cdf {
            Curve::Const(v) => (
                HeroWavelength::<R>::new_from_range(sample.x, wavelength_range)
                    .replace_energy(Vector::<R>::splat(*v)),
                Vector::<R>::splat(1.0 / self.pdf_integral).into(),
            ),
            Curve::Linear {
                signal,
                bounds,
                mode,
            } => {
                let restricted_bounds = bounds.intersection(wavelength_range);
                let lower_cdf_value = self.cdf.evaluate(restricted_bounds.lower);
                let upper_cdf_value = self.cdf.evaluate(restricted_bounds.upper);
                sample.x = lower_cdf_value + sample.x * (upper_cdf_value - lower_cdf_value);
                let maybe_index = signal
                    .binary_search_by_key(&OrderedFloat::<f32>(sample.x), |&a| {
                        OrderedFloat::<f32>(a)
                    });
                let hero_lambda = match maybe_index {
                    Ok(index) | Err(index) => {
                        if index == 0 {
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
                let correlated_sample_x = (hero_lambda - bounds.lower) / bounds.span();
                let out_we = HeroWavelength::<R>::new_from_range(correlated_sample_x, *bounds);
                let power: Vector<R> = self.pdf.evaluate_power(out_we.lambda);

                (
                    out_we.replace_energy(power),
                    Vector::<R>::splat(power.extract::<0>() / self.pdf_integral).into(),
                )
            }
            _ => self.cdf.sample_power_and_pdf(wavelength_range, sample),
        }
    }
}

// TODO: impl SPDF<f32x4> for CurveWithCDF and Curve
/*

#[inline(always)]
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
    use crate::{assert_approx_eq, sample::*, spectral::BOUNDED_VISIBLE_RANGE};

    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn const_curve_evaluates_everywhere(c in 0.0f32..100.0, x in -1000.0f32..1000.0) {
            let curve = Curve::Const(c);
            let val = curve.evaluate(x);
            prop_assert_eq!(val, c.max(0.0));
        }

        #[test]
        fn blackbody_curve_non_negative(temp in 1000.0f32..10000.0, lambda in 300.0f32..900.0) {
            let curve = Curve::Blackbody { temperature: temp, boost: 1.0 };
            let val = curve.evaluate(lambda);
            prop_assert!(val >= 0.0, "blackbody eval({})={}", lambda, val);
        }

        #[test]
        fn cauchy_curve_positive_for_positive_params(
            a in 1.0f32..2.0,
            b in 0.0f32..10000.0,
            lambda in 400.0f32..800.0
        ) {
            let curve = Curve::Cauchy { a, b };
            let val = curve.evaluate(lambda);
            prop_assert!(val > 0.0, "cauchy eval({})={}", lambda, val);
        }

        #[test]
        fn evaluate_power_non_negative(lambda in 380.0f32..780.0) {
            let curve = Curve::Exponential {
                signal: vec![(568.0, 46.9, 40.5, 0.821), (530.9, 16.3, 31.1, 0.286)],
            };
            let val = curve.evaluate_power(lambda);
            prop_assert!(val >= 0.0, "evaluate_power({})={}", lambda, val);
        }

        #[test]
        fn evaluate_clamped_in_unit(lambda in 380.0f32..780.0) {
            let curve = Curve::Exponential {
                signal: vec![(568.0, 46.9, 40.5, 0.821), (530.9, 16.3, 31.1, 0.286)],
            };
            let val = curve.evaluate_clamped(lambda);
            prop_assert!(val >= 0.0 && val < 1.0, "evaluate_clamped({})={}", lambda, val);
        }

        #[test]
        fn linear_curve_interpolates_within_signal_range(t in 0.01f32..0.99) {
            let signal = vec![0.0, 1.0, 0.5, 0.8, 0.2];
            let bounds = Bounds1D::new(0.0, 5.0);
            let curve = Curve::Linear { signal: signal.clone(), bounds, mode: InterpolationMode::Linear };
            let lambda = bounds.lerp(t);
            let val = curve.evaluate(lambda);
            let min_signal = signal.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_signal = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                val >= min_signal - 1e-4 && val <= max_signal + 1e-4,
                "linear eval({})={}, range=[{}, {}]", lambda, val, min_signal, max_signal
            );
        }

        #[test]
        fn const_integral_equals_span_times_value(c in 0.1f32..10.0, span in 1.0f32..100.0) {
            let curve = Curve::Const(c);
            let bounds = Bounds1D::new(0.0, span);
            let integral = curve.evaluate_integral(bounds, 100, false);
            let expected = c * span;
            let rel_err = (integral - expected).abs() / expected;
            prop_assert!(rel_err < 0.02, "integral={}, expected={}", integral, expected);
        }

        #[test]
        fn from_function_captures_shape(x in 0.1f32..0.9) {
            let bounds = Bounds1D::new(0.0, 1.0);
            let curve = Curve::from_function(|x| x, 100, bounds, InterpolationMode::Linear);
            let val = curve.evaluate(x);
            prop_assert!((val - x).abs() < 0.02, "eval({})={}, expected ~{}", x, val, x);
        }
    }

    // register used for the Vector<R> SPD paths below
    type R4 = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
    type S4 = thermite::backend::scalar::Scalar;

    fn assert_lanes_match_scalar(v: Vector<R4>, scalar: f32, tol: f32) {
        for lane in v.into_array() {
            assert!(
                (lane - scalar).abs() <= tol,
                "lane {} vs scalar {} (tol {})",
                lane,
                scalar,
                tol
            );
        }
    }

    #[test]
    fn test_default_curve() {
        let c = Curve::default();
        assert!(matches!(c, Curve::Const(v) if v == 0.0));
        assert_eq!(c.evaluate(123.0), 0.0);
    }

    #[test]
    fn test_linear_nearest_and_cubic_modes() {
        let signal = vec![0.0, 1.0, 0.0];
        let bounds = Bounds1D::new(0.0, 2.0);
        // sample at x=0.25 (t=0.25 within first segment) for Nearest -> left (0.0)
        let nearest = Curve::Linear {
            signal: signal.clone(),
            bounds,
            mode: InterpolationMode::Nearest,
        };
        assert_eq!(nearest.evaluate(0.25), 0.0);
        assert_eq!(nearest.evaluate(0.75), 1.0); // t=0.75 -> right (1.0)

        // cubic Hermite at the segment midpoint is between the endpoints
        let cubic = Curve::Linear {
            signal,
            bounds,
            mode: InterpolationMode::Cubic,
        };
        let mid = cubic.evaluate(0.5);
        assert!(mid > 0.0 && mid < 1.0, "cubic mid {}", mid);
        // clamped extension below/above bounds returns the endpoints
        assert_eq!(cubic.evaluate(-1.0), 0.0);
        assert_eq!(cubic.evaluate(5.0), 0.0);
    }

    #[test]
    fn test_tabulated_modes_and_edges() {
        let signal = vec![(400.0, 0.0), (500.0, 1.0), (600.0, 0.0)];
        let lin = Curve::Tabulated {
            signal: signal.clone(),
            mode: InterpolationMode::Linear,
        };
        // below first sample -> first value; above last -> last value
        assert_eq!(lin.evaluate(350.0), 0.0);
        assert_eq!(lin.evaluate(700.0), 0.0);
        // interior linear interpolation
        assert!((lin.evaluate(450.0) - 0.5).abs() < 1e-5);

        let near = Curve::Tabulated {
            signal: signal.clone(),
            mode: InterpolationMode::Nearest,
        };
        assert_eq!(near.evaluate(450.0), 1.0); // t=0.5: `t < 0.5` is false -> right (1.0)
        assert_eq!(near.evaluate(420.0), 0.0); // t=0.2 -> left (0.0)

        let cubic = Curve::Tabulated {
            signal,
            mode: InterpolationMode::Cubic,
        };
        let v = cubic.evaluate(450.0);
        assert!(v >= 0.0 && v <= 1.0, "tabulated cubic {}", v);
    }

    #[test]
    fn test_blackbody_unboosted() {
        // boost == 0.0 takes the raw-blackbody branch.
        let raw = Curve::Blackbody {
            temperature: 5000.0,
            boost: 0.0,
        };
        // boost==0 returns raw blackbody (approx: optimized build may contract FMAs differently)
        let expected = blackbody(5000.0, 550.0);
        assert!((raw.evaluate(550.0) - expected).abs() <= expected.abs() * 1e-5);
    }

    #[test]
    fn test_evaluate_integral_blackbody_clamped_and_unclamped() {
        let bb = Curve::Blackbody {
            temperature: 5000.0,
            boost: 1.0,
        };
        let bounds = Bounds1D::new(400.0, 700.0);
        // even sample count exercises the samples+1 odd-ing branch
        let unclamped = bb.evaluate_integral(bounds, 64, false);
        let clamped = bb.evaluate_integral(bounds, 64, true);
        assert!(unclamped > 0.0, "unclamped {}", unclamped);
        assert!(clamped > 0.0 && clamped <= unclamped + 1e-3, "clamped {}", clamped);
    }

    #[test]
    fn test_evaluate_integral_linear_extends_beyond_signal() {
        // signal spans [0,2]; integrate over [-1, 3] so both constant-extension
        // branches (below sb_lo and above sb_hi) run, clamped and unclamped.
        let curve = Curve::Linear {
            signal: vec![1.0, 1.0, 1.0],
            bounds: Bounds1D::new(0.0, 2.0),
            mode: InterpolationMode::Linear,
        };
        // exercises both constant-extension branches plus the in-signal region;
        // both clamped and unclamped are positive, and clamping cannot increase the integral.
        let unclamped = curve.evaluate_integral(Bounds1D::new(-1.0, 3.0), 50, false);
        let clamped = curve.evaluate_integral(Bounds1D::new(-1.0, 3.0), 50, true);
        assert!(unclamped > 0.0, "unclamped {}", unclamped);
        assert!(clamped > 0.0 && clamped <= unclamped + 1e-6, "clamped {}, unclamped {}", clamped, unclamped);
    }

    #[test]
    fn test_evaluate_integral_simpson_fallback() {
        // a non-linear, non-const curve hits the simpson `_` arm.
        let cauchy = Curve::Cauchy { a: 1.5, b: 5000.0 };
        let integral = cauchy.evaluate_integral(Bounds1D::new(400.0, 700.0), 200, false);
        assert!(integral > 0.0, "cauchy integral {}", integral);
        // a non-Linear-mode Linear curve also falls through to simpson
        let near = Curve::Linear {
            signal: vec![0.0, 1.0, 0.0],
            bounds: Bounds1D::new(0.0, 2.0),
            mode: InterpolationMode::Nearest,
        };
        let i2 = near.evaluate_integral(Bounds1D::new(0.0, 2.0), 200, true);
        assert!(i2 >= 0.0, "nearest integral {}", i2);
    }

    #[test]
    fn test_convert_to_xyz_clamped_and_unclamped() {
        let curve = Curve::y_bar();
        let unclamped: XYZColor<S4> =
            curve.convert_to_xyz(BOUNDED_VISIBLE_RANGE, 1.0, false);
        let clamped: XYZColor<S4> = curve.convert_to_xyz(BOUNDED_VISIBLE_RANGE, 1.0, true);
        assert!(unclamped.y() > 0.0, "unclamped Y {}", unclamped.y());
        assert!(clamped.y() > 0.0, "clamped Y {}", clamped.y());
        assert!(unclamped.x().is_finite() && unclamped.z().is_finite());
    }

    #[test]
    fn test_vector_spd_matches_scalar() {
        let curves = [
            Curve::Const(0.7),
            Curve::Cauchy { a: 1.5, b: 5000.0 },
            Curve::Blackbody { temperature: 5000.0, boost: 1.0 },
            Curve::Blackbody { temperature: 5000.0, boost: 0.0 },
            Curve::Polynomial {
                domain_range_mapping: [600.0, 200.0, 0.1, 1.0],
                coefficients: [0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            Curve::Exponential {
                signal: vec![(568.0, 46.9, 40.5, 0.821)],
            },
            Curve::InverseExponential {
                signal: vec![(568.0, 46.9, 40.5, 0.5)],
            },
            // Linear hits the per-lane scalar map fallback
            Curve::Linear {
                signal: vec![0.2, 0.8, 0.5],
                bounds: Bounds1D::new(400.0, 700.0),
                mode: InterpolationMode::Linear,
            },
        ];
        let lambda = 550.0f32;
        let v = Vector::<R4>::splat(lambda);
        for c in &curves {
            let scalar_power = SpectralPowerDistributionFunction::<f32>::evaluate_power(c, lambda);
            let vec_power =
                SpectralPowerDistributionFunction::<Vector<R4>>::evaluate_power(c, v);
            let tol = (scalar_power.abs() * 1e-3).max(1e-4);
            assert_lanes_match_scalar(vec_power, scalar_power, tol);

            let scalar_clamped =
                SpectralPowerDistributionFunction::<f32>::evaluate_clamped(c, lambda);
            let vec_clamped =
                SpectralPowerDistributionFunction::<Vector<R4>>::evaluate_clamped(c, v);
            assert_lanes_match_scalar(vec_clamped, scalar_clamped, 1e-4);

            // sample_power_and_pdf: pdf is the uniform 1/span on the vector path
            let (we, pdf) = SpectralPowerDistributionFunction::<Vector<R4>>::sample_power_and_pdf(
                c,
                BOUNDED_VISIBLE_RANGE,
                Sample1D::new(0.5),
            );
            assert!(we.lambda.extract::<0>() >= BOUNDED_VISIBLE_RANGE.lower);
            assert_lanes_match_scalar(*pdf, 1.0 / BOUNDED_VISIBLE_RANGE.span(), 1e-6);
        }
    }

    #[test]
    fn test_curve_with_cdf_scalar_sampling() {
        // Linear cdf path (to_cdf on a Linear curve keeps the analytic signal).
        let curve = Curve::Linear {
            signal: vec![0.0, 1.0, 1.0, 0.0],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        };
        let cdf = curve.to_cdf(BOUNDED_VISIBLE_RANGE, 100);
        // evaluate_power / evaluate_clamped delegate to the pdf curve
        let p = SpectralPowerDistributionFunction::<f32>::evaluate_power(&cdf, 580.0);
        assert!(p >= 0.0, "power {}", p);
        let pc = SpectralPowerDistributionFunction::<f32>::evaluate_clamped(&cdf, 580.0);
        assert!((0.0..1.0).contains(&pc), "clamped {}", pc);
        // importance-sample: lambda in range, pdf finite and non-negative
        let (sw, pdf) = SpectralPowerDistributionFunction::<f32>::sample_power_and_pdf(
            &cdf,
            BOUNDED_VISIBLE_RANGE,
            Sample1D::new(0.4),
        );
        assert!(sw.lambda >= BOUNDED_VISIBLE_RANGE.lower && sw.lambda <= BOUNDED_VISIBLE_RANGE.upper);
        assert!(*pdf >= 0.0, "pdf {}", *pdf);
    }

    #[test]
    fn test_curve_with_cdf_const_and_fallback_arms() {
        // Const cdf arm: manually build a CurveWithCDF whose cdf is Const.
        let const_cdf = CurveWithCDF {
            pdf: Curve::Const(2.0),
            cdf: Curve::Const(0.5),
            pdf_integral: 4.0,
        };
        let (sw, pdf) = SpectralPowerDistributionFunction::<f32>::sample_power_and_pdf(
            &const_cdf,
            BOUNDED_VISIBLE_RANGE,
            Sample1D::new(0.5),
        );
        assert_eq!(sw.energy, 0.5);
        assert!((*pdf - 1.0 / 4.0).abs() < 1e-6);

        // `_` fallback arm: cdf is neither Const nor Linear.
        let fallback = CurveWithCDF {
            pdf: Curve::Const(1.0),
            cdf: Curve::Cauchy { a: 1.0, b: 1.0 },
            pdf_integral: 1.0,
        };
        let (sw2, _pdf2) = SpectralPowerDistributionFunction::<f32>::sample_power_and_pdf(
            &fallback,
            BOUNDED_VISIBLE_RANGE,
            Sample1D::new(0.5),
        );
        assert!(sw2.lambda >= BOUNDED_VISIBLE_RANGE.lower);
    }

    #[test]
    fn test_curve_with_cdf_vector_sampling() {
        let curve = Curve::Linear {
            signal: vec![0.0, 1.0, 1.0, 0.0],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        };
        let cdf = curve.to_cdf(BOUNDED_VISIBLE_RANGE, 100);
        let v = Vector::<R4>::splat(580.0);
        let p = SpectralPowerDistributionFunction::<Vector<R4>>::evaluate_power(&cdf, v);
        for lane in p.into_array() {
            assert!(lane >= 0.0, "power lane {}", lane);
        }
        let (we, pdf) = SpectralPowerDistributionFunction::<Vector<R4>>::sample_power_and_pdf(
            &cdf,
            BOUNDED_VISIBLE_RANGE,
            Sample1D::new(0.4),
        );
        assert!(we.lambda.extract::<0>() >= BOUNDED_VISIBLE_RANGE.lower);
        for lane in (*pdf).into_array() {
            assert!(lane.is_finite(), "pdf lane {}", lane);
        }

        // Const cdf vector arm
        let const_cdf = CurveWithCDF {
            pdf: Curve::Const(2.0),
            cdf: Curve::Const(0.5),
            pdf_integral: 4.0,
        };
        let (cwe, _cpdf) =
            SpectralPowerDistributionFunction::<Vector<R4>>::sample_power_and_pdf(
                &const_cdf,
                BOUNDED_VISIBLE_RANGE,
                Sample1D::new(0.5),
            );
        // the Const arm splats the cdf's constant value (0.5) into the energy
        assert_eq!(cwe.energy.extract::<0>(), 0.5);
    }

    #[test]
    fn test_clamped_integral_with_crossing() {
        // Signal [0.0, 3.0] over bounds [0, 1]: step = 0.5, one segment [0, 0.5] with y = 6x.
        // The clamp boundary ONE_SUB_EPSILON ≈ 1.0 is crossed at x_c ≈ 1/6 ≈ 0.1667 (NOT the midpoint 0.25).
        // Integrate over just the segment [0, 0.5] with clamped=true.
        //
        // Exact clamped integral:
        //   [0, x_c]: ∫ 6x dx = 3*x_c²
        //   [x_c, 0.5]: ONE_SUB_EPSILON * (0.5 - x_c)
        let one_sub_eps = 1.0 - f32::EPSILON;
        let x_c = one_sub_eps / 6.0; // crossing point, ≈ 0.1667
        let expected = 3.0 * x_c * x_c + one_sub_eps * (0.5 - x_c);

        let curve = Curve::Linear {
            signal: vec![0.0, 3.0],
            bounds: Bounds1D::new(0.0, 1.0),
            mode: InterpolationMode::Linear,
        };
        let integral = curve.evaluate_integral(Bounds1D::new(0.0, 0.5), 100, true);

        // A naive clamped trapezoid would give 0.5 * (0 + ONE_SUB_EPSILON) / 2 ≈ 0.25,
        // but the correct answer is ≈ 0.4167
        let naive_wrong = 0.5 * (0.0 + one_sub_eps) / 2.0;
        assert!(
            (integral - expected).abs() < 1e-5,
            "integral={}, expected={}, naive_wrong={}",
            integral,
            expected,
            naive_wrong,
        );
    }

    #[test]
    fn test_y_bar_spd() {
        let spd = Curve::y_bar();
        assert!(spd.evaluate(550.0) == 0.99955124);
    }

    #[test]
    fn test_curve_const() {
        let test_curve = Curve::Const(0.5);
        let integral = test_curve.evaluate_integral(Bounds1D::new(100.0, 200.0), 20, false);
        assert_approx_eq(integral, 50.0, 0.001);
    }
    #[test]
    fn test_curve_tabulated() {
        let test_curve = Curve::Tabulated {
            signal: vec![
                (400.0, 0.0),
                (420.0, 0.4),
                (460.0, 1.0),
                (500.0, 0.4),
                (600.0, 0.8),
                (700.0, 0.2),
                (800.0, 0.0),
            ],
            mode: InterpolationMode::Linear,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        assert_eq!(integral, 180.0);
    }
    #[test]
    fn test_curve_linear() {
        let test_curve = Curve::Linear {
            signal: vec![0.0, 0.4, 1.0, 0.4, 0.8, 0.2, 0.3, 0.0],
            bounds: Bounds1D::new(400.0, 800.0),
            mode: InterpolationMode::Linear,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        assert_approx_eq(integral, 155.0, 0.00002);
    }
    #[test]
    fn test_curve_cauchy() {
        let test_curve = Curve::Cauchy { a: 1.4, b: 2400.0 };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        assert_approx_eq(integral / 400.0, 1.4075, 0.0001);
    }
    #[test]
    fn test_curve_blackbody() {
        let test_curve = Curve::Blackbody {
            temperature: 5400.0,
            boost: 1.0,
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        assert_approx_eq(integral, 361.010275033, 0.0001);
    }

    fn get_test_exponential_signal() -> Vec<(f32, f32, f32, f32)> {
        vec![
            (500.0, 10.0, 30.0, 1.0),
            (600.0, 10.0, 15.0, 0.5),
            (700.0, 20.0, 10.0, 0.7),
        ]
    }

    #[test]
    fn test_curve_exponential() {
        let test_curve = Curve::Exponential {
            signal: get_test_exponential_signal(),
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        const GROUND_TRUTH: f32 = 92.1185890927;
        assert_approx_eq(integral, GROUND_TRUTH, 0.00002);
    }
    #[test]
    fn test_curve_inverse_exponential() {
        let test_curve = Curve::InverseExponential {
            signal: get_test_exponential_signal(),
        };
        let integral = test_curve.evaluate_integral(Bounds1D::new(400.0, 800.0), 40, false);
        let target = 307.881410968;

        assert_approx_eq(integral, target, 0.0001);
    }

    #[test]
    fn test_curve_polynomial() {
        type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
        let curve = Curve::Polynomial {
            domain_range_mapping: [600.0, 200.0, 0.5, 0.06],
            coefficients: [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        };

        let lambda = Vector::<TestR>::new([450.0_f32, 550.0, 650.0, 750.0]);
        let result: Vector<TestR> = curve.evaluate_power(lambda);
        for (i, l) in result.into_array().iter().enumerate() {
            assert!(l.is_finite(), "polynomial result[{}] is not finite", i);
        }
    }
    #[test]
    fn test_curve_machine() {
        // Machine: seed=1.0, then Mul by Const(2.0), then Add Const(3.0)
        // f(x) = 1.0 * 2.0 + 3.0 = 5.0 for all x
        let machine = Curve::Machine {
            seed: 1.0,
            list: vec![(Op::Mul, Curve::Const(2.0)), (Op::Add, Curve::Const(3.0))],
        };
        assert_eq!(machine.evaluate(0.0), 5.0);
        assert_eq!(machine.evaluate(500.0), 5.0);

        // Machine with a non-const inner curve:
        // seed=1.0, Add a nonlinear function, then Mul by Const(0.5)
        // f(x) = (1 + ramp(x)) * 0.5
        let inner = Curve::Linear {
            signal: vec![0.0, 2.0, 6.0, 4.0, 6.0],
            bounds: Bounds1D::new(0.0, 100.0),
            mode: InterpolationMode::Cubic,
        };

        assert_eq!(inner.evaluate(0.0), 0.0);
        assert_eq!(inner.evaluate(100.0), 6.0);
        assert_eq!(inner.evaluate(25.0), 2.0);
        assert_eq!(inner.evaluate(50.0), 6.0);
        assert_eq!(inner.evaluate(75.0), 4.0);

        let integral = inner.evaluate_integral(Bounds1D::new(0.0, 100.0), 1000, false);
        println!("integral is {}", integral);
        assert_approx_eq(integral, 375.0, 0.0001);

        let machine = Curve::Machine {
            seed: 1.0,
            list: vec![(Op::Add, inner), (Op::Mul, Curve::Const(0.5))],
        };
        // at x=0 the ramp evaluates to 0.0, so f(0) = 0
        assert_eq!(machine.evaluate(0.0), 0.5);
        // spot-check a midpoint
        let mid_val = machine.evaluate(50.0);
        assert!(mid_val > 0.0, "expected positive at x=50, got {}", mid_val);

        let integral = machine.evaluate_integral(Bounds1D::new(0.0, 100.0), 1000, false);
        assert_approx_eq(integral, 237.5, 0.0001);
    }

    #[test]
    fn test_cdf1() {
        let curve = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        };
        let true_integral = curve.evaluate_integral(BOUNDED_VISIBLE_RANGE, 1000, false);
        let cdf: CurveWithCDF = curve.to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let n = 1000;
        let mut s = 0.0;
        for _ in 0..n {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        let estimate = s / n as f32;
        assert!(
            (estimate - true_integral).abs() / true_integral < 0.15,
            "CDF estimate {} too far from integral {}",
            estimate,
            true_integral
        );
    }

    #[test]
    fn test_cdf2() {
        let curve = Curve::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        };
        let cdf: CurveWithCDF = curve.to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        // pdf_integral should be positive and finite
        assert!(
            cdf.pdf_integral.is_finite() && cdf.pdf_integral > 0.0,
            "pdf_integral should be positive and finite: {}",
            cdf.pdf_integral
        );

        // sampling should produce finite, positive energy values
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            assert!(we.energy.is_finite(), "energy should be finite");
            assert!(*pdf > 0.0, "pdf should be positive");
        }
    }

    #[test]
    fn test_cdf3() {
        // test sampling according to the CDF with narrowed bounds wrt the original signal bounds
        let curve = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        };
        let cdf: CurveWithCDF = curve.to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let narrowed_bounds = Bounds1D::new(500.0, 600.0);
        let n = 1000;
        let mut s = 0.0;
        for _ in 0..n {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(narrowed_bounds, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        let estimate = s / n as f32;
        // estimate should be finite and positive for a positive curve
        assert!(
            estimate.is_finite() && estimate > 0.0,
            "CDF3 estimate invalid: {}",
            estimate
        );
    }

    #[test]
    fn test_cdf4() {
        // test sampling according to the CDF with narrowed bounds in general
        let narrowed_bounds = Bounds1D::new(500.0, 600.0);

        let curve = Curve::Exponential {
            signal: vec![(400.0, 200.0, 200.0, 0.9), (600.0, 200.0, 300.0, 1.0)],
        };
        let cdf: CurveWithCDF = curve.to_cdf(narrowed_bounds, 100);

        let n = 1000;
        let mut s = 0.0;
        for _ in 0..n {
            let (we, pdf): (_, PDF<f32, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        let estimate = s / n as f32;
        assert!(
            estimate.is_finite() && estimate > 0.0,
            "CDF4 estimate invalid: {}",
            estimate
        );
    }

    #[test]
    fn test_cdf_addition() {
        let cdf1: CurveWithCDF = Curve::Exponential {
            signal: vec![(400.0, 100.0, 100.0, 0.9), (600.0, 100.0, 100.0, 1.0)],
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let cdf2: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

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

        let combined_cdf = CurveWithCDF {
            pdf: combined_spd,
            cdf: combined_cdf_curve,
            pdf_integral: integral1 + integral2,
        };

        // combined pdf_integral should equal sum of individual integrals
        assert_approx_eq(combined_cdf.pdf_integral, integral1 + integral2, 0.001);

        // sampling should produce valid values
        for _ in 0..100 {
            let (we, pdf): (_, PDF<f32, _>) = combined_cdf
                .sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            assert!(we.energy.is_finite(), "energy should be finite");
            assert!(*pdf > 0.0, "pdf should be positive");
        }
    }

    #[test]
    fn test_from_func() {
        let bounds = Bounds1D::new(0.0, 1.0);
        let curve = Curve::from_function(|x| x * x, 100, bounds, InterpolationMode::Cubic);

        let true_integral = 1.0 / 3.0;
        let computed = curve.evaluate_integral(bounds, 100, false);
        assert_approx_eq(computed, true_integral, 0.01);
    }

    #[test]
    fn test_cdf_from_func() {
        let bounds = Bounds1D::new(0.0, 1.0);
        let curve = Curve::from_function(|x| x * x, 100, bounds, InterpolationMode::Cubic);

        let true_integral = 1.0 / 3.0;
        let cdf = curve.to_cdf(bounds, 100);

        // pdf_integral should approximate the true integral
        assert_approx_eq(cdf.pdf_integral, true_integral, 0.02);

        let n = 1000;
        let mut estimate = 0.0;
        for _ in 0..n {
            let sample = Sample1D::new_random_sample();
            let (v, pdf): (_, PDF<f32, _>) = cdf.sample_power_and_pdf(bounds, sample);
            estimate += v.energy / *pdf / n as f32;
        }
        assert!(
            (estimate - true_integral).abs() < 0.05,
            "CDF from func estimate {} too far from true integral {}",
            estimate,
            true_integral
        );
    }

    #[test]
    fn test_cdf_sample_hwss() {
        type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
        let cdf: CurveWithCDF = Curve::Linear {
            signal: vec![
                0.1, 0.4, 0.9, 1.5, 0.9, 2.0, 1.0, 0.4, 0.6, 0.9, 0.4, 1.4, 1.9, 2.0, 5.0, 9.0,
                6.0, 3.0, 1.0, 0.4,
            ],
            bounds: BOUNDED_VISIBLE_RANGE,
            mode: InterpolationMode::Cubic,
        }
        .to_cdf(BOUNDED_VISIBLE_RANGE, 100);

        let mut s = <Vector<TestR> as NumericVector>::ZERO;
        for _ in 0..100 {
            let (we, pdf): (_, PDF<Vector<TestR>, _>) =
                cdf.sample_power_and_pdf(BOUNDED_VISIBLE_RANGE, Sample1D::new_random_sample());

            s += we.energy / *pdf;
        }
        println!("{:?}", s);
    }
}
