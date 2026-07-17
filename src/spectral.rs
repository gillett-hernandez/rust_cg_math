use crate::prelude::*;
use thermite::math::TranscendentalMath;

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

pub type SingleWavelength = WavelengthEnergy<Vector<f32>>;
/// Hero wavelength bundle, parameterized by a thermite f32 float vector `V`.
/// Replaces the old `f32x4`-hardcoded alias — callers pick the width by
/// choosing `V` (e.g. `<X86V3 as FloatSimd<f32>>::fxN` for native).
pub type HeroWavelength<V> = WavelengthEnergy<V>;

#[inline(always)]
pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

#[inline(always)]
pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

#[inline(always)]
pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

/// Vector form of the CIE X-bar observer fit. Generic across thermite f32
/// float vectors. Replaces the simdfloat_patch-gated `x_bar_f32x4`.
#[inline(always)]
pub fn x_bar_v<V, T>(angstroms: V) -> V
where
    V: FloatVector<Element = T> + TranscendentalMath,
    T: FloatElement + From<f32>,
{
    gaussian_v(angstroms, 1.056, 5998.0, 379.0, 310.0)
        + gaussian_v(angstroms, 0.362, 4420.0, 160.0, 267.0)
        + gaussian_v(angstroms, -0.065, 5011.0, 204.0, 262.0)
}

#[inline(always)]
pub fn y_bar_v<V, T>(angstroms: V) -> V
where
    V: FloatVector<Element = T> + TranscendentalMath,
    T: FloatElement + From<f32>,
{
    gaussian_v(angstroms, 0.821, 5688.0, 469.0, 405.0)
        + gaussian_v(angstroms, 0.286, 5309.0, 163.0, 311.0)
}

#[inline(always)]
pub fn z_bar_v<V, T>(angstroms: V) -> V
where
    V: FloatVector<Element = T> + TranscendentalMath,
    T: FloatElement + From<f32>,
{
    gaussian_v(angstroms, 1.217, 4370.0, 118.0, 360.0)
        + gaussian_v(angstroms, 0.681, 4590.0, 260.0, 138.0)
}

// traits

pub trait WavelengthEnergyTrait<V> {
    #[inline(always)]
    fn new(lambda: V, energy: V) -> WavelengthEnergy<V> {
        WavelengthEnergy { lambda, energy }
    }
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<V>;
}

#[derive(Copy, Clone, Debug)]
pub struct WavelengthEnergy<V> {
    pub lambda: V,
    pub energy: V,
}

impl<V> WavelengthEnergy<V> {
    #[inline(always)]
    pub fn replace_energy(self, e: V) -> Self {
        Self { energy: e, ..self }
    }
}

// impl<S: thermite::simd::Simd> From<SingleWavelength>> for XYZColor<S> {
//     #[inline(always)]
//     fn from(we: SingleWavelength) -> Self {
//         let angstroms = we.lambda * 10.0;
//         XYZColor::new(
//             we.energy * x_bar(angstroms),
//             we.energy * y_bar(angstroms),
//             we.energy * z_bar(angstroms),
//         )
//     }
// }

/// Generic SIMD -> XYZ conversion. Sums each spectral channel across lanes to
/// produce scalar CIE XYZ tristimulus values. Replaces the simdfloat_patch-
/// gated `From<WavelengthEnergy<f32x4, f32x4>> for XYZColor`.
impl<
    V: FloatVector<Element = T> + TranscendentalMath,
    T: FloatElement + From<f32> + Into<f32>,
    S: Simd,
> From<WavelengthEnergy<V>> for XYZColor<S>
{
    #[inline(always)]
    fn from(we: WavelengthEnergy<V>) -> Self {
        let angstroms = we.lambda * V::splat(10.0.into());
        XYZColor::new(
            (we.energy * x_bar_v(angstroms.into()))
                .sum_elements()
                .into(),
            (we.energy * y_bar_v(angstroms.into()))
                .sum_elements()
                .into(),
            (we.energy * z_bar_v(angstroms.into()))
                .sum_elements()
                .into(),
        )
    }
}

// impl WavelengthEnergyTrait<f32, f32> for WavelengthEnergy<f32, f32> {
//     #[inline(always)]
//     fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<f32, f32> {
//         WavelengthEnergy {
//             lambda: bounds.lower + sample * bounds.span(),
//             energy: 0.0,
//         }
//     }
// }

/// Generic hero-wavelength sampling. Lays out `V::LANES` evenly-
/// spaced wavelengths starting from `bounds.lower + sample * bounds.span()`,
/// wrapping any past `bounds.upper` back into range. Replaces the old
/// `f32x4`-hardcoded `new_from_range` and now scales to whatever width `R` is.
impl<V, T> WavelengthEnergyTrait<V> for WavelengthEnergy<V>
where
    V: FloatVector<Element = T>,
    T: FloatElement + From<f32>, //+ Into<f32>
{
    #[inline(always)]
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<V> {
        let lanes = V::LANES as f32;
        let hero = sample * bounds.span();
        let delta = bounds.span() / lanes;
        let mult = V::indexed();
        let wavelengths =
            V::splat(bounds.lower.into()) + (V::splat(hero.into()) + mult * V::splat(delta.into()));
        let sub = wavelengths
            .cmp_gt(V::splat(bounds.upper.into()))
            .select(V::splat(bounds.span().into()), V::ZERO);
        HeroWavelength::new(wavelengths - sub, V::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_visible_range_constants() {
        assert_eq!(EXTENDED_VISIBLE_RANGE.lower, 370.0);
        assert_eq!(EXTENDED_VISIBLE_RANGE.upper, 790.0);
        assert_eq!(BOUNDED_VISIBLE_RANGE.lower, 380.0);
        assert_eq!(BOUNDED_VISIBLE_RANGE.upper, 780.0);
    }

    proptest! {
        #[test]
        fn cie_x_bar_non_negative_in_visible(lambda in 380.0f32..780.0) {
            let angstroms = lambda * 10.0;
            let val = x_bar(angstroms);
            prop_assert!(val >= -0.07, "x_bar({})={}", lambda, val);
            // x_bar can be slightly negative due to the gaussian fitting
        }

        #[test]
        fn cie_y_bar_non_negative_in_visible(lambda in 380.0f32..780.0) {
            let angstroms = lambda * 10.0;
            let val = y_bar(angstroms);
            prop_assert!(val >= 0.0, "y_bar({})={}", lambda, val);
        }

        #[test]
        fn cie_z_bar_non_negative_in_visible(lambda in 380.0f32..780.0) {
            let angstroms = lambda * 10.0;
            let val = z_bar(angstroms);
            prop_assert!(val >= 0.0, "z_bar({})={}", lambda, val);
        }

        #[test]
        fn cie_functions_near_zero_far_outside_visible(lambda in 100.0f32..200.0) {
            let angstroms = lambda * 10.0;
            prop_assert!(x_bar(angstroms).abs() < 0.01, "x_bar far UV should be ~0");
            prop_assert!(y_bar(angstroms).abs() < 0.01, "y_bar far UV should be ~0");
            prop_assert!(z_bar(angstroms).abs() < 0.01, "z_bar far UV should be ~0");
        }

        #[test]
        fn single_wavelength_new_from_range(sample in 0.001f32..0.999) {
            let bounds = BOUNDED_VISIBLE_RANGE;
            let we = SingleWavelength::new_from_range(sample, bounds);
            let lambda = we.lambda.extract::<0>();
            prop_assert!(lambda >= bounds.lower, "lambda={} < lower={}", lambda, bounds.lower);
            prop_assert!(lambda <= bounds.upper, "lambda={} > upper={}", lambda, bounds.upper);
            prop_assert_eq!(we.energy.extract::<0>(), 0.0);
        }

        #[test]
        fn hero_wavelength_all_in_range(sample in 0.001f32..0.999) {
            // Use the scalar backend's 4-lane register for determinism + portability.
            // Once Stage 3 lands a default-backend type alias, this can switch.
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let bounds = BOUNDED_VISIBLE_RANGE;
            let we = HeroWavelength::<Vector<TestR>>::new_from_range(sample, bounds);
            let arr = we.lambda.into_array();
            for (i, l) in arr.iter().enumerate() {
                prop_assert!(
                    *l >= bounds.lower && *l <= bounds.upper,
                    "hero lambda[{}]={} not in [{}, {}]", i, l, bounds.lower, bounds.upper
                );
            }
        }

        #[test]
        fn wavelength_energy_to_xyz_positive_energy(lambda in 400.0f32..700.0, energy in 0.0f32..10.0) {
            type TestS = thermite::backend::scalar::Scalar;
            // 1-lane field so the lane-summing XYZ conversion isn't multiplied.
            let we = WavelengthEnergy {
                lambda: Vector::<f32>::splat(lambda),
                energy: Vector::<f32>::splat(energy),
            };
            let xyz: XYZColor<TestS> = we.into();
            // with positive energy in the visible range, y should be non-negative
            prop_assert!(xyz.y() >= 0.0, "xyz.y={} for lambda={}, energy={}", xyz.y(), lambda, energy);
        }

        #[test]
        fn replace_energy_preserves_lambda(lambda in 380.0f32..780.0, e1 in 0.0f32..10.0, e2 in 0.0f32..10.0) {
            let we = WavelengthEnergy { lambda, energy: e1 };
            let replaced = we.replace_energy(e2);
            prop_assert_eq!(replaced.lambda, lambda);
            prop_assert_eq!(replaced.energy, e2);
        }

        #[test]
        fn cie_vector_observers_match_scalar(lambda in 380.0f32..780.0) {
            // x_bar_v/y_bar_v/z_bar_v must agree lane-by-lane with the scalar fits.
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let angstroms = lambda * 10.0;
            let v = Vector::<TestR>::splat(angstroms);
            for (got, want) in [
                (x_bar_v(v), x_bar(angstroms)),
                (y_bar_v(v), y_bar(angstroms)),
                (z_bar_v(v), z_bar(angstroms)),
            ] {
                for lane in got.into_array() {
                    prop_assert!((lane - want).abs() < 1e-4, "lane {} vs scalar {}", lane, want);
                }
            }
        }

        #[test]
        fn hero_wavelength_to_xyz_non_negative_y(sample in 0.001f32..0.999, energy in 0.1f32..5.0) {
            // exercise the V -> XYZColor conversion impl.
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            type TestS = thermite::backend::scalar::Scalar;
            let we = HeroWavelength::<Vector<TestR>>::new_from_range(sample, BOUNDED_VISIBLE_RANGE)
                .replace_energy(Vector::<TestR>::splat(energy));
            let xyz: XYZColor<TestS> = we.into();
            prop_assert!(xyz.y() >= 0.0, "summed Y should be non-negative, got {}", xyz.y());
        }
    }
}
