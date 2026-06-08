use crate::prelude::*;
use thermite::math::TranscendentalMath;

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

pub type SingleWavelength = WavelengthEnergy<f32, f32>;
/// Hero wavelength bundle, parameterized by an inner thermite::Vector type V
pub type HeroWavelength<V> = WavelengthEnergy<V, V>;

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

/// Vector form of the CIE X-bar observer fit. Generic across thermite f32
/// float vectors. Replaces the simdfloat_patch-gated `x_bar_f32x4`.
pub fn x_bar_v<V>(angstroms: V) -> V
where
    V: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    gaussian_v(angstroms, 1.056, 5998.0, 379.0, 310.0)
        + gaussian_v(angstroms, 0.362, 4420.0, 160.0, 267.0)
        + gaussian_v(angstroms, -0.065, 5011.0, 204.0, 262.0)
}

pub fn y_bar_v<V>(angstroms: V) -> V
where
    V: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    gaussian_v(angstroms, 0.821, 5688.0, 469.0, 405.0)
        + gaussian_v(angstroms, 0.286, 5309.0, 163.0, 311.0)
}

pub fn z_bar_v<V>(angstroms: V) -> V
where
    V: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    gaussian_v(angstroms, 1.217, 4370.0, 118.0, 360.0)
        + gaussian_v(angstroms, 0.681, 4590.0, 260.0, 138.0)
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

impl<S: thermite::simd::Simd> From<WavelengthEnergy<f32, f32>> for XYZColor<S> {
    fn from(we: WavelengthEnergy<f32, f32>) -> Self {
        let angstroms = we.lambda * 10.0;
        XYZColor::new(
            we.energy * x_bar(angstroms),
            we.energy * y_bar(angstroms),
            we.energy * z_bar(angstroms),
        )
    }
}

/// Generic SIMD -> XYZ conversion. Sums each spectral channel across lanes to
/// produce scalar CIE XYZ tristimulus values. Replaces the simdfloat_patch-
/// gated `From<WavelengthEnergy<f32x4, f32x4>> for XYZColor`.
impl<R, S> From<WavelengthEnergy<Vector<R>, Vector<R>>> for XYZColor<S>
where
    R: thermite::register::FloatRegister<Element = f32>,
    S: thermite::simd::Simd,
    Vector<R>: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    fn from(we: WavelengthEnergy<Vector<R>, Vector<R>>) -> Self {
        let angstroms = we.lambda * Vector::<R>::splat(10.0);
        XYZColor::new(
            (we.energy * x_bar_v(angstroms)).sum_elements(),
            (we.energy * y_bar_v(angstroms)).sum_elements(),
            (we.energy * z_bar_v(angstroms)).sum_elements(),
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

/// Generic hero-wavelength sampling. Lays out `Vector::<R>::LANES` evenly-
/// spaced wavelengths starting from `bounds.lower + sample * bounds.span()`,
/// wrapping any past `bounds.upper` back into range. Replaces the old
/// `f32x4`-hardcoded `new_from_range` and now scales to whatever width `R` is.
impl<R> WavelengthEnergyTrait<Vector<R>, Vector<R>> for WavelengthEnergy<Vector<R>, Vector<R>>
where
    R: thermite::register::FloatRegister<Element = f32>,
    Vector<R>: FloatVector<Element = f32>,
{
    fn new_from_range(sample: f32, bounds: Bounds1D) -> WavelengthEnergy<Vector<R>, Vector<R>> {
        let lanes = Vector::<R>::LANES as f32;
        let hero = sample * bounds.span();
        let delta = bounds.span() / lanes;
        let mult = Vector::<R>::indexed();
        let wavelengths = Vector::<R>::splat(bounds.lower)
            + (Vector::<R>::splat(hero) + mult * Vector::<R>::splat(delta));
        let sub = wavelengths
            .cmp_gt(Vector::<R>::splat(bounds.upper))
            .select(Vector::<R>::splat(bounds.span()), Vector::<R>::splat(0.0));
        HeroWavelength::new(wavelengths - sub, Vector::<R>::splat(0.0))
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
            prop_assert!(we.lambda >= bounds.lower, "lambda={} < lower={}", we.lambda, bounds.lower);
            prop_assert!(we.lambda <= bounds.upper, "lambda={} > upper={}", we.lambda, bounds.upper);
            prop_assert_eq!(we.energy, 0.0);
        }

        #[test]
        fn hero_wavelength_all_in_range(sample in 0.001f32..0.999) {
            // Use the scalar backend's 4-lane register for determinism + portability.
            // Once Stage 3 lands a default-backend type alias, this can switch.
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let bounds = BOUNDED_VISIBLE_RANGE;
            let we = HeroWavelength::<TestR>::new_from_range(sample, bounds);
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
            let we = WavelengthEnergy { lambda, energy };
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
    }
}
