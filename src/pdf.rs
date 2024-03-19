use std::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Div, Mul},
};

use crate::prelude::*;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PDF<T: Field, M: Measure> {
    v: T,
    // TODO: determine if this should be PhantomData<*const M> or instead just M
    // just M would allow for M: Measure to be some arbitrary type
    measure: M,
}

impl<T: Field, M: Measure> PDF<T, M> {
    pub fn new(v: T) -> Self {
        Self {
            v,
            measure: M::default(),
        }
    }
    pub fn new_with_measure(v: T, m: M) -> Self {
        Self { v, measure: m }
    }
}

// deref, to make things easier. don't need to access pdf.0 anymore, just do *pdf
impl<T: Field, M: Measure> Deref for PDF<T, M> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T: Field, M: Measure> DerefMut for PDF<T, M> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

// impl From (and Into) when Measure can be inferred
impl<T: Field, M: Measure> From<T> for PDF<T, M> {
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

impl<T: Field, M: Measure> Add for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn add(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v + rhs.v, self.measure.combine(rhs.measure))
    }
}
impl<T: Field, M: Measure> Mul for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn mul(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v * rhs.v, self.measure.combine(rhs.measure))
    }
}
impl<T: Field, M: Measure> Div for PDF<T, M> {
    // must be under the same field and measure
    // FIXME if you divide two pdfs of the same measure, does that result in a dimensionless quantity? or is it still a pdf? not sure.
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v / rhs.v, self.measure.combine(rhs.measure))
    }
}

impl<T: Field, S: Scalar, M: Measure> Mul<S> for PDF<T, M>
where
    T: FromScalar<S>,
{
    type Output = Self;

    fn mul(self, rhs: S) -> Self::Output {
        PDF::new(self.v * T::from_scalar(rhs))
    }
}
impl<T: Field, S: Scalar, M: Measure> Div<S> for PDF<T, M>
where
    T: FromScalar<S>,
{
    type Output = Self;

    fn div(self, rhs: S) -> Self::Output {
        PDF::new(self.v / T::from_scalar(rhs))
    }
}

// special conversions
impl<T: Field> PDF<T, SolidAngle> {
    pub fn convert_to_projected_solid_angle<S: Scalar>(
        &self,
        cos_theta: S,
    ) -> PDF<T, ProjectedSolidAngle>
    where
        T: FromScalar<S>,
    {
        PDF::new(self.v * T::from_scalar(cos_theta).abs())
    }
}

impl<T: Field> PDF<T, Area> {
    pub fn convert_to_solid_angle<S: Scalar>(
        &self,
        cos_theta: S,
        distance_squared: S,
    ) -> PDF<T, SolidAngle>
    where
        T: FromScalar<S>,
    {
        PDF::new(self.v * T::from_scalar(cos_theta).abs() / T::from_scalar(distance_squared))
    }
}

impl<T: Field> PDF<T, Area> {
    pub fn convert_to_projected_solid_angle<S: Scalar>(
        &self,
        cos_i: S,
        cos_o: S,
        distance_squared: S,
    ) -> PDF<T, ProjectedSolidAngle>
    where
        T: FromScalar<S>,
    {
        // this is valid, but probably somewhat slower.
        // self.convert_to(cos_i, distance_squared).convert_to(cos_o)
        PDF::new(self.v * T::from_scalar(cos_o * cos_i).abs() / T::from_scalar(distance_squared))
    }
}

// impl<T> PDF<T, ProjectedSolidAngle> where T: Field {}
impl<T: Field> PDF<T, ProjectedSolidAngle> {
    fn convert_to_throughput(self, area_pdf: PDF<T, Area>) -> PDF<T, Throughput> {
        (*area_pdf * *self).into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    // TODO: come up with some tests that demonstrate monte carlo integration with change of variables using pdf conversions
    #[test]
    fn test_area_pdf() {
        let area_pdf_distant_object: PDF<f32, Area> = PDF::new(1.0);
        let solid_angle = area_pdf_distant_object.convert_to_solid_angle(0.5, 2.0);
        let projected_solid_angle0 =
            area_pdf_distant_object.convert_to_projected_solid_angle(0.5, 0.5, 2.0);
        let projected_solid_angle1 = solid_angle.convert_to_projected_solid_angle(0.5);

        println!("{:?}", area_pdf_distant_object);
        println!("{:?}", solid_angle);
        println!("{:?}", projected_solid_angle0);
        println!("{:?}", projected_solid_angle1);
        assert!(*projected_solid_angle0 == *projected_solid_angle1);

        let area_pdf_distant_object: PDF<f32x4, Area> =
            PDF::new(f32x4::from_array([0.1, 0.4, 0.2, 10.0]));
        let solid_angle = area_pdf_distant_object.convert_to_solid_angle(0.5, 2.0);
        let projected_solid_angle0 = area_pdf_distant_object.convert_to_projected_solid_angle(
            0.5.into(),
            0.5.into(),
            2.0.into(),
        );
        let projected_solid_angle1 = solid_angle.convert_to_projected_solid_angle(0.5.into());

        println!("{:?}", area_pdf_distant_object);
        println!("{:?}", solid_angle);
        println!("{:?}", projected_solid_angle0);
        println!("{:?}", projected_solid_angle1);
        assert!(*projected_solid_angle0 == *projected_solid_angle1);
    }
    #[test]
    fn test_solid_angle_pdf() {}
    #[test]
    fn test_projected_solid_angle_pdf() {}
}
