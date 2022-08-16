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
    measure: PhantomData<*const M>,
}

impl<T: Field, M: Measure> PDF<T, M> {
    pub fn new(v: T) -> Self {
        Self {
            v,
            measure: PhantomData,
        }
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
        Self {
            v,
            measure: PhantomData,
        }
    }
}

impl<T: Field, M: Measure> Add for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn add(self, rhs: Self) -> Self::Output {
        PDF::new(self.v + rhs.v)
    }
}
impl<T: Field, M: Measure> Mul for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn mul(self, rhs: Self) -> Self::Output {
        PDF::new(self.v * rhs.v)
    }
}
impl<T: Field, M: Measure> Div for PDF<T, M> {
    // must be under the same field and measure
    // FIXME if you divide two pdfs of the same measure, does that result in a dimensionless quantity? or is it still a pdf? not sure.
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        PDF::new(self.v / rhs.v)
    }
}

impl<T: Field, S: Scalar, M: Measure> Mul<S> for PDF<T, M>
where
    T: Mul<S, Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: S) -> Self::Output {
        PDF::new(self.v * rhs)
    }
}
impl<T: Field, S: Scalar, M: Measure> Div<S> for PDF<T, M>
where
    T: Div<S, Output = T>,
{
    type Output = Self;

    fn div(self, rhs: S) -> Self::Output {
        PDF::new(self.v / rhs)
    }
}

// special conversions
impl<T: Field> PDF<T, SolidAngle> {
    pub fn convert_to_projected_solid_angle<S: Scalar>(
        &self,
        cos_theta: S,
    ) -> PDF<T, ProjectedSolidAngle>
    where
        T: Mul<S, Output = T>,
    {
        PDF::new(self.v * cos_theta.abs())
    }
}

impl<T: Field> PDF<T, Area> {
    pub fn convert_to_solid_angle<S: Scalar>(
        &self,
        cos_theta: S,
        distance_squared: S,
    ) -> PDF<T, SolidAngle>
    where
        T: Mul<S, Output = T> + Div<S, Output = T>,
    {
        PDF::new(self.v * cos_theta.abs() / distance_squared)
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
        T: Mul<S, Output = T> + Div<S, Output = T>,
    {
        // this is valid, but probably somewhat slower.
        // self.convert_to(cos_i, distance_squared).convert_to(cos_o)
        PDF::new(self.v * (cos_o * cos_i).abs() / (distance_squared))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    // TODO: come up with some tests that demonstrate monte carlo integration with change of variables using pdf conversions
    #[test]
    fn test_area_pdf() {
        let area_pdf: PDF<f32, Area> = PDF::new(1.0);
        let solid_angle = area_pdf.convert_to_solid_angle(0.5, 2.0);
        let projected_solid_angle0 = area_pdf.convert_to_projected_solid_angle(0.5, 0.5, 2.0);
        let projected_solid_angle1 = solid_angle.convert_to_projected_solid_angle(0.5);

        println!("{:?}", area_pdf);
        println!("{:?}", solid_angle);
        println!("{:?}", projected_solid_angle0);
        println!("{:?}", projected_solid_angle1);
        assert!(*projected_solid_angle0 == *projected_solid_angle1);

        let area_pdf: PDF<f32x4, Area> = PDF::new(f32x4::new(0.1, 0.4, 0.2, 10.0));
        let solid_angle = area_pdf.convert_to_solid_angle(0.5, 2.0);
        let projected_solid_angle0 = area_pdf.convert_to_projected_solid_angle(0.5, 0.5, 2.0);
        let projected_solid_angle1 = solid_angle.convert_to_projected_solid_angle(0.5);

        println!("{:?}", area_pdf);
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
