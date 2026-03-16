use std::{
    // marker::PhantomData,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{prelude::*, spaces::SpaceParameterization};

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
    pub fn new_with_measure(v: T, m: M) -> Self {
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
        Self::new(v)
    }
}

/* impl<T: Field, M: Measure> Add for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn add(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v + rhs.v, self.measure.combine(rhs.measure))
    }
} */
impl<T: Field, M: Measure> Mul<T> for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn mul(self, rhs: T) -> Self::Output {
        PDF::new(self.v * rhs)
    }
}

/*impl<T: Field, M: Measure> Div for PDF<T, M> {
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
} */

// special conversions
impl<T: Field, P: SpaceParameterization> PDF<T, SolidAngle<P>>
where
    SolidAngle<P>: Measure,
{
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
    pub fn convert_to_solid_angle<S: Scalar, P: SpaceParameterization>(
        &self,
        cos_theta: S,
        distance_squared: S,
    ) -> PDF<T, SolidAngle<P>>
    where
        T: FromScalar<S>,
        SolidAngle<P>: Measure,
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

// // impl<T> PDF<T, ProjectedSolidAngle> where T: Field {}
// impl<T: Field> PDF<T, ProjectedSolidAngle> {
//     fn convert_to_throughput(self, area_pdf: PDF<T, Area>) -> PDF<T, Throughput> {
//         (*area_pdf * *self).into()
//     }
// }

// TODO: come up with some tests that demonstrate monte carlo integration with change of variables using pdf conversions
