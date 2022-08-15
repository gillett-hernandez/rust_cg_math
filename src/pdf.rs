use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
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
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T: Field, M: Measure> DerefMut for PDF<T, M> {
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

// special conversions
impl<T: Field> PDF<T, SolidAngle> {
    pub fn convert_to(&self, cos_theta: T) -> PDF<T, ProjectedSolidAngle> {
        PDF::new(self.v * cos_theta.abs())
    }
}

impl<T: Field> PDF<T, Area> {
    pub fn convert_to(&self, cos_theta: T, distance: T) -> PDF<T, ProjectedSolidAngle> {
        PDF::new(self.v * cos_theta.abs() / (distance * distance))
    }
}
