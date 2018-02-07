//! This file implements most of the standard color functions that essentially work on 3D space,
//! including Euclidean distance, midpoints, and more. All of these methods work on `Color` types
//! that implement `Into<Coord>` and `From<Coord>`, and some don't require `From<Coord>`. This makes
//! it easy to provide these for custom `Color` types.

use coord::Coord;
use color::Color;


/// Some errors that might pop up when dealing with colors as coordinates.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ColorCalcError {
    MismatchedWeights,
}


/// A trait that indicates that the current Color can be embedded in 3D space. This also requires
/// `Clone` and `Copy`: there shouldn't be any necessary information outside of the coordinate data.
pub trait ColorPoint : Color + Into<Coord> + From<Coord> + Clone + Copy {
    /// Gets the Euclidean distance between these two points when embedded in 3D space. This should
    /// **not** be used as an analog of color similarity: use the `distance()` function for
    /// that. Formally speaking, this is a *metric*: it is 0 if and only if self and other are the
    /// same, the distance between two points A and B is never larger than the distance from A to C
    /// and the distance from B to C summed, and it is never negative.
    fn euclidean_distance(self, other: Self) -> f64 {
        let c1: Coord = self.into();
        let c2: Coord = other.into();
        c1.euclidean_distance(&c2)
    }

    /// Gets the *weighted midpoint* of two colors in a space as a new `Color`. This is defined as the
    /// color corresponding to the point along the line segment connecting the two points such that
    /// the distance to the second point is the weight, which for most applications needs to be
    /// between 0 and 1. For example, a weight of 0.9 would make the midpoint one-tenth as much
    /// affected by the second points as the first.
    fn weighted_midpoint(self, other: Self, weight: f64) -> Self {
        let c1: Coord = self.into();
        let c2: Coord = other.into();
        Self::from(c1.weighted_midpoint(&c2, weight))
    }

    /// Like `weighted_midpoint`, but with `weight = 0.5`: essentially, the `Color` representing the
    /// midpoint of the two inputs in 3D space.
    fn midpoint(self, other: Self) -> Self {
        let c1: Coord = self.into();
        let c2: Coord = other.into();
        Self::from(c1.midpoint(&c2))
    }

    /// Returns the weighted average of a given set of colors. Weights will be normalized so that they
    /// sum to 1. Each component of the final value will be calculated by summing the components of
    /// each of the input colors multiplied by their given weight.
    /// # Errors
    /// Returns `ColorCalcError::MismatchedWeights` if the number of colors (`self` and anything in
    /// `others`) and the number of weights mismatch.
    fn weighted_average(self, others: Vec<Self>, weights: Vec<f64>) -> Result<Self, ColorCalcError>  {
        if others.len() + 1 != weights.len() {
            Err(ColorCalcError::MismatchedWeights)
        }
        else {
            let c1: Coord = self.into();
            let norm: f64 = weights.iter().sum();
            let mut coord = c1 * weights[0] / norm;
            for i in 1..weights.len() {
                coord = coord + others[i-1].into() * weights[i] / norm;
            }
            Ok(Self::from(coord))
        }
    }
    /// Returns the arithmetic mean of a given set of colors. Equivalent to `weighted_average` in the
    /// case where each weight is the same.
    fn average(self, others: Vec<Self>) -> Coord {
        let c1: Coord = self.into();
        let other_cs = others.iter().map(|x| (*x).into()).collect();
        c1.average(other_cs)
    }
}

impl<T: Color + Into<Coord> + From<Coord> + Copy + Clone> ColorPoint for T {
    // nothing to do
}


#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use colors::cielabcolor::CIELABColor;

    #[test]
    fn test_cielab_distance() {
        // pretty much should work the same for any type, so why not just CIELAB?
        let lab1 = CIELABColor{l: 10.5, a: -45.0, b: 40.0};
        let lab2 = CIELABColor{l: 54.2, a: 65.0, b: 100.0};
        println!("{}", lab1.euclidean_distance(lab2));
        assert!((lab1.euclidean_distance(lab2) - 132.70150715).abs() <= 1e-7);
    }
}
