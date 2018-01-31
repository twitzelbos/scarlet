//! This module implements the CIELUV color specification, which was adopted concurrently with
//! CIELAB. CIELUV is very similar to CIELAB, but with the difference that u and v are roughly
//! equivalent to red and green and luminance is then used to calculate the blue part.

use color::{Color, XYZColor};
use illuminants::Illuminant;


pub struct CIELUVColor {
    /// The luminance component of LUV. Ranges from 0 to 100 by definition.
    pub l: f64,
    /// The component of LUV that roughly equates to how red the color is. Ranges from 0 to 100 in
    /// most visible colors.
    pub u: f64,
    /// The component of LUV that roughly equates to how green vs. blue the color is. Ranges from 0 to
    /// 100 in most visible colors.
    pub v: f64,
}

impl Color for CIELUVColor {
    /// Given an XYZ color, gets a new CIELUV color.
    fn from_xyz(xyz: XYZColor) -> CIELUVColor {
        // this is not bad: LUV is meant to be easy from XYZ
        // https://en.wikipedia.org/wiki/CIELUV

        // do u and v chromaticity conversions on whitepoint and on given color
        let wp = XYZColor::white_point(xyz.illuminant);
        let denom = |color: XYZColor| {
            color.x + 15.0 * color.y + 3.0 * color.z
        };
        let u_func = |color: XYZColor| {
            4.0 * color.x / denom(color)
        };
        let v_func = |color: XYZColor| {
            9.0 * color.y / denom(color)
        };
            
        let u_prime_n = u_func(wp);
        let v_prime_n = v_func(wp);

        let u_prime = u_func(xyz);
        let v_prime = v_func(xyz);

        let delta: f64 = 6.0 / 29.0; // like CIELAB

        // technically this next division should do nothing: idk if it gets factored out at compile
        // time, but it's just insurance if someone ever decides not to normalize whitepoints to Y=1
        let y_scaled = xyz.y / wp.y; // ranges from 0-1
        let l = if y_scaled <= delta.powf(3.0) {
            (2.0 / delta).powf(3.0) * y_scaled
        } else {
            116.0 * y_scaled.powf(1.0 / 3.0) - 16.0
        };

        let u = 13.0 * l * (u_prime - u_prime_n);
        let v = 13.0 * l * (v_prime - v_prime_n);
        CIELUVColor{l, u, v}
    }
    /// Returns a new `XYZColor` that matches the given color. Note that CIELUV uses its own,
    /// translational chromatic adaptation. Because of this, this will produce inconsistent results
    /// with other chromatic adaptations and may even generate colors that cannot physically
    /// exist. It's best practice to only use the illuminant of the XYZColor that created this color,
    /// which is D50 if you created this using the `Color::convert` method. Because D50 is used in
    /// every usage of `convert`, as long as you don't invoke this by hand results will be fine.
    fn to_xyz(&self, illuminant: Illuminant) -> XYZColor {
        // https://en.wikipedia.org/wiki/CIELUV literally has the equations in order
        // pretty straightforward
        let wp = XYZColor::white_point(illuminant);
        let denom = |color: XYZColor| {
            color.x + 15.0 * color.y + 3.0 * color.z
        };
        let u_func = |color: XYZColor| {
            4.0 * color.x / denom(color)
        };
        let v_func = |color: XYZColor| {
            9.0 * color.y / denom(color)
        };
        let u_prime_n = u_func(wp);
        let v_prime_n = v_func(wp);

        let u_prime = self.u / (13.0 * self.l) + u_prime_n;
        let v_prime = self.v / (13.0 * self.l) + v_prime_n;
        
        let delta: f64 = 6.0 / 29.0;
        
        let y = if self.l <= 8.0 {
            wp.y * self.l * (delta / 2.0).powf(3.0)
        } else {
            wp.y * ((self.l + 16.0) / 116.0).powf(3.0)
        };
        
        let x = y * 9.0 * u_prime / (4.0 * v_prime);
        let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);
        XYZColor{x, y, z, illuminant}
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    
    #[test]
    fn test_cieluv_xyz_conversion_d50() {
        let xyz = XYZColor{x: 0.3, y: 0.53, z: 0.65, illuminant: Illuminant::D50};
        let luv: CIELUVColor = xyz.convert();
        let xyz2: XYZColor = luv.convert();
        assert!(xyz2.approx_equal(&xyz));
    }
}
