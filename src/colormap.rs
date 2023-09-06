//! This module defines a generalized trait, [`ColorMap`], for a colormap—a
//! mapping of the numbers between 0 and 1 to colors in a continuous way—and
//! provides some common ones used in programs like MATLAB and in data
//! visualization everywhere.

use color::{Color, RGBColor};
use colorcet_cmaps;
use colorpoint::ColorPoint;
use coord::Coord;
use matplotlib_cmaps;
use std::iter::Iterator;

/// A trait that models a colormap, a continuous mapping of the numbers between 0 and 1 to
/// colors. Any color output format is supported, but it must be consistent.
pub trait ColorMap<T: Color + Sized> {
    /// Maps a given number between 0 and 1 to a given output `Color`. This should never fail or panic
    /// except for NaN and similar: there should be some Color that marks out-of-range data.
    fn transform_single(&self, color: f64) -> T;
    /// Maps a given collection of numbers between 0 and 1 to an iterator of `Color`s. Does not evaluate
    /// lazily, because the colormap could have some sort of state that changes between iterations otherwise.
    fn transform<U: IntoIterator<Item = f64>>(&self, inputs: U) -> Vec<T> {
        // TODO: make to work on references?
        inputs
            .into_iter()
            .map(|x| self.transform_single(x))
            .collect()
    }
}

/// A struct that describes different transformations of the numbers between 0 and 1 to themselves,
/// used for controlling the linearity or nonlinearity of gradients.
#[derive(Debug, PartialEq, Clone)]
pub enum NormalizeMapping {
    /// A normal linear mapping: each number maps to itself.
    Linear,
    /// A cube root mapping: 1/8 would map to 1/2, for example. This has the effect of emphasizing the
    /// differences in the low end of the range, which is useful for some data like sound intensity
    /// that isn't perceived linearly.
    Cbrt,
    /// A generic mapping, taking as a value any function or closure that maps the integers from 0-1
    /// to the same range. This should never fail.
    Generic(fn(f64) -> f64),
}

impl NormalizeMapping {
    /// Performs the given mapping on an input number, with undefined behavior or panics if the given
    /// number is outside of the range (0, 1). Given an input between 0 and 1, should always output
    /// another number in the same range.
    pub fn normalize(&self, x: f64) -> f64 {
        match *self {
            NormalizeMapping::Linear => x,
            NormalizeMapping::Cbrt => x.cbrt(),
            NormalizeMapping::Generic(func) => func(x),
        }
    }
}

/// A gradient colormap: a continuous, evenly-spaced shift between two colors A and B such that 0 maps
/// to A, 1 maps to B, and any number in between maps to a weighted mix of them in a given
/// coordinate space. Uses the gradient functions in the [`ColorPoint`] trait to complete this.
/// Out-of-range values are simply clamped to the correct range: calling this on negative numbers
/// will return A, and calling this on numbers larger than 1 will return B.
#[derive(Debug, Clone)]
pub struct GradientColorMap<T: ColorPoint> {
    /// The start of the gradient. Calling this colormap on 0 or any negative number returns this color.
    pub start: T,
    /// The end of the gradient. Calling this colormap on 1 or any larger number returns this color.
    pub end: T,
    /// Any additional added nonlinearity imposed on the gradient: for example, a cube root mapping
    /// emphasizes differences in the low end of the range.
    pub normalization: NormalizeMapping,
    /// Any desired padding: offsets introduced that artificially shift the limits of the
    /// range. Expressed as `(new_min, new_max)`, where both are floats and `new_min < new_max`. For
    /// example, having padding of `(1/8, 1)` would remove the lower eighth of the color map while
    /// keeping the overall map smooth and continuous. Padding of `(0., 1.)` is the default and normal
    /// behavior.
    pub padding: (f64, f64),
}

impl<T: ColorPoint> GradientColorMap<T> {
    /// Constructs a new linear [`GradientColorMap`], without padding, from two colors.
    pub fn new_linear(start: T, end: T) -> GradientColorMap<T> {
        GradientColorMap {
            start,
            end,
            normalization: NormalizeMapping::Linear,
            padding: (0., 1.),
        }
    }
    /// Constructs a new cube root [`GradientColorMap`], without padding, from two colors.
    pub fn new_cbrt(start: T, end: T) -> GradientColorMap<T> {
        GradientColorMap {
            start,
            end,
            normalization: NormalizeMapping::Cbrt,
            padding: (0., 1.),
        }
    }
}

impl<T: ColorPoint> ColorMap<T> for GradientColorMap<T> {
    fn transform_single(&self, x: f64) -> T {
        // clamp between 0 and 1 beforehand
        let clamped = if x < 0. {
            0.
        } else if x > 1. {
            1.
        } else {
            x
        };
        self.start
            .padded_gradient(&self.end, self.padding.0, self.padding.1)(
            self.normalization.normalize(clamped),
        )
    }
}

/// A colormap that linearly interpolates between a given series of values in an equally-spaced
/// progression. This is modeled off of the `matplotlib` Python library's `ListedColormap`, and is
/// only used to provide reference implementations of the standard matplotlib colormaps. Clamps values
/// outside of 0 to 1.
#[derive(Debug, Clone)]
pub struct ListedColorMap {
    /// The list of values, as a vector of `[f64]` arrays that provide equally-spaced RGB values.
    pub vals: Vec<[f64; 3]>,
}

impl<T: ColorPoint> ColorMap<T> for ListedColorMap {
    /// Linearly interpolates by first finding the two colors on either boundary, and then using a
    /// simple linear gradient. There's no need to instantiate every single Color, because the vast
    /// majority of them aren't important for one computation.
    fn transform_single(&self, x: f64) -> T {
        let clamped = if x < 0. {
            0.
        } else if x > 1. {
            1.
        } else {
            x
        };
        // TODO: keeping every Color in memory might be more efficient for large-scale
        // transformation; if it's a performance issue, try and fix

        // now find the two values that bound the clamped x
        // get the index as a floating point: the integers on either side bound it
        // we subtract 1 because 0-n is n+1 numbers, not n
        // otherwise, 1 would map out of range
        let float_ind = clamped * (self.vals.len() as f64 - 1.);
        let ind1 = float_ind.floor() as usize;
        let ind2 = float_ind.ceil() as usize;
        if ind1 == ind2 {
            // x is exactly on the boundary, no interpolation needed
            let arr = self.vals[ind1]; // guaranteed to be in range
            RGBColor::from(Coord {
                x: arr[0],
                y: arr[1],
                z: arr[2],
            })
            .convert()
        } else {
            // interpolate
            let arr1 = self.vals[ind1];
            let arr2 = self.vals[ind2];
            let coord1 = Coord {
                x: arr1[0],
                y: arr1[1],
                z: arr1[2],
            };
            let coord2 = Coord {
                x: arr2[0],
                y: arr2[1],
                z: arr2[2],
            };
            // now interpolate and convert to the desired type
            let rgb: RGBColor = coord2.weighted_midpoint(&coord1, clamped).into();
            rgb.convert()
        }
    }
}

// now just constructors
impl ListedColorMap {
    // TODO: In the future, I'd like to remove this weird array type bound if possible
    /// Initializes a ListedColorMap from an iterator of arrays [R, G, B].
    pub fn new<T: Iterator<Item = [f64; 3]>>(vals: T) -> ListedColorMap {
        ListedColorMap {
            vals: vals.collect(),
        }
    }
    /// Initializes a viridis colormap, a pleasing blue-green-yellow colormap that is perceptually
    /// uniform with respect to luminance, found in Python's `matplotlib` as the default
    /// colormap.
    pub fn viridis() -> ListedColorMap {
        let vals = matplotlib_cmaps::VIRIDIS_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// Initializes a magma colormap, a pleasing blue-purple-red-yellow map that is perceptually
    /// uniform with respect to luminance, found in Python's `matplotlib.`
    pub fn magma() -> ListedColorMap {
        let vals = matplotlib_cmaps::MAGMA_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// Initializes an inferno colormap, a pleasing blue-purple-red-yellow map similar to magma, but
    /// with a slight shift towards red and yellow, that is perceptually uniform with respect to
    /// luminance, found in Python's `matplotlib.`
    pub fn inferno() -> ListedColorMap {
        let vals = matplotlib_cmaps::INFERNO_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// Initializes a plasma colormap, a pleasing blue-purple-red-yellow map that is perceptually
    /// uniform with respect to luminance, found in Python's `matplotlib.` It eschews the really dark
    /// blue found in inferno and magma, instead starting at a fairly bright blue.
    pub fn plasma() -> ListedColorMap {
        let vals = matplotlib_cmaps::PLASMA_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// Initializes a cividis colormap, a pleasing shades of blue-yellow map that is perceptually
    /// uniform with respect to luminance, found in Python's `matplotlib.`
    pub fn cividis() -> ListedColorMap {
        let vals = matplotlib_cmaps::CIVIDIS_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// Initializes a turbo colormap, a pleasing blue-green-red map that is perceptually
    /// uniform with respect to luminance, found in Python's `matplotlib.`
    pub fn turbo() -> ListedColorMap {
        let vals = matplotlib_cmaps::TURBO_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "circle" is a constant-brightness, perceptually uniform cyclic rainbow map
    /// going from magenta through blue, green and red back to magenta.
    pub fn circle() -> ListedColorMap {
        let vals = matplotlib_cmaps::CIRCLE_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "bluered" is a diverging colormap going from dark magenta/blue/cyan to yellow/red/dark purple,
    /// analogously to "RdBu_r" but with higher contrast and more uniform gradient. It is suitable for
    /// plotting velocity maps (blue/redshifted) and is similar to "breeze" and "mist" in this respect,
    /// but has (nearly) white as the central color instead of green.
    /// It is also cyclic (same colors at endpoints).
    pub fn bluered() -> ListedColorMap {
        let vals = matplotlib_cmaps::BLUERED_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "breeze" is a better-balanced version of "jet", with diverging luminosity profile,
    /// going from dark blue to bright green in the center and then back to dark red.
    /// It is nearly perceptually uniform, unlike the original jet map.
    pub fn breeze() -> ListedColorMap {
        let vals = matplotlib_cmaps::BREEZE_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "mist" is another replacement for "jet" or "rainbow" maps, which differs from "breeze" by
    /// having smaller dynamical range in brightness. The red and blue endpoints are darker than
    /// the green center, but not as dark as in "breeze", while the center is not as bright.
    pub fn mist() -> ListedColorMap {
        let vals = matplotlib_cmaps::MIST_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "earth" is a rainbow-like colormap with increasing luminosity, going from black through
    //  dark blue, medium green in the middle and light red/orange to white.
    // # It is nearly perceptually uniform, monotonic in luminosity, and is suitable for
    // # plotting nearly anything, especially velocity maps (blue/redshifted).
    // # It resembles "gist_earth" (but with more vivid colors) or MATLAB's "parula".
    pub fn earth() -> ListedColorMap {
        let vals = matplotlib_cmaps::EARTH_DATA.to_vec();
        ListedColorMap { vals }
    }
    /// "hell" is a slightly tuned version of "inferno", with the main difference that it goes to
    // # pure white at the bright end (starts from black, then dark blue/purple, red in the middle,
    // # yellow and white). It is fully perceptually uniform and monotonic in luminosity.
    pub fn hell() -> ListedColorMap {
        let vals = matplotlib_cmaps::HELL_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_tritanopic_kcw_5_95_c22() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TRITANOPIC_KCW_5_95_C22_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_wcmr_100_45_c42() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_WCMR_100_45_C42_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kgy_5_95_c69() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KGY_5_95_C69_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_grey_10_95_c0() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_GREY_10_95_C0_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_protanopic_deuteranopic_kbw_5_95_c34() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_PROTANOPIC_DEUTERANOPIC_KBW_5_95_C34_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kbgyw_10_98_c63() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KBGYW_10_98_C63_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_tritanotopic_krjcw_5_95_c24() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TRITANOPIC_KRJCW_5_95_C24_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kbc_5_95_c73() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KBC_5_95_C73_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_wyor_100_45_c55() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_WYOR_100_45_C55_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_ternary_green_0_46_c42() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TERNARY_GREEN_0_46_C42_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kryw_5_100_c64() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRYW_5_100_C64_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bgyw_15_100_c68() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BGYW_15_100_C68_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bmw_5_95_c86() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BMW_5_95_C86_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_grey_0_100_c0() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_GREY_0_100_C0_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bmy_10_95_c71() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BMY_10_95_C71_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_protanopic_deuteranopic_kbjyw_5_95_c25() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_PROTANOPIC_DEUTERANOPIC_KBJYW_5_95_C25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kry_5_98_c75() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRY_5_98_C75_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_blue_95_50_c20() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BLUE_95_50_C20_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_ternary_red_0_50_c52() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TERNARY_RED_0_50_C52_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_tritanotopic_krw_5_95_c46() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TRITANOPIC_KRW_5_95_C46_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kry_0_97_c73() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRY_0_97_C73_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kgboy_20_95_c57() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KBGOY_20_95_C57_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kry_5_95_c72() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRY_5_95_C72_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_worb_100_25_c53() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_WORB_100_25_C53_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kryw_0_100_c71() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRYW_0_100_C71_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bmw_5_95_c89() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BMW_5_95_C89_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_tritanopic_krjcw_5_98_c46() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TRITANOPIC_KRJCW_5_98_C46_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kbgyw_5_98_c62() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KBGYW_5_98_C62_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_protanopic_deuteranopic_kyw_5_95_c49() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_PROTANOPIC_DEUTERANOPIC_KYW_5_95_C49_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_gow_60_85_c27() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_GOW_60_85_C27_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bmy_10_95_c78() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BMY_10_95_C78_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_gow_65_90_c35() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_GOW_65_90_C35_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_linear_protanopic_deuteranopic_bjy_57_89_c34() -> ListedColorMap {
        let vals =
            colorcet_cmaps::DIVERGING_LINEAR_PROTANOPIC_DEUTERANOPIC_BJY_57_89_C34_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_ternary_blue_0_44_c57() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_TERNARY_BLUE_0_44_C57_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_protanopic_deuteranopic_kbw_5_98_c40() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_PROTANOPIC_DEUTERANOPIC_KBW_5_98_C40_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_linear_bjr_30_55_c53() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_LINEAR_BJR_30_55_C53_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bgyw_15_100_c67() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BGYW_15_100_C67_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bgyw_20_98_c66() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BGYW_20_98_C66_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_bgy_10_95_c74() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_BGY_10_95_C74_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn linear_kryw_5_100_c67() -> ListedColorMap {
        let vals = colorcet_cmaps::LINEAR_KRYW_5_100_C67_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_linear_bjy_30_90_c45() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_LINEAR_BJY_30_90_C45_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_tritanopic_cwr_75_98_c20() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_TRITANOPIC_CWR_75_98_C20_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_isoluminant_cjm_75_c23() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_ISOLUMINANT_CJM_75_C23_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bky_60_10_c30() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BKY_60_10_C30_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bwr_40_95_c42() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BWR_40_95_C42_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_gkr_60_10_c40() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_GKR_60_10_C40_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_gwv_55_95_c39() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_GWV_55_95_C39_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bwr_55_98_c37() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BWR_55_98_C37_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_rainbow_bgymr_45_85_c67() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_RAINBOW_BGYMR_45_85_C67_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_isoluminant_cjo_70_c25() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_ISOLUMINANT_CJO_70_C25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_isoluminant_cjm_75_c24() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_ISOLUMINANT_CJM_75_C24_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bkr_55_10_c35() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BKR_55_10_C35_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bwr_20_95_c54() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BWR_20_95_C54_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_protanopic_deuteranopic_bwy_60_95_c32() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_PROTANOPIC_DEUTERANOPIC_BWY_60_95_C32_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_cwm_80_100_c22() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_CWM_80_100_C22_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_gwr_55_95_c38() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_GWR_55_95_C38_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn diverging_bwg_20_95_c41() -> ListedColorMap {
        let vals = colorcet_cmaps::DIVERGING_BWG_20_95_C41_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_rygcbmr_50_90_c64() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_RYGCBMR_50_90_C64_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mybm_20_100_c48_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYBM_20_100_C48_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_bgrmb_35_70_c75_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_BGRMB_35_70_C75_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_wrkbw_10_90_c43() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_WRKBW_10_90_C43_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_bgrmb_35_70_c75() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_BGRMB_35_70_C75_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mygbm_50_90_c46() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYGBM_50_90_C46_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_tritanopic_cwrk_40_100_c20() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_TRITANOPIC_CWRK_40_100_C20_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mrybm_35_75_c68() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MRYBM_35_75_C68_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_ymcgy_60_90_c67() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_YMCGY_60_90_C67_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mygbm_50_90_c46_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYGBM_50_90_C46_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mygbm_30_95_c78_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYGBM_30_95_C78_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mygbm_30_95_c78() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYGBM_30_95_C78_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_protanopic_deuteranopic_bw_yk_16_96_c31() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_PROTANOPIC_DEUTERANOPIC_BWYK_16_96_C31_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_tritanopic_wrwc_70_100_c20() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_TRITANOPIC_WRWC_70_100_C20_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mrybm_35_75_c68_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MRYBM_35_75_C68_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_rygcbmr_50_90_c64_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_RYGCBMR_50_90_C64_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_protanopic_deuteranopic_wywb_55_96_c33() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_PROTANOPIC_DEUTERANOPIC_WYWB_55_96_C33_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_wrkbw_10_90_c43_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_WRKBW_10_90_C43_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_wrwbw_40_90_c42_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_WRWBW_40_90_C42_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_grey_15_85_c0_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_GREY_15_85_C0_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_ymcgy_60_90_c67_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_YMCGY_60_90_C67_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_grey_15_85_c0() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_GREY_15_85_C0_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_mybm_20_100_c48() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_MYBM_20_100_C48_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn cyclic_wrwbw_40_90_c42() -> ListedColorMap {
        let vals = colorcet_cmaps::CYCLIC_WRWBW_40_90_C42_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn isoluminant_cgo_70_c39() -> ListedColorMap {
        let vals = colorcet_cmaps::ISOLUMINANT_CGO_70_C39_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn isoluminant_cgo_80_c38() -> ListedColorMap {
        let vals = colorcet_cmaps::ISOLUMINANT_CGO_80_C38_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn isoluminant_cm_70_c39() -> ListedColorMap {
        let vals = colorcet_cmaps::ISOLUMINANT_CM_70_C39_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn rainbow_bgyrm_35_85_c71() -> ListedColorMap {
        let vals = colorcet_cmaps::RAINBOW_BGYRM_35_85_C71_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn rainbow_bgyr_35_85_c73() -> ListedColorMap {
        let vals = colorcet_cmaps::RAINBOW_BGYR_35_85_C73_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn rainbow_bgyr_35_85_c72() -> ListedColorMap {
        let vals = colorcet_cmaps::RAINBOW_BGYR_35_85_C72_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn rainbow_bgyr_10_90_c83() -> ListedColorMap {
        let vals = colorcet_cmaps::RAINBOW_BGYR_10_90_C83_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn rainbow_bgyrm_35_85_c69() -> ListedColorMap {
        let vals = colorcet_cmaps::RAINBOW_BGYRM_35_85_C69_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_category10() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_CATEGORY10_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw_minc_20_minl_30() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_MINC_20_MINL_30_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw_minc_20_hue_330_100() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_MINC_20_HUE_330_100_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw_minc_20() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_MINC_20_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw_minc_20_maxl_70() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_MINC_20_MAXL_70_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_hv() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_HV_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn glasbey_bw_minc_20_hue_150_280() -> ListedColorMap {
        let vals = colorcet_cmaps::GLASBEY_BW_MINC_20_HUE_150_280_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn circle_mgbm_67_c31_s25() -> ListedColorMap {
        let vals = colorcet_cmaps::CIRCLE_MGBM_67_C31_S25_DATA.to_vec();
        ListedColorMap { vals }
    }

    pub fn circle_mgbm_67_c31() -> ListedColorMap {
        let vals = colorcet_cmaps::CIRCLE_MGBM_67_C31_DATA.to_vec();
        ListedColorMap { vals }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use color::RGBColor;

    #[test]
    fn test_linear_gradient() {
        let red = RGBColor::from_hex_code("#ff0000").unwrap();
        let blue = RGBColor::from_hex_code("#0000ff").unwrap();
        let cmap = GradientColorMap::new_linear(red, blue);
        let vals = vec![-0.2, 0., 1. / 15., 1. / 5., 4. / 5., 1., 100.];
        let cols = cmap.transform(vals);
        let strs = vec![
            "#FF0000", "#FF0000", "#EE0011", "#CC0033", "#3300CC", "#0000FF", "#0000FF",
        ];
        for (i, col) in cols.into_iter().enumerate() {
            assert_eq!(col.to_string(), strs[i]);
        }
    }
    #[test]
    fn test_cbrt_gradient() {
        let red = RGBColor::from_hex_code("#CC0000").unwrap();
        let blue = RGBColor::from_hex_code("#0000CC").unwrap();
        let cmap = GradientColorMap::new_cbrt(red, blue);
        let vals = vec![-0.2, 0., 1. / 27., 1. / 8., 8. / 27., 1., 100.];
        let cols = cmap.transform(vals);
        let strs = vec![
            "#CC0000", "#CC0000", "#880044", "#660066", "#440088", "#0000CC", "#0000CC",
        ];
        for (i, col) in cols.into_iter().enumerate() {
            assert_eq!(col.to_string(), strs[i]);
        }
    }
    #[test]
    fn test_padding() {
        let red = RGBColor::from_hex_code("#CC0000").unwrap();
        let blue = RGBColor::from_hex_code("#0000CC").unwrap();
        let mut cmap = GradientColorMap::new_cbrt(red, blue);
        cmap.padding = (0.25, 0.75);
        // essentially, start and end are now #990033 and #330099
        let vals = vec![-0.2, 0., 1. / 27., 1. / 8., 8. / 27., 1., 100.];
        let cols = cmap.transform(vals);
        let strs = vec![
            "#990033", "#990033", "#770055", "#660066", "#550077", "#330099", "#330099",
        ];
        for (i, col) in cols.into_iter().enumerate() {
            assert_eq!(col.to_string(), strs[i]);
        }
    }
    #[test]
    fn test_mpl_colormaps() {
        let viridis = ListedColorMap::viridis();
        let magma = ListedColorMap::magma();
        let inferno = ListedColorMap::inferno();
        let plasma = ListedColorMap::plasma();
        let vals = vec![-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1., 100.];
        // these values were taken using matplotlib
        let viridis_colors = [
            [0.267004, 0.004874, 0.329415],
            [0.267004, 0.004874, 0.329415],
            [0.253935, 0.265254, 0.529983],
            [0.163625, 0.471133, 0.558148],
            [0.134692, 0.658636, 0.517649],
            [0.477504, 0.821444, 0.318195],
            [0.993248, 0.906157, 0.143936],
            [0.993248, 0.906157, 0.143936],
        ];
        let magma_colors = [
            [1.46200000e-03, 4.66000000e-04, 1.38660000e-02],
            [1.46200000e-03, 4.66000000e-04, 1.38660000e-02],
            [2.32077000e-01, 5.98890000e-02, 4.37695000e-01],
            [5.50287000e-01, 1.61158000e-01, 5.05719000e-01],
            [8.68793000e-01, 2.87728000e-01, 4.09303000e-01],
            [9.94738000e-01, 6.24350000e-01, 4.27397000e-01],
            [9.87053000e-01, 9.91438000e-01, 7.49504000e-01],
            [9.87053000e-01, 9.91438000e-01, 7.49504000e-01],
        ];
        let plasma_colors = [
            [5.03830000e-02, 2.98030000e-02, 5.27975000e-01],
            [5.03830000e-02, 2.98030000e-02, 5.27975000e-01],
            [4.17642000e-01, 5.64000000e-04, 6.58390000e-01],
            [6.92840000e-01, 1.65141000e-01, 5.64522000e-01],
            [8.81443000e-01, 3.92529000e-01, 3.83229000e-01],
            [9.88260000e-01, 6.52325000e-01, 2.11364000e-01],
            [9.40015000e-01, 9.75158000e-01, 1.31326000e-01],
            [9.40015000e-01, 9.75158000e-01, 1.31326000e-01],
        ];
        let inferno_colors = [
            [1.46200000e-03, 4.66000000e-04, 1.38660000e-02],
            [1.46200000e-03, 4.66000000e-04, 1.38660000e-02],
            [2.58234000e-01, 3.85710000e-02, 4.06485000e-01],
            [5.78304000e-01, 1.48039000e-01, 4.04411000e-01],
            [8.65006000e-01, 3.16822000e-01, 2.26055000e-01],
            [9.87622000e-01, 6.45320000e-01, 3.98860000e-02],
            [9.88362000e-01, 9.98364000e-01, 6.44924000e-01],
            [9.88362000e-01, 9.98364000e-01, 6.44924000e-01],
        ];
        let colors = vec![viridis_colors, magma_colors, inferno_colors, plasma_colors];
        let cmaps = vec![viridis, magma, inferno, plasma];
        for (colors, cmap) in colors.iter().zip(cmaps.iter()) {
            for (ref_arr, test_color) in colors.iter().zip(cmap.transform(vals.clone()).iter()) {
                let ref_color = RGBColor {
                    r: ref_arr[0],
                    g: ref_arr[1],
                    b: ref_arr[2],
                };
                let deref_test_color: RGBColor = *test_color;
                assert_eq!(deref_test_color.to_string(), ref_color.to_string());
            }
        }
    }
}
