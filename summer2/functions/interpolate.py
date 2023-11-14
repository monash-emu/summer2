"""Builder functions for interpolating data - replaces scale_up.py

Note: For most user applications, refer to the get_*_function wrappers in time.py

build_static_sigmoidal_multicurve is the equivalent of scale_up_function(method=4),
with build_sigmoidal_multicurve producing a dynamic (parameterizable) version of this

"""

from jax import Array, lax, numpy as jnp, lax

from .util import binary_search_sum_ge
from collections import namedtuple

# Define datatypes used within interpolation functions
# These carry intermediary compute-once values like ranges/bounds

InterpolatorScaleData = namedtuple("InterpolatorScaleData", ["points", "ranges", "bounds"])


def _uncorrected_sigmoid(x: float, curvature: float) -> float:
    """Return a sigmoid of x (assumed to be between 0.0 and 1.0),
    whose shape and range depends on curvature

    Args:
        x: Input to the function
        curvature: Degree of sigmoidal flattening - 1.0 is linear, higher values increase smoothing

    Returns:
        Output of sigmoidal curve function
    """
    arg = curvature * (0.5 - x)
    return 1.0 / (1.0 + jnp.exp(arg))


def make_norm_sigmoid(curvature: float) -> callable:
    """
    Build a sigmoid function with fixed curvature, whose output is normalized to (0.0,1.0) across
    the (0.0,1.0) input range
    Args:
        curvature: See _uncorrected_sigmoid

    Returns:
        The normalized sigmoid function
    """
    offset = _uncorrected_sigmoid(0.0, curvature)
    scale = 1.0 / (1.0 - (offset * 2.0))

    def sig(x):
        return (_uncorrected_sigmoid(x, curvature) - offset) * scale

    return sig


def build_sigmoidal_multicurve(curvature=16.0) -> callable:
    """Build a sigmoidal smoothing function across points specified by
    x_points; the returned function takes the y values as arguments in
    form of a scale_data dict with keys [min, max, values, ranges]

    Args:
        x_points: x values to interpolate across
        curvature: Sigmoidal curvature.  Default produces the same behaviour as the old
                   scale_up_function

    Returns:
        The multisigmoidal function of (x, scale_data)
    """

    sig = make_norm_sigmoid(curvature)

    def _get_sigmoidal_curve_at_x(x, xdata, ydata):
        idx = binary_search_sum_ge(x, xdata.points) - 1

        offset = x - xdata.points[idx]
        relx = offset / xdata.ranges[idx]
        rely = sig(relx)
        return ydata.points[idx] + (rely * ydata.ranges[idx])

    def interpolate_sigmoidal(t: float, xdata: InterpolatorScaleData, ydata: InterpolatorScaleData):
        # Branch on whether t is in bounds
        bounds_state = sum(t > xdata.bounds)
        branches = [
            lambda _, __, ___: ydata.bounds[0],
            _get_sigmoidal_curve_at_x,
            lambda _, __, ___: ydata.bounds[1],
        ]
        return lax.switch(bounds_state, branches, t, xdata, ydata)

    return interpolate_sigmoidal


#
# Linear Interpolation
#


def _get_linear_curve_at_x(
    x: float, xdata: InterpolatorScaleData, ydata: InterpolatorScaleData
) -> float:
    """Inner function for interpolate_linear

    Args:
        x: x position (usually Time)
        xdata: Scale data for x ('index') points
        ydata: Scale data for y ('value') points

    Returns:
        The interpolated output value
    """
    idx = binary_search_sum_ge(x, xdata.points) - 1

    offset = x - xdata.points[idx]
    relx = offset / xdata.ranges[idx]
    return ydata.points[idx] + (relx * ydata.ranges[idx])


def interpolate_linear(t: float, xdata: InterpolatorScaleData, ydata: InterpolatorScaleData):
    # Branch on whether t is in bounds
    bounds_state = sum(t > xdata.bounds)
    branches = [
        lambda _, __, ___: ydata.bounds[0],
        _get_linear_curve_at_x,
        lambda _, __, ___: ydata.bounds[1],
    ]
    return lax.switch(bounds_state, branches, t, xdata, ydata)


def get_scale_data(points: Array) -> InterpolatorScaleData:
    """
    Precompute ranges (diffs) and bounds (left and right extrema) for a set of data to be used in
    a scaling function such as that produced by build_sigmoidal_multicurve.  The onus is on the
    caller of this function to ensure they are the length expected by the target callee
    """
    ranges = jnp.diff(points)
    lpoint = points[0]
    rpoint = points[-1]

    # data = {"min": ymin, "max": ymax, "values": points, "ranges": ranges}
    return InterpolatorScaleData(points, ranges, jnp.array([lpoint, rpoint]))
