"""Builder functions for interpolating data - replaces scale_up.py

Note: For most user applications, refer to the get_*_function wrappers in time.py

build_static_sigmoidal_multicurve is the equivalent of scale_up_function(method=4),
with build_sigmoidal_multicurve producing a dynamic (parameterizable) version of this

"""

from jax import Array, lax, numpy as jnp

from .util import binary_search_ge


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


def build_sigmoidal_multicurve(x_points: Array, curvature=16.0) -> callable:
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
    xranges = jnp.diff(jnp.array(x_points))
    x_points = jnp.array(x_points)

    xmin = x_points.min()
    xmax = x_points.max()
    xbounds = jnp.array([xmin, xmax])

    sig = make_norm_sigmoid(curvature)

    def get_curve_at_t(t, values, ranges):
        # idx = sum(t >= x_points) - 1
        idx = binary_search_ge(t, x_points)

        offset = t - x_points[idx]
        relx = offset / xranges[idx]
        rely = sig(relx)
        return values[idx] + (rely * ranges[idx])

    def scaled_curve(t: float, ydata: dict):
        # Branch on whether t is in bounds
        bounds_state = sum(t > xbounds)
        branches = [
            lambda _, __, ___: ydata["min"],
            get_curve_at_t,
            lambda _, __, ___: ydata["max"],
        ]
        return lax.switch(bounds_state, branches, t, ydata["values"], ydata["ranges"])

    return scaled_curve


def build_linear_interpolator(x_points: Array) -> callable:
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
    xranges = jnp.diff(jnp.array(x_points))
    x_points = jnp.array(x_points)

    xmin = x_points.min()
    xmax = x_points.max()
    xbounds = jnp.array([xmin, xmax])

    def get_curve_at_t(t, values, ranges):
        # idx = sum(t >= x_points) - 1
        idx = binary_search_ge(t, x_points)

        offset = t - x_points[idx]
        relx = offset / xranges[idx]
        return values[idx] + (relx * ranges[idx])

    from jax import lax

    def linear(t: float, ydata: dict):
        # Branch on whether t is in bounds
        bounds_state = sum(t > xbounds)
        branches = [
            lambda _, __, ___: ydata["min"],
            get_curve_at_t,
            lambda _, __, ___: ydata["max"],
        ]
        return lax.switch(bounds_state, branches, t, ydata["values"], ydata["ranges"])

    return linear


def get_scale_data(points: Array) -> dict:
    """
    Precompute min, max, and ranges for a set of data to be used in a scaling function such as that
    produced by build_sigmoidal_multicurve.  The onus is on the caller of this function to ensure
    they are the length expected by the target callee
    """
    ranges = jnp.diff(points)
    ymin = points[0]
    ymax = points[-1]

    data = {"min": ymin, "max": ymax, "values": points, "ranges": ranges}

    return data
