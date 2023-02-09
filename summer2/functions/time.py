from typing import Union, Callable
from numbers import Number

from computegraph import ComputeGraph
from computegraph.types import GraphObject, Function
import jax

from summer2.parameters import Time

from .util import _as_graphobj, piecewise_constant
from . import interpolate


def get_linear_interpolation_function(
    x_pts: jax.Array, y_pts: Union[jax.Array, GraphObject], x_axis=Time
) -> Function:
    """Build a linear interpolator over the x and y points supplied

    Args:
        x_pts: Array of x points over which to interpolate
        y_pts: Values at f(x) for x_pts, over which to interpolate
        x_axis: First argument to this function; defaults to Time.

    Returns:
        Resultant GraphObject Function
    """
    y_pts = _as_graphobj(y_pts)
    yscale = Function(interpolate.get_scale_data, [y_pts])
    curve = interpolate.build_linear_interpolator(x_pts)

    return Function(curve, [x_axis, yscale])


def get_sigmoidal_interpolation_function(
    x_pts: jax.Array, y_pts: Union[jax.Array, GraphObject], x_axis=Time, curvature=16.0
) -> Function:
    """Build a piecewise sigmoidal curve interpolator over the x and y points supplied

    Args:
        x_pts: Array of x points over which to interpolate
        y_pts: Values at f(x) for x_pts, over which to interpolate
        x_axis: First argument to this function; defaults to Time.

        curvature (optional): Curvature of the sigmoid. Defaults to 16.0.

    Returns:
        Function: _description_
    """
    y_pts = _as_graphobj(y_pts)
    yscale = Function(interpolate.get_scale_data, [y_pts])
    curve = interpolate.build_sigmoidal_multicurve(x_pts, curvature=curvature)

    return Function(curve, [x_axis, yscale])


def get_piecewise_scalar_function(
    breakpoints: jax.Array, values: Union[jax.Array, GraphObject], x_axis=Time
) -> Function:
    """Create a Function object to be evaluated in the {x} domain, returning
    a value selected from {values}, the index of which is the x-bounds
    described by {breakpoints}.

    By default x is summer2 Time (time-varying function)

    Args:
        breakpoints: Breakpoints to which x is compared
        values: Array-like of length len(breakpoints) + 1
        x: GraphObject supplying the x value.  Defaults to Time.

    Returns:
        The resulting wrapped Function object
    """
    breakpoints = _as_graphobj(breakpoints)
    values = _as_graphobj(values)
    return Function(piecewise_constant, (x_axis, breakpoints, values))


def _tfunc(f: Function):
    return lambda t, parameters: f(model_variables={"time": t}, parameters=parameters)["out"]


def get_time_callable(f: Function, jit_compile=True) -> Callable:
    """For a Function f which takes Time and Parameters as arguments,
    return a python callable accepting either a number or an an array,
    and a dictionary of parameter values.

    E.g
    returned_func(1.0, parameters={"scale": 0.5})

    Args:
        f: The Time Function
        jit_compile (optional): JIT the function

    Returns:
        The resulting function
    """
    cg = ComputeGraph(f)
    tf = _tfunc(cg.get_callable())
    if jit_compile:
        tf = jax.jit(tf)
    vmapped = jax.vmap(tf, in_axes=(0, None))
    if jit_compile:
        vmapped = jax.jit(vmapped)

    def wrapped(t, parameters):
        if isinstance(t, Number):
            return tf(t, parameters)
        else:
            return vmapped(t, parameters)

    return wrapped
