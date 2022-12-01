from numbers import Number
from collections.abc import Container
from jax import lax, numpy as jnp
import jax

from computegraph.types import GraphObject, Function, Data
from summer2.parameters import Time


def _as_graphobj(x) -> GraphObject:
    if isinstance(x, GraphObject):
        return x
    elif isinstance(x, Number):
        return x
    else:
        jnp_arr = jnp.array(x)
        return Data(jnp_arr)


def get_piecewise_scalar_function(breakpoints, values, x=Time) -> Function:
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
    return Function(piecewise_constant, (x, breakpoints, values))


def piecewise_function(x, breakpoints, functions):
    index = sum(x >= breakpoints)
    return lax.switch(index, functions, x)


# All
def piecewise_constant(x, breakpoints, values):
    index = sum(x >= breakpoints)
    return values[index]


def windowed_constant(x: float, value: float, window_start: float, window_length: float):
    breakpoints = jnp.array((window_start, window_start + window_length))
    values = jnp.array((0.0, value, 0.0))
    return piecewise_constant(x, breakpoints, values)


def binary_search_ge(x: float, points: jax.Array) -> int:
    """Find the lowest index of the value within points
    to which x is greater than or equal to, using a binary
    search.

    A value of x lower than min(points) will return 0, and higher
    than max(points) will return len(points)-1

    Args:
        x: Value to find
        points: Array to search

    Returns:
        The index value satisying the above conditions
    """

    def cond(state):
        low, high = state
        return (high - low) > 1

    def body(state):
        low, high = state
        midpoint = (0.5 * (low + high)).astype(int)
        update_upper = x < points[midpoint]
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    low, high = lax.while_loop(cond, body, (0, len(points) - 1))
    return lax.cond(x < points[high], lambda: low, lambda: high)
