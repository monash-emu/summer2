from numbers import Number
from collections.abc import Container
from jax import lax, numpy as jnp, Array
from numpy import ndarray
import pandas as pd
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


def enforce_graph_array(x) -> GraphObject:
    # Already a GraphObject
    if isinstance(x, GraphObject):
        return x

    if isinstance(x, pd.Index) or isinstance(x, pd.Series):
        x = x.to_numpy()

    # Numpy array, cast to jax
    if isinstance(x, ndarray):
        x = jnp.array(x)

    # Jax array, return as Data object
    if isinstance(x, Array):
        return Data(x)

    # Something else - let capture_array handle it

    return capture_array(x)


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


def capture_dict(**kwargs) -> Function:
    """Create a Function that captures a dictionary

    Returns:
        The GraphObject representation of the dict
    """
    return Function(_capture_dict, kwargs=kwargs)


def _capture_dict(**kwargs):
    return kwargs


def capture_array(arr_obj: list, dtype=None) -> Function:
    """Create a GraphObject representation of an (optionally nested) array type,
    declared as a list

    Args:
        arr_obj: A list of the form consumed by (j)np.array(arr_obj)
        dtype (optional): Datatype of the returned array (inferred by default)

    Returns:
        The GraphObject representation of the array
    """
    ishape = _infer_shape(arr_obj)

    if len(ishape) > 1:
        # arr_obj is a nested listed, which we must flatten to expose args to the graph
        flattened = _flatten_arr_obj(arr_obj)

        def _capture_array_reshape(*args):
            return jnp.array(args, dtype=dtype).reshape(ishape)

        return Function(_capture_array_reshape, args=flattened)
    else:
        # Simple instance of 1d array
        def _capture_array(*args):
            return jnp.array(args, dtype=dtype)

        return Function(_capture_array, args=arr_obj)


def _infer_shape(arr_obj: list, cur_shape=()):
    new_shape = (*cur_shape, len(arr_obj))
    if isinstance(arr_obj[0], list):
        msg = "All subarrays must be lists"
        assert all([isinstance(sub_arr, list) for sub_arr in arr_obj]), msg
        return _infer_shape(arr_obj[0], new_shape)
    return new_shape


def _flatten_arr_obj(arr_obj: list, cur_obj=None):
    if cur_obj is None:
        cur_obj = []
    for v in arr_obj:
        if isinstance(v, list):
            _flatten_arr_obj(v, cur_obj)
        else:
            cur_obj.append(v)
    return cur_obj
