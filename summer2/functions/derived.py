"""
Functions operating over arrays

These are designed for use on derived outputs; ie it is expected that data
will be a time series whose length is equal to model.times, and that these
functions will operate over the whole array, and return an array of the same
shape as the input

Where possible, these functions will mimic (or at least be similar to) their
pandas.Series counterparts

"""

from typing import Callable

from jax import numpy as jnp
from jax import Array


def get_rolling_diff(periods: int = 1) -> Callable[[Array], Array]:
    """Build a function that returns the difference of each value at index i,
    to the value at index i-periods ala pandas.Series.diff

    Args:
        periods: The index distance to diff

    Returns:
        A function over a 1d array that returns an array of the same shape,
        whose values are the diffs
    """

    def rolling_diff(x):
        out_arr = jnp.empty_like(x)
        out_arr = out_arr.at[periods:].set(x[periods:] - x[:-periods])
        out_arr = out_arr.at[:periods].set(jnp.nan)
        return out_arr

    return rolling_diff


def _rolling_index(a: jnp.ndarray, window: int):
    idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
    return a[idx]


def get_rolling_reduction(func: callable, window: int) -> Callable[[Array], Array]:
    """Build a function that comptues a reduction function 'func' over each
    rolling window of length 'window'

    Reduction functions are those that take an array as input and return a scalar,
    (or in general reduce array axes to scalar values), such as jnp.mean, jnp.max etc

    This is designed to operate like pandas.Series.rolling (with its default
    window parameters)

    Args:
        func: The reduction function to call; must be jax jittable
        window: The window length

    Returns:
        A function over a 1d array that returns an array of the same shape, but with
        the rolling reduction applied

    """

    def rolling_func(x):
        out_arr = jnp.empty_like(x)
        windowed = _rolling_index(x, window)
        agg = func(windowed, axis=1)
        out_arr = out_arr.at[:window].set(jnp.nan)
        out_arr = out_arr.at[window - 1 :].set(agg)
        return out_arr

    return rolling_func
