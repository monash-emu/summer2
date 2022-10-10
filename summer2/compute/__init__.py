from computegraph.jaxify import get_using_jax

if get_using_jax():
    from .compute_jax import *
else:
    from .compute_numba import *
