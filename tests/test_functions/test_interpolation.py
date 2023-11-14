import numpy as np

from summer2.functions import time as stf


def test_linear_2points():
    xrange = np.linspace(0.0, 25.0, 9)
    xpts = np.array([5.0, 20.0])
    ypts = np.array([-1.5, 1.5])

    ifunc = stf.get_linear_interpolation_function(xpts, ypts)
    interp_vals = stf.get_time_callable(ifunc)(xrange)

    assert interp_vals[0] == ypts[0]
    assert interp_vals[1] == ypts[0]

    assert interp_vals[-1] == ypts[-1]
    assert interp_vals[-2] == ypts[-1]

    assert (interp_vals[4]) == 0.0


def test_sigmoidal_2points():
    xrange = np.linspace(0.0, 25.0, 9)
    xpts = np.array([5.0, 20.0])
    ypts = np.array([-1.5, 1.5])

    ifunc = stf.get_linear_interpolation_function(xpts, ypts)
    interp_vals = stf.get_time_callable(ifunc)(xrange)

    assert interp_vals[0] == ypts[0]
    assert interp_vals[1] == ypts[0]

    assert interp_vals[-1] == ypts[-1]
    assert interp_vals[-2] == ypts[-1]

    assert (interp_vals[4]) == 0.0


def test_piecewise_1point():
    xrange = np.array([0.0, 5.0, 10.0])
    xpts = np.array([5.0])
    ypts = np.array([-1.5, 1.5])

    ifunc = stf.get_piecewise_function(xpts, ypts)
    interp_vals = stf.get_time_callable(ifunc)(xrange)

    assert interp_vals[0] == ypts[0]  # < all xpts
    assert interp_vals[1] == ypts[1]  # == first xpts
    assert interp_vals[2] == ypts[-1]  # >= all xpts


def test_piecewise_2points():
    xrange = np.array([0.0, 5.0, 15.0])
    xpts = np.array([5.0, 10.0])
    ypts = np.array([-1.5, 0.0, 2.5])

    ifunc = stf.get_piecewise_function(xpts, ypts)
    interp_vals = stf.get_time_callable(ifunc)(xrange)

    assert interp_vals[0] == ypts[0]  # < all xpts
    assert interp_vals[1] == ypts[1]  # == first xpts
    assert interp_vals[2] == ypts[-1]  # >= all xpts
