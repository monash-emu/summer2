"""
Ensure that the CompartmentalModel model produces the correct force of infection multipliers
See https://parasiteecology.wordpress.com/2013/10/17/density-dependent-vs-frequency-dependent-disease-transmission/
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from summer2.model import Compartment as C
from summer2.model import CompartmentalModel, Stratification
from summer2.runner.jax.model_impl import StepResults


def test_basic_get_infection_multiplier(backend):
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    ons_res = model._get_step_test()

    multiplier = ons_res.infectious_multipliers[0]
    assert multiplier == 10 / 1000


def test_strat_get_infection_multiplier__with_age_strat_and_no_mixing(backend):
    """
    Check FoI when a simple 2-strata stratification applied and no mixing matrix.
    Expect the same results as with the basic case.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    comp_inf = ons_res.model_data["compartment_infectiousness"]

    assert_array_equal(
        comp_inf["default"],
        np.array([1.0, 1.0]),
    )
    assert_array_equal(model._backend._category_lookup, np.zeros(6))

    multiplier = ons_res.infectious_multipliers

    assert_array_equal(multiplier, np.array((0.01, 0.01)))


def test_strat_get_infection_multiplier__with_age_strat_and_simple_mixing(backend):
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Expect same frequency as before, different density.
    Note that the mixing matrix has different meanings for density / vs frequency.

    N.B Mixing matrix format.
    Columns are  the people who are infectors
    Rows are the people who are infected
    So matrix has following values

                  child               adult

      child       child -> child      adult -> child
      adult       child -> adult      adult -> adult

    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    strat.set_mixing_matrix(mixing_matrix)
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    comp_inf = ons_res.model_data["compartment_infectiousness"]
    ifect_multiplier = ons_res.infectious_multipliers

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(ons_res.initial_population, np.array([495, 495, 5, 5, 0.0, 0.0]))

    # Do pre-run force of infection calcs.
    assert_array_equal(
        comp_inf["default"],
        np.array([1.0, 1.0]),
    )
    assert_array_equal(
        model._backend._category_lookup,
        np.array([0, 1, 0, 1, 0, 1]),
    )

    # Do pre-iteration force of infection calcs
    child_density = 5
    adult_density = 5
    assert child_density == 0.5 * 5 + 0.5 * 5
    assert adult_density == 0.5 * 5 + 0.5 * 5

    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == child_density / 500
    assert adult_freq == adult_density / 500

    # Get multipliers
    assert all(ifect_multiplier == child_freq)
    # Santiy check frequency-dependent force of infection
    assert 500.0 * child_freq + 500.0 * adult_freq == 10


def test_strat_get_infection_multiplier__with_age_split_and_simple_mixing(backend):
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Unequally split the children and adults.
    Expect same density as before, different frequency.
    Note that the mixing matrix has different meanings for density / vs frequency.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    strat.set_mixing_matrix(mixing_matrix)
    strat.set_population_split({"child": 0.2, "adult": 0.8})
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(ons_res.initial_population, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run force of infection calcs.

    assert_array_equal(
        ons_res.model_data["compartment_infectiousness"]["default"],
        np.array([1.0, 1.0]),
    )
    assert_array_equal(
        model._backend._category_lookup,
        np.array([0, 1, 0, 1, 0, 1]),
    )
    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert adult_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert_array_equal(ons_res.infectious_multipliers, np.array([child_freq, adult_freq]))


def test_strat_get_infection_multiplier__with_age_strat_and_mixing(backend):
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Use a non-uniform mixing matrix
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[2, 3], [5, 7]], dtype=float)
    strat.set_mixing_matrix(mixing_matrix)
    strat.set_population_split({"child": 0.2, "adult": 0.8})
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(ons_res.initial_population, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run force of infection calcs.

    assert_array_equal(
        ons_res.model_data["compartment_infectiousness"]["default"],
        np.array([1.0, 1.0]),
    )
    assert_array_equal(
        model._backend._category_lookup,
        np.array([0, 1, 0, 1, 0, 1]),
    )

    child_freq = 0.05
    adult_freq = 0.12000000000000001
    assert child_freq == 2 * 2.0 / 200 + 3 * 8.0 / 800
    assert adult_freq == 5 * 2.0 / 200 + 7 * 8.0 / 800

    assert_array_equal(ons_res.infectious_multipliers, np.array([child_freq, adult_freq]))


def test_strat_get_infection_multiplier__with_double_strat_and_both_strats_mixing(backend):
    """
    Check FoI when a two 2-strata stratification applied and both stratifications have a mixing matrix.
    """
    # Create the model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"child": 0.3, "adult": 0.7})
    am = np.array([[2, 3], [5, 7]], dtype=float)
    strat.set_mixing_matrix(am)
    model.stratify_with(strat)
    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    lm = np.array([[11, 13], [17, 19]], dtype=float)
    strat.set_mixing_matrix(lm)
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    expected_mixing = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )

    model_mm = ons_res.ts_graph_vals["mixing_matrix"]

    assert_array_equal(model_mm, expected_mixing)
    assert model._mixing_categories == [
        {"age": "child", "location": "urban"},
        {"age": "child", "location": "rural"},
        {"age": "adult", "location": "urban"},
        {"age": "adult", "location": "rural"},
    ]
    assert model.compartments == [
        C("S", {"age": "child", "location": "urban"}),
        C("S", {"age": "child", "location": "rural"}),
        C("S", {"age": "adult", "location": "urban"}),
        C("S", {"age": "adult", "location": "rural"}),
        C("I", {"age": "child", "location": "urban"}),
        C("I", {"age": "child", "location": "rural"}),
        C("I", {"age": "adult", "location": "urban"}),
        C("I", {"age": "adult", "location": "rural"}),
        C("R", {"age": "child", "location": "urban"}),
        C("R", {"age": "child", "location": "rural"}),
        C("R", {"age": "adult", "location": "urban"}),
        C("R", {"age": "adult", "location": "rural"}),
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(ons_res.initial_population, expected_comp_vals)

    # Do pre-run force of infection calcs.

    exp_lookup = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    assert_array_equal(model._backend._category_lookup, exp_lookup)

    exp_pops = np.array(
        [
            150,  # children at urban
            150,  # children at rural
            350,  # adults at urban
            350,  # adults at rural
        ]
    )

    child_urban_freq = (
        2 * 11 * 1.5 / 150 + 2 * 13 * 1.5 / 150 + 3 * 11 * 3.5 / 350 + 3 * 13 * 3.5 / 350
    )
    child_rural_freq = (
        2 * 17 * 1.5 / 150 + 2 * 19 * 1.5 / 150 + 3 * 17 * 3.5 / 350 + 3 * 19 * 3.5 / 350
    )
    adult_urban_freq = (
        5 * 11 * 1.5 / 150 + 5 * 13 * 1.5 / 150 + 7 * 11 * 3.5 / 350 + 7 * 13 * 3.5 / 350
    )
    adult_rural_freq = (
        5 * 17 * 1.5 / 150 + 5 * 19 * 1.5 / 150 + 7 * 17 * 3.5 / 350 + 7 * 19 * 3.5 / 350
    )

    exp_frequency = np.array(
        [
            child_urban_freq,  # children at urban
            child_rural_freq,  # children at rural
            adult_urban_freq,  # adults at urban
            adult_rural_freq,  # adults at rural
        ]
    )

    assert_allclose(ons_res.infectious_multipliers, exp_frequency, rtol=0, atol=1e-9)
