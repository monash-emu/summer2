"""
Ensure infectiousness adjustments are applied correctly in stratification.
See See https://parasiteecology.wordpress.com/2013/10/17/density-dependent-vs-frequency-dependent-disease-transmission/
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from summer2 import Compartment as C
from summer2 import CompartmentalModel, StrainStratification, Stratification, adjust
from summer2.population import calculate_initial_population


def test_strat_infectiousness__with_adjustments(backend):
    """
    Ensure multiply infectiousness adjustment is applied.
    """
    # Create a model
    def get_model(mode: str):
        model = CompartmentalModel(
            times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
        )
        # model._set_backend(backend)
        model.set_initial_population(distribution={"S": 900, "I": 100})
        if mode == "freq":
            model.add_infection_frequency_flow("infection", 1.0, "S", "I")
        elif mode == "dens":
            model.add_infection_density_flow("infection", 1.0, "S", "I")
        strat = Stratification("age", ["baby", "child", "adult"], ["S", "I", "R"])
        strat.set_population_split({"baby": 0.1, "child": 0.3, "adult": 0.6})
        strat.add_infectiousness_adjustments(
            "I", {"child": adjust.Multiply(3), "adult": adjust.Multiply(0.5), "baby": None}
        )
        model.stratify_with(strat)
        return model

    model = get_model("freq")

    assert_array_equal(
        calculate_initial_population(model),
        np.array([90, 270, 540, 10, 30, 60, 0, 0, 0]),
    )

    # Do pre-run force of infection calcs.
    # model._backend.prepare_to_run()
    ons_res = model._get_step_test()

    assert_array_equal(
        ons_res.model_data["compartment_infectiousness"]["default"],
        np.array([1, 3, 0.5]),
    )

    # Do pre-iteration force of infection calcs

    # Get multipliers
    expected_density = 10 * 1 + 30 * 3 + 60 * 0.5
    expected_frequency = expected_density / 1000

    assert_array_equal(ons_res.infectious_multipliers, expected_frequency)

    model = get_model("dens")
    ons_res = model._get_step_test()
    assert_array_equal(ons_res.infectious_multipliers, expected_density)


def test_strat_infectiousness__with_multiple_adjustments(backend):
    """
    Ensure multiply infectiousness adjustment is applied.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model._set_backend(backend)
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_infection_frequency_flow("infection", 1.0, "S", "I")
    strat = Stratification("age", ["baby", "child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"baby": 0.1, "child": 0.3, "adult": 0.6})
    strat.add_infectiousness_adjustments(
        "I", {"child": adjust.Multiply(3), "adult": adjust.Multiply(0.5), "baby": None}
    )
    model.stratify_with(strat)
    assert_array_equal(
        calculate_initial_population(model),
        np.array([90, 270, 540, 10, 30, 60, 0, 0, 0]),
    )
    # Stratify again, now with overwrites
    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    strat.add_infectiousness_adjustments(
        "I", {"urban": adjust.Overwrite(1), "rural": adjust.Multiply(7)}
    )
    model.stratify_with(strat)
    assert_array_equal(
        calculate_initial_population(model),
        np.array([45, 45, 135, 135, 270.0, 270, 5, 5, 15, 15, 30, 30, 0, 0, 0, 0, 0, 0]),
    )

    ons_res = model._get_step_test()

    assert_array_equal(
        ons_res.model_data["compartment_infectiousness"]["default"],
        np.array([1, 7, 1, 21, 1, 3.5]),
    )

    # Get multipliers
    expected_density = 5 * 1 + 5 * 7 + 15 * 1 + 15 * 21 + 30 * 1 + 30 * 3.5
    expected_frequency = expected_density / 1000

    assert_array_equal(ons_res.infectious_multipliers, expected_frequency)
