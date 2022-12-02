"""
Tests for the 'strains' feature of SUMMER.

Strains allows for multiple concurrent infections, which have different properties.

- all infected compartments are stratified into strains (all, not just diseased or infectious, etc)
- assume that a person can only have one strain (simplifying assumption)
- strains can have different infectiousness, mortality rates, etc (set via flow adjustment)
- strains can progress from one to another (inter-strain flows)
- each strain has a different force of infection calculation
- any strain stratification you must be applied to all infected compartments

Force of infection:

- we have multiple infectious populations (one for each strain)
- people infected by a particular strain get that strain
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from summer2 import Compartment as C
from summer2 import CompartmentalModel, StrainStratification, Stratification, adjust


def test_strains__with_two_symmetric_strains(backend):
    """
    Adding two strains with the same properties should yield the same infection dynamics and outputs as having no strains at all.
    We expect the force of infection for each strain to be 1/2 of the unstratified model,
    but the stratification process will not apply the usual conservation fraction to the fan out flows.
    """
    # Create an unstratified model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    model.add_transition_flow("recovery", 0.1, "I", "R")

    # Do pre-run force of infection calcs.

    # Check infectiousness multipliers
    model.run()

    # Create a stratified model where the two strain strata are symmetric
    strain_model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    strain_model.set_initial_population(distribution={"S": 900, "I": 100})
    strain_model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    strain_model.add_transition_flow("recovery", 0.1, "I", "R")
    strat = StrainStratification("strain", ["a", "b"], ["I"])
    strain_model.stratify_with(strat)
    strain_model.run()

    # Ensure stratified model has the same results as the unstratified model.
    merged_outputs = np.zeros_like(model.outputs)
    merged_outputs[:, 0] = strain_model.outputs[:, 0]
    merged_outputs[:, 1] = strain_model.outputs[:, 1] + strain_model.outputs[:, 2]
    merged_outputs[:, 2] = strain_model.outputs[:, 3]
    assert_allclose(merged_outputs, model.outputs, atol=0.01, rtol=0.01, verbose=True)


def test_strain__with_infectious_multipliers(backend):
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different infectiousness levels.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
    strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strat.add_infectiousness_adjustments(
        "I",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as infectious
            "b": adjust.Multiply(3),  # 3x as infectious
            "c": adjust.Multiply(2),  # 2x as infectious
        },
    )
    model.stratify_with(strat)

    # Do pre-run force of infection calcs.

    ons_res = model._get_step_test()

    exp_inf_mul = np.array([0.5, 3, 2]) * np.array([0.7, 0.2, 0.1]) * (100 / 1000)
    assert_allclose(ons_res.infectious_multipliers, exp_inf_mul)

    assert_array_equal(model._backend._category_lookup, np.zeros(5))

    # Get infection flow rates
    comp_rates = ons_res.comp_rates
    sus_pop = 900
    flow_to_a = sus_pop * contact_rate * (70 * 0.5 / 1000)
    flow_to_b = sus_pop * contact_rate * (20 * 3 / 1000)
    flow_to_c = sus_pop * contact_rate * (10 * 2 / 1000)
    expected_comp_rates = np.array(
        [-flow_to_a - flow_to_b - flow_to_c, flow_to_a, flow_to_b, flow_to_c, 0.0]
    )
    assert_allclose(expected_comp_rates, comp_rates, verbose=True)


def test_strain__with_flow_adjustments(backend):
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different flow adjustments.

    These flow adjustments would correspond to some physical process that we're modelling,
    and they should be effectively the same as applying infectiousness multipliers.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")
    strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strat.set_flow_adjustments(
        "infection",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as susceptible
            "b": adjust.Multiply(3),  # 3x as susceptible
            "c": adjust.Multiply(2),  # 2x as susceptible
        },
    )
    model.stratify_with(strat)

    ons_res = model._get_step_test()

    actual_flow_rates = ons_res.flow_rates
    # Get infection flow rates
    # flow_rates = model._backend.get_compartment_rates(model.initial_population, 0)
    sus_pop = 900
    flow_to_a = sus_pop * contact_rate * (70 * 0.5 / 1000)
    flow_to_b = sus_pop * contact_rate * (20 * 3 / 1000)
    flow_to_c = sus_pop * contact_rate * (10 * 2 / 1000)
    expected_flow_rates = np.array((flow_to_a, flow_to_b, flow_to_c))
    assert_allclose(expected_flow_rates, actual_flow_rates, verbose=True)


def test_strain__with_infectious_multipliers_and_heterogeneous_mixing(backend):
    """
    Test infectious multiplier and flow rate calculations for
    3 strains which have different infectiousness levels plus a seperate
    stratification which has a mixing matrix.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    model.set_initial_population(distribution={"S": 900, "I": 100})
    contact_rate = 0.2
    model.add_infection_frequency_flow("infection", contact_rate, "S", "I")

    age_strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    age_strat.set_population_split(
        {
            "child": 0.6,  # 600 people
            "adult": 0.4,  # 400 people
        }
    )
    # Higher mixing among adults or children,
    # than between adults or children.
    age_strat.set_mixing_matrix(np.array([[1.5, 0.5], [0.5, 1.5]]))
    model.stratify_with(age_strat)

    strain_strat = StrainStratification("strain", ["a", "b", "c"], ["I"])
    strain_strat.set_population_split(
        {
            "a": 0.7,  # 70 people
            "b": 0.2,  # 20 people
            "c": 0.1,  # 10 people
        }
    )
    strain_strat.add_infectiousness_adjustments(
        "I",
        {
            "a": adjust.Multiply(0.5),  # 0.5x as susceptible
            "b": adjust.Multiply(3),  # 3x as susceptible
            "c": adjust.Multiply(2),  # 2x as susceptible
        },
    )
    model.stratify_with(strain_strat)

    ons_res = model._get_step_test()

    inf_mul_strains = ons_res.infect_mul_per_strain

    assert_array_equal(
        inf_mul_strains["a"],  # [strain_infectious_idx["a"]],
        np.array(
            [
                0.5 * ((42 / 600) * 1.5 + (28 / 400) * 0.5),
                0.5 * ((42 / 600) * 0.5 + (28 / 400) * 1.5),
            ]
        ),
    )
    assert_array_equal(
        inf_mul_strains["b"],
        np.array(
            [
                3 * ((12 / 600) * 1.5 + (8 / 400) * 0.5),
                3 * ((8 / 400) * 1.5 + (12 / 600) * 0.5),
            ]
        ),
    )
    assert_array_equal(
        inf_mul_strains["c"],
        np.array(
            [2 * ((6 / 600) * 1.5 + (4 / 400) * 0.5), 2 * ((4 / 400) * 1.5 + (6 / 600) * 0.5)]
        ),
    )

    assert_allclose(
        ons_res.infectious_multipliers,
        np.array(
            [
                0.5 * ((42 / 600) * 1.5 + (28 / 400) * 0.5),
                3 * ((12 / 600) * 1.5 + (8 / 400) * 0.5),
                2 * ((6 / 600) * 1.5 + (4 / 400) * 0.5),
                0.5 * ((42 / 600) * 0.5 + (28 / 400) * 1.5),
                3 * ((8 / 400) * 1.5 + (12 / 600) * 0.5),
                2 * ((4 / 400) * 1.5 + (6 / 600) * 0.5),
            ]
        ),
    )

    # Get infection flow rates
    flow_to_inf_child_a = 540 * contact_rate * ons_res.infectious_multipliers[0]
    flow_to_inf_adult_a = 360 * contact_rate * ons_res.infectious_multipliers[3]
    flow_to_inf_child_b = 540 * contact_rate * ons_res.infectious_multipliers[1]
    flow_to_inf_adult_b = 360 * contact_rate * ons_res.infectious_multipliers[4]
    flow_to_inf_child_c = 540 * contact_rate * ons_res.infectious_multipliers[2]
    flow_to_inf_adult_c = 360 * contact_rate * ons_res.infectious_multipliers[5]
    expected_comp_rates = np.array(
        [
            -flow_to_inf_child_a - flow_to_inf_child_b - flow_to_inf_child_c,
            -flow_to_inf_adult_a - flow_to_inf_adult_b - flow_to_inf_adult_c,
            flow_to_inf_child_a,
            flow_to_inf_child_b,
            flow_to_inf_child_c,
            flow_to_inf_adult_a,
            flow_to_inf_adult_b,
            flow_to_inf_adult_c,
            0.0,
            0.0,
        ]
    )
    assert_allclose(expected_comp_rates, ons_res.comp_rates, verbose=True)
