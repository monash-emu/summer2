"""
Basic test of all the flows that CompartmentalModel provides, with no stratifications.
Ensure that the model produces the correct flow rates when run.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from summer2.model import CompartmentalModel


def create_simple_model():
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    return model


def test_apply_flows__with_no_flows(backend):
    """
    Expect no flow to occur because there are no flows.
    """
    model = create_simple_model()
    actual_flow_rates = model._get_step_test().comp_rates
    expected_flow_rates = np.array([0, 0])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_flows__too_many_birth_flows():
    """
    Ensure that misguided attempts to add multiple types of birth flows fail - in four different ways.
    If you want lots of different entries, use the importation flows and adapt them as you need, because the birth flows
    are intended to be pretty rigid in their structures.
    """

    model = create_simple_model()
    model.add_crude_birth_flow("birth", birth_rate=0.05, dest="S")
    with pytest.raises(ValueError):
        model.add_replacement_birth_flow("births", dest="S")

    another_model = create_simple_model()
    another_model.add_replacement_birth_flow("birth", dest="S")
    with pytest.raises(ValueError):
        another_model.add_crude_birth_flow("births", birth_rate=0.05, dest="S")

    another_model_again = create_simple_model()
    another_model_again.add_replacement_birth_flow("birth", dest="S")
    with pytest.raises(ValueError):
        another_model_again.add_replacement_birth_flow("birth", dest="S")

    yet_another_model = create_simple_model()
    yet_another_model.add_crude_birth_flow("births", birth_rate=0.05, dest="S")
    with pytest.raises(ValueError):
        yet_another_model.add_crude_birth_flow("births", birth_rate=0.05, dest="S")


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 99), (500, 500, 50), (0, 1000, 100), (1000, 0, 0)]
)
def test_apply_flows__with_transition_flow__expect_flows_applied(
    backend, inf_pop, sus_pop, exp_flow
):
    """
    Expect a flow to occur proportional to the compartment size and parameter.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_transition_flow("deliberately_infected", 0.1, "S", "I")
    actual_flow_rates = model._get_step_test().comp_rates
    # Expect sus_pop * 0.1 = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
def test_apply_flows__with_infection_frequency(backend, inf_pop, sus_pop, exp_flow):
    """
    Use infection frequency, expect infection multiplier to be proportional
    to the proprotion of infectious to total pop.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_infection_frequency_flow("infection", 20, "S", "I")
    actual_flow_rates = model._get_step_test().comp_rates
    # Expect sus_pop * 20 * (inf_pop / 1000) = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize(
    "inf_pop, sus_pop, exp_flow", [(10, 990, 198), (500, 500, 5000), (0, 1000, 0), (1000, 0, 0)]
)
def test_apply_flows__with_infection_density(backend, inf_pop, sus_pop, exp_flow):
    """
    Use infection density, expect infection multiplier to be proportional
    to the infectious pop.
    """
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": sus_pop, "I": inf_pop})
    model.add_infection_density_flow("infection", 0.02, "S", "I")
    actual_flow_rates = model._get_step_test().comp_rates
    # Expect 0.2 * sus_pop * inf_pop = exp_flow
    expected_flow_rates = np.array([-exp_flow, exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize("inf_pop, exp_flow", [(1000, 100), (990, 99), (50, 5), (0, 0)])
def test_apply_infect_death_flows(backend, inf_pop, exp_flow):
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"I": inf_pop})
    model.add_death_flow("infection_death", 0.1, "I")
    actual_flow_rates = model._get_step_test().comp_rates
    # Expect 0.1 * inf_pop = exp_flow
    expected_flow_rates = np.array([0, -exp_flow])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_universal_death_flow(backend):
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_universal_death_flows("universal_death", 0.1)
    actual_flow_rates = model._get_step_test().comp_rates
    expected_flow_rates = np.array([-99, -1])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


@pytest.mark.parametrize("birth_rate, exp_flow", [[0.0035, 3.5], [0, 0]])
def test_apply_crude_birth_rate_flow(backend, birth_rate, exp_flow):
    """
    Expect births proportional to the total population and birth rate when
    the birth approach is "crude birth rate".
    """
    model = CompartmentalModel(times=[0, 1], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model.add_crude_birth_flow("births", birth_rate, "S")
    model.run(solver="euler")
    # Expect birth_rate * total_population = exp_flow
    expected_outputs = np.array([[990, 10], [990 + exp_flow, 10]])
    assert_array_equal(model.outputs, expected_outputs)


def test_apply_replace_death_birth_flow(backend):
    model = CompartmentalModel(times=[0, 5], compartments=["S", "I"], infectious_compartments=["I"])
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_death_flow("infection_death", 0.1, "I")
    model.add_universal_death_flows("universal_death", 0.05)
    model.add_replacement_birth_flow("births", "S")
    actual_flow_rates = model._get_step_test().comp_rates
    # Expect 10 people to die and 10 to be born
    exp_i_flow_rate = -0.1 * 100 - 0.05 * 100
    exp_s_flow_rate = -exp_i_flow_rate  # N.B births + deaths in the S compartment should balance.
    expected_flow_rates = np.array([exp_s_flow_rate, exp_i_flow_rate])
    assert_array_equal(actual_flow_rates, expected_flow_rates)


def test_apply_many_flows(backend):
    """
    Expect multiple flows to operate independently and produce the correct final flow rate.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    model.add_death_flow("infection_death", 0.1, "I")
    model.add_universal_death_flows("universal_death", 0.1)
    model.add_infection_frequency_flow("infection", 0.2, "S", "I")
    model.add_transition_flow("recovery", 0.1, "I", "R")
    model.add_transition_flow("vaccination", 0.1, "S", "R")
    model.add_crude_birth_flow("births", 0.1, "S")
    actual_flow_rates = model._get_step_test().comp_rates

    # Expect the effects of all these flows to be linearly superimposed.
    infect_death_flows = np.array([0, -10, 0])
    universal_death_flows = np.array([-90, -10, 0])
    infected = 900 * 0.2 * (100 / 1000)
    infection_flows = np.array([-infected, infected, 0])
    recovery_flows = np.array([0, -10, 10])
    vaccination_flows = np.array([-90, 0, 90])
    birth_flows = np.array([100, 0, 0])
    expected_flow_rates = (
        infect_death_flows
        + universal_death_flows
        + infection_flows
        + recovery_flows
        + vaccination_flows
        + birth_flows
    )
    assert_array_equal(actual_flow_rates, expected_flow_rates)
