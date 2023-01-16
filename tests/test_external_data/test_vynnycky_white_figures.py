from jax import numpy as jnp
import pandas as pd
from pathlib import Path
import numpy as np

from summer2 import CompartmentalModel, Stratification, Multiply
from summer2.parameters import Parameter, DerivedOutput, Time, Function

TEST_OUTPUTS_PATH = Path(__file__).with_name("vynnycky_white_figures")
TOLERANCE = 1e-9


def test_2_08():

    config = {
        "population": 1e5,
        "infectious_seed": 1.0,
        "end_time": 200.0,
    }
    parameters = {
        "r0": 2.0,
        "infous_rate": 1.0 / 2.0,
        "rec_rate": 1.0 / 2.0,
    }

    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0.0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=("Infectious",),
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    rec_rate = Parameter("rec_rate")
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=Parameter("r0") * rec_rate,
        source="Susceptible",
        dest="Pre-infectious",
    )
    model.add_transition_flow(
        name="progression",
        fractional_rate=Parameter("infous_rate"),
        source="Pre-infectious",
        dest="Infectious",
    )
    model.add_transition_flow(
        name="recovery", fractional_rate=rec_rate, source="Infectious", dest="Immune"
    )
    model.request_output_for_flow(
        name="incidence",
        flow_name="progression",
        raw_results=True,
    )

    model.run(parameters=parameters)
    compartments = model.get_outputs_df()

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "2_08_outputs.csv", index_col=0)
    model_results = pd.concat(
        (compartments[["Susceptible", "Immune"]], model.get_derived_outputs_df()), axis=1
    )

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_03():

    config = {
        "total_population": 1e5,
        "infectious_seed": 1.0,
        "end_time": 120.0,
    }

    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0.0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible",
        dest="Pre-infectious",
    )
    model.add_transition_flow(
        name="progression",
        fractional_rate=1.0 / Parameter("latent_period"),
        source="Pre-infectious",
        dest="Infectious",
    )
    model.add_transition_flow(
        name="recovery", fractional_rate=1.0 / infectious_period, source="Infectious", dest="Immune"
    )
    model.request_output_for_flow(name="incidence", flow_name="progression")
    model.request_output_for_compartments(
        name="n_suscept",
        compartments=["Susceptible"],
        save_results=False,
    )
    model.request_function_output(
        name="suscept_prop", func=DerivedOutput("n_suscept") / config["total_population"]
    )
    model.request_function_output(
        name="Rn", func=Parameter("r0") * DerivedOutput("suscept_prop")
    )

    # Measles
    parameters = {
        "r0": 13.0,
        "latent_period": 7.0,
        "infectious_period": 8.0,
    }

    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_03_measles_outputs.csv", index_col=0)
    model_results = pd.concat(
        (
            model.get_outputs_df()["Susceptible"] / config["total_population"],
            model.get_derived_outputs_df()["incidence"],
            model.get_derived_outputs_df()["Rn"],
        ),
        axis=1,
    )

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE

    # Run for flu
    parameters = {
        "r0": 2.,
        "infectious_period": 2.,
        "latent_period": 2.,
    }

    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_03_flu_outputs.csv", index_col=0)
    model_results = pd.concat(
        (
            model.get_outputs_df()["Susceptible"] / config["total_population"],
            model.get_derived_outputs_df()["incidence"],
            model.get_derived_outputs_df()["Rn"],
        ),
        axis=1,
    )

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_04():

    config = {
        "total_population": 1e5,
        "infectious_seed": 1.,
        "end_time": 120.,
    }
    parameters = {
        "r0": 13.,
        "latent_period": 7.,
        "infectious_period": 8.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune"
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression"
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_04_measles_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = pd.concat(
        (
            model.get_outputs_df()["Susceptible"] / config["total_population"], 
            model.get_outputs_df()["Immune"] / config["total_population"], 
            model.get_derived_outputs_df()["incidence"],
        ), 
        axis=1,
    )

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE

    parameters = {
        "r0": 2.,
        "infectious_period": 2.,
        "latent_period": 2.,
    }

    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_04_flu_outputs.csv", index_col=0)
    model_results = pd.concat(
        (
            model.get_outputs_df()["Susceptible"] / config["total_population"], 
            model.get_outputs_df()["Immune"] / config["total_population"], 
            model.get_derived_outputs_df()["incidence"],
        ), 
        axis=1,
    )

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_05():

    config = {
        "total_population": 1e5,
        "infectious_seed": 1.,
        "end_time": 2e3,
    }
    parameters = {
        "r0": 13.,
        "latent_period": 8.,
        "infectious_period": 7.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] * (1. - Parameter("prop_recovered")) - config["infectious_seed"], 
            "Infectious": config["infectious_seed"],
            "Immune": config["total_population"] * Parameter("prop_recovered"),
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune"
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression"
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_05_measles_outputs.csv", index_col=0)
    model_results = pd.DataFrame()
    immune_props = (0., 0.9, 0.92, 0.923, 0.93, 0.95)
    for immune_prop in immune_props:
        parameters.update({"prop_recovered": immune_prop})
        model.run(parameters=parameters, solver="euler")
        model_results[str(immune_prop)] = model.get_derived_outputs_df()["incidence"]

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE

    parameters.update(
        {
            "r0": 2.,
            "latent_period": 2.,
            "infectious_period": 2.,
        }
    )
    immune_props = (0., 0.45, 0.49, 0.5, 0.51, 0.55)

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_05_flu_outputs.csv", index_col=0)
    model_results = pd.DataFrame()
    for immune_prop in immune_props:
        parameters.update({"prop_recovered": immune_prop})
        model.run(parameters=parameters, solver="euler")
        model_results[str(immune_prop)] = model.get_derived_outputs_df()["incidence"]

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_06():

    config = {
        "total_population": 1e5,
        "infectious_seed": 1.,
        "end_time": 120.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=("Infectious",),
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"], 
            "Infectious": config["infectious_seed"]
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune"
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression",
        raw_results=True,
    )

    # Measles
    parameters = {
        "r0": 13.,
        "latent_period": 8.,
        "infectious_period": 7.,
    }
    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_06_measles_outputs.csv", index_col=0)
    model_results = pd.DataFrame()
    model_results["Prevalence of infectious individuals"] = model.get_outputs_df()["Infectious"]
    model_results["New infectious individuals/day"] = model.get_derived_outputs_df()["incidence"]
    model_results["Cumulative number of infectious individuals"] = model_results.loc[:, "New infectious individuals/day"].cumsum()

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE

    # Flu
    parameters = {
        "r0": 2.,
        "latent_period": 2.,
        "infectious_period": 2.,
    }
    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_06_flu_outputs.csv", index_col=0)
    model_results = pd.DataFrame()
    model_results["Prevalence of infectious individuals"] = model.get_outputs_df()["Infectious"]
    model_results["New infectious individuals/day"] = model.get_derived_outputs_df()["incidence"]
    model_results["Cumulative number of infectious individuals"] = model_results.loc[:, "New infectious individuals/day"].cumsum()

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_12():

    config = {
        "end_time": 20.,
        "total_population": 1.,
        "infectious_seed": 0.001,
    }
    parameters = {
        "infectious_period": 1.,
    }

    compartments = (
        "susceptible",
        "infectious",
        "recovered",
    )
    analysis_times = (0., config["end_time"])
    model = CompartmentalModel(
        times=analysis_times,
        compartments=compartments,
        infectious_compartments=["infectious"],
        timestep=0.01,
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["total_population"] - config["infectious_seed"], 
            "infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="infectious", 
        dest="recovered",
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_12_outputs.csv", index_col=0)["0"]
    r0s = np.linspace(0.99, 15., 100)
    model_results = pd.Series(index=range(len(r0s)))
    for i_r0, r0 in enumerate(r0s):
        parameters.update({"r0": r0})
        model.run(parameters=parameters, solver="euler")
        model_results[i_r0] = 1. - model.get_outputs_df().loc[config["end_time"], "susceptible"]

    differences = model_results - expected_results
    assert max(differences.abs()) < TOLERANCE


def test_4_17():

    config = {
        "end_time": 25550.,
        "total_population": 1e5,
        "infectious_seed": 1.,
    }
    parameters = {
        "latent_period": 8.,
        "infectious_period": 7.,
        "r0": 13.,
        "life_expectancy": 70.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune",
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune",
    )
    model.add_replacement_birth_flow(
        "births",
        "Susceptible",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.,
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression",
    )
    model.request_output_for_compartments(
        name="total_population",
        compartments=compartments,
    )
    model.request_function_output(
        name="incidence_rate",
        func=DerivedOutput("incidence") / DerivedOutput("total_population") * 1e5,
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_17_outputs.csv", index_col=0)["incidence"]
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()["incidence"]

    differences = model_results - expected_results
    assert max(differences.abs()) < TOLERANCE


def test_4_19():
    
    config = {
        "end_time": 25550.,
        "total_population": 1e5,
        "infectious_seed": 1.,
    }
    parameters = {
        "latent_period": 8.,
        "infectious_period": 7.,
        "r0": 13.,
        "life_expectancy": 70.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune",
    )
    model.add_replacement_birth_flow(
        "births",
        "Susceptible",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.,
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression",
    )
    model.request_output_for_compartments(
        name="total_population",
        compartments=compartments,
    )
    model.request_output_for_compartments(
        name="susceptible_compartment",
        compartments=["Susceptible"],
    )
    model.request_output_for_compartments(
        name="immune_compartment",
        compartments=["Immune"],
    )
    total_pop = DerivedOutput("total_population")
    model.request_function_output(
        name="incidence_rate",
        func=DerivedOutput("incidence") / total_pop * 1e5,
    )
    model.request_function_output(
        name="susceptible_prop",
        func=DerivedOutput("susceptible_compartment") / total_pop,
    )
    model.request_function_output(
        name="immune_prop",
        func=DerivedOutput("immune_compartment") / total_pop,
    )
    model.request_function_output(
        name="R_n",
        func=DerivedOutput("susceptible_prop") * Parameter("r0"),
    )
    
    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_19_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_26():

    config = {
        "end_time": 25550.,
        "total_population": 1e5,
        "infectious_seed": 1.,
    }
    parameters = {
        "latent_period": 8.,
        "infectious_period": 7.,
        "r0": 13.,
        "life_expectancy": 70.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune",
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune",
    )
    model.add_crude_birth_flow(
        "births",
        Parameter("crude_birth_rate") / 365.,
        "Susceptible",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.,
    )
    model.request_output_for_flow(
        name="incidence", 
        flow_name="progression",
    )
    model.request_output_for_compartments(
        name="total_population",
        compartments=compartments,
    )
    model.request_function_output(
        name="incidence_rate",
        func=DerivedOutput("incidence") / DerivedOutput("total_population") * 1e5,
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_26_outputs.csv", index_col=0)
    birth_rates = (0.015, 0.025, 0.04)
    model_results = pd.DataFrame(columns=birth_rates)
    for rate in birth_rates:
        parameters.update({"crude_birth_rate": rate})
        model.run(parameters=parameters, solver="euler")
        model_results[str(rate)] = model.get_derived_outputs_df()["incidence_rate"]

    differences = model_results - expected_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_29():

    config = {
        "total_population": 1e5,
        "infectious_seed": 1.,
        "start_time": -108.,
        "end_time": 60.,
    }
    parameters = {
        "r0": 13.,
        "latent_period": 8.,
        "infectious_period": 7.,
        "life_expectancy": 70.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(
            config["start_time"] * 365.,
            config["end_time"] * 365.,
        ),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"], 
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    life_expectancy = Parameter("life_expectancy")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / life_expectancy / 365.,
    )
    model.add_crude_birth_flow(
        "births",
        1. / life_expectancy / 365.,
        "Susceptible",
    )
    model.request_output_for_compartments(
        "total_population",
        compartments,
    )
    model.request_output_for_flow(
        name="incidence",
        flow_name="progression",
    )
    model.request_function_output(
        name="incidence_rate",
        func=DerivedOutput("incidence") / DerivedOutput("total_population") * 1e5,
    )
    vacc_strat = Stratification(
        "vaccination",
        ["vaccinated", "unvaccinated"],
        ["Susceptible"],
    )
    vacc_strat.set_population_split(
        {
            "vaccinated": 0.,
            "unvaccinated": 1.,
        }
    )
    vacc_strat.set_flow_adjustments(
        flow_name="infection",
        adjustments={
            "vaccinated": 0.,
            "unvaccinated": 1.,
        },
    )
    
    def step_up(time, vacc_coverage):
        return jnp.where(time > 0., vacc_coverage, 0.)
    
    def step_down(time, vacc_coverage):
        return jnp.where(time > 0., 1. - vacc_coverage, 1.)
    
    vacc_strat.set_flow_adjustments(
        flow_name="births",
        adjustments={
            "vaccinated": Function(step_up, [Time, Parameter("vacc_coverage")]),
            "unvaccinated": Function(step_down, [Time, Parameter("vacc_coverage")]),
        },
    )
    model.stratify_with(vacc_strat)

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_29_outputs.csv", index_col=0)
    coverage_values = (0.5, 0.8, 0.9)
    model_results = pd.DataFrame(columns=coverage_values)
    for coverage in coverage_values:
        parameters.update({"vacc_coverage": coverage})
        model.run(parameters=parameters, solver="euler")
        model_results[str(coverage)] = model.get_derived_outputs_df()["incidence_rate"]

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE


def test_4_31_sis():

    config = {
        "start_time": 0.,
        "end_time": 50. * 365.,
        "population": 1.,
        "seed": 1e-5,
    }
    parameters = {
        "contact_rate": 1.5 / 60.,
        "recovery": 1. / 60.,
    }

    compartments = (
        "susceptible",
        "infectious",
    )
    analysis_times = (
        config["start_time"], 
        config["end_time"],
    )
    model = CompartmentalModel(
        times=analysis_times,
        compartments=compartments,
        infectious_compartments=["infectious"],
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=Parameter("recovery"),
        source="infectious", 
        dest="susceptible",
    )
    model.request_output_for_flow(
        "incidence",
        "infection",
        save_results=False,
    )
    model.request_function_output(
        "incidence_rate",
        func=DerivedOutput("incidence") * 1e5 * 30,
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_31_sis_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()

    differences = expected_results - model_results
    assert differences["incidence_rate"].abs().max() < TOLERANCE


def test_4_31_sirs():

    config = {
        "start_time": 0.,
        "end_time": 50. * 365.,
        "population": 1.,
        "seed": 1e-5,
    }
    parameters = {
        "contact_rate": 1.5 / 60.,
        "recovery": 1. / 60.,
    }

    compartments = (
        "susceptible",
        "infectious",
        "recovered",
    )
    analysis_times = (
        config["start_time"], 
        config["end_time"],
    )
    model = CompartmentalModel(
        times=analysis_times,
        compartments=compartments,
        infectious_compartments=["infectious"],
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=Parameter("recovery"),
        source="infectious", 
        dest="recovered",
    )
    model.add_transition_flow(
        name="waning",
        fractional_rate=Parameter("waning"),
        source="recovered",
        dest="susceptible",
    )
    model.request_output_for_flow(
        "incidence",
        "infection",
        save_results=False,
    )
    model.request_function_output(
        "incidence_rate",
        func=DerivedOutput("incidence") * 1e5 * 30,
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_31_sirs_outputs.csv", index_col=0)
    wane_rates = (8, 10, 12)
    model_results = pd.DataFrame(columns=wane_rates)
    for wane_rate in wane_rates:
        parameters.update(
            {
                "waning": (1. / wane_rate + 1. / 30.) / 365.,
            }
        )
        model.run(parameters=parameters, solver="euler")
        model_results[str(wane_rate)] = model.get_derived_outputs_df()["incidence_rate"]
    model_results.index = model_results.index / 365.

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE
 

def test_5_02():

    config = {
        "end_time": 100. * 365.,
        "total_population": 1e5,
        "infectious_seed": 1.,
    }
    parameters = {
        "latent_period": 8.,
        "infectious_period": 7.,
        "r0": 13.,
        "life_expectancy": 70.,
    }

    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    infectious_period = Parameter("infectious_period")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / infectious_period,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("latent_period"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / infectious_period,
        source="Infectious", 
        dest="Immune",
    )
    model.add_replacement_birth_flow(
        "births",
        "Susceptible",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.,
    )
    model.request_output_for_compartments(
        name="infectious",
        compartments="Infectious",
        save_results=False,
    )
    model.request_output_for_compartments(
        name="total_population",
        compartments=compartments,
        save_results=False,
    )
    model.request_function_output(
        name="prevalence",
        func=DerivedOutput("infectious") / DerivedOutput("total_population") * 1e5,
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "5_02_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()

    differences = expected_results - model_results
    assert differences["prevalence"].abs().max() < TOLERANCE


def test_5_13():

    def build_demog_model(vacc_coverage):

        config = {
            "start_time": -72.,
            "end_time": 30.,
            "total_population": 1e5,
            "infectious_seed": 1.,
        }

        compartments = (
            "Susceptible", 
            "Pre-infectious", 
            "Infectious", 
            "Immune"
        )
        model = CompartmentalModel(
            times=(
                config["start_time"] * 365.,
                config["end_time"] * 365.,
            ),
            compartments=compartments,
            infectious_compartments=["Infectious"],
        )
        model.set_initial_population(
            distribution={
                "Susceptible": config["total_population"] - config["infectious_seed"], 
                "Infectious": config["infectious_seed"],
            }
        )
        infectious_period = Parameter("infectious_period")
        life_expectancy = Parameter("life_expectancy")
        model.add_infection_frequency_flow(
            name="infection", 
            contact_rate=Parameter("r0") / infectious_period,
            source="Susceptible", 
            dest="Pre-infectious"
        )
        model.add_transition_flow(
            name="progression", 
            fractional_rate=1. / Parameter("latent_period"),
            source="Pre-infectious", 
            dest="Infectious"
        )
        model.add_transition_flow(
            name="recovery", 
            fractional_rate=1. / infectious_period,
            source="Infectious", 
            dest="Immune",
        )
        model.add_universal_death_flows(
            "universal_death",
            death_rate=1. / life_expectancy / 365.,
        )
        model.add_crude_birth_flow(
            "births",
            1. / life_expectancy / 365.,
            "Susceptible",
        )
        model.request_output_for_compartments(
            "total_population",
            compartments,
        )
        model.request_output_for_flow(
            name="incidence",
            flow_name="progression",
        )
        model.request_function_output(
            name="incidence_rate",
            func=DerivedOutput("incidence") / DerivedOutput("total_population") * 1e5
        )
        vacc_strat = Stratification(
            "vaccination",
            ["vaccinated", "unvaccinated"],
            ["Susceptible"],
        )
        vacc_strat.set_population_split(
            {
                "vaccinated": 0.,
                "unvaccinated": 1.,
            }
        )
        vacc_strat.set_flow_adjustments(
            flow_name="infection",
            adjustments={
                "vaccinated": 0.,
                "unvaccinated": 1.,
            },
        )
        
        def step_up(time, values):
            return jnp.where(time > 0., vacc_coverage, 0.)
        
        def step_down(time, values):
            return jnp.where(time > 0., 1. - vacc_coverage, 1.)
        
        vacc_strat.set_flow_adjustments(
            flow_name="births",
            adjustments={
                "vaccinated": step_up,
                "unvaccinated": step_down,
            },
        )
        model.stratify_with(vacc_strat)
        return model

    parameters = {
        "r0": 13.,
        "latent_period": 8.,
        "infectious_period": 7.,
        "life_expectancy": 70.,
    }

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "5_13_outputs.csv", index_col=0)
    coverage_values = (0., 0.6, 0.75)
    model_results = pd.DataFrame(columns=coverage_values)
    for coverage in coverage_values:
        vacc_model = build_demog_model(coverage)
        vacc_model.run(parameters=parameters, solver="euler")
        model_results[str(coverage)] = vacc_model.get_outputs_df()["Infectious"]

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE


def test_8_05():

    config = {
        "start_time": 0.,
        "end_time": 10. * 365.,
        "population": 1.,
        "seed": 0.01,
    }
    parameters = {
        "recovery": 6. / 365.,
        "contact_rate": 0.75 / 365.,
    }

    compartments = (
        "susceptible",
        "infectious",
    )
    analysis_times = (
        config["start_time"], 
        config["end_time"],
    )
    model = CompartmentalModel(
        times=analysis_times,
        compartments=compartments,
        infectious_compartments=["infectious"],
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=Parameter("recovery"),
        source="infectious", 
        dest="susceptible",
    )
    model.request_output_for_compartments(
        "infectious",
        ["infectious"],
        save_results=False,
    )
    model.request_output_for_compartments(
        "total",
        compartments,
        save_results=False,
    )
    model.request_function_output(
        "prevalence",
        DerivedOutput("infectious") / DerivedOutput("total")
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "8_05_outputs.csv", index_col=0)["prevalence"]
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()["prevalence"]

    differences = expected_results - model_results
    assert max(differences.abs()) < TOLERANCE


def test_8_08():

    config = {
        "end_time": 10. * 365.,
        "population": 1.,
        "seed": 1e-6,
    }
    parameters = {
        "recovery": 6. / 365.,
        "contact_rate": 0.75 / 365.,
        "high_prop": 0.02,
        "average_partner_change": 2.,
        "low_partner_change": 1.4,
    }

    compartments = (
        "susceptible",
        "infectious",
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["infectious"],
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=Parameter("recovery"),
        source="infectious", 
        dest="susceptible",
    )
    activity_strat = Stratification(
        "activity",
        ["High", "Low"],
        compartments,
    )
    high_prop = Parameter("high_prop")
    low_prop = 1. - high_prop
    activity_strat.set_population_split(
        {
            "High": high_prop,
            "Low": low_prop,
        }
    )
    low_partner_change = Parameter("low_partner_change")
    high_partner_change = (Parameter("average_partner_change") - low_partner_change * low_prop) / high_prop  # Equation 8.15
    high_change_rate_abs = high_partner_change * high_prop  # Absolute partner change rate, high stratum
    low_change_rate_abs = low_partner_change * low_prop  # Absolute partner change rate, low stratum
    total_change_rate = high_change_rate_abs + low_change_rate_abs  # Total rate of partner changes across the population
    high_change_prop = high_change_rate_abs / total_change_rate  # Equation 8.20
    low_change_prop = low_change_rate_abs / total_change_rate
    
    def build_matrix(high_change_prop, low_change_prop):
        mixing_matrix = jnp.array([[high_change_prop, low_change_prop]])  # The "g" values
        mixing_matrix = jnp.repeat(mixing_matrix, 2, axis=0)  # Double up to a square array
        return mixing_matrix
    
    mixing_matrix = Function(build_matrix, (high_change_prop, low_change_prop))
    activity_strat.set_mixing_matrix(mixing_matrix)
    activity_strat.set_flow_adjustments(
        "infection",
        {
            "High": Multiply(high_partner_change),  # Or multiply top row of matrix by this
            "Low": Multiply(low_partner_change),  # Or multiply bottom row of matrix by this
        },
    )
    model.stratify_with(activity_strat)
    model.request_output_for_compartments(
        "infectious",
        ["infectious"],
        save_results=False,
    )
    model.request_output_for_compartments(
        "total",
        compartments,
        save_results=False,
    )
    model.request_function_output(
        "Overall",
        DerivedOutput("infectious") / DerivedOutput("total"), 
    )
    for stratum in ["High", "Low"]:
        model.request_output_for_compartments(
            f"infectiousX{stratum}",
            ["infectious"],
            strata={"activity": stratum},
            save_results=False,
        )
        model.request_output_for_compartments(
            f"totalX{stratum}",
            compartments,
            strata={"activity": stratum},
            save_results=False,
        )
        model.request_function_output(
            stratum,
            DerivedOutput(f"infectiousX{stratum}") / 
            DerivedOutput(f"totalX{stratum}")
        )
    model.request_output_for_flow(
        "incidence",
        "infection",
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "8_08_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()
    model_results.index = model_results.index / 365.
    model_results = model_results * 100.

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE


def test_8_09():

    config = {
        "end_time": 1000.,
        "population": 1.,
        "seed": 1e-6,
    }
    parameters = {
        "recovery": 6.,
        "contact_rate": 0.75,
        "high_prop": 0.02,
        "average_partner_change": 2.,
    }

    compartments = (
        "susceptible",
        "infectious",
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["infectious"],
        timestep=0.01,
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=Parameter("recovery"),
        source="infectious", 
        dest="susceptible",
    )
    activity_strat = Stratification(
        "activity",
        ["High", "Low"],
        compartments,
    )

    high_prop = Parameter("high_prop")
    low_prop = 1. - high_prop
    activity_strat.set_population_split(
        {
            "High": high_prop,
            "Low": low_prop,
        }
    )
    low_partner_change = Parameter("low_partner_change")
    high_partner_change = (Parameter("average_partner_change") - low_partner_change * low_prop) / high_prop
    high_change_rate_abs = high_partner_change * high_prop
    low_change_rate_abs = low_partner_change * low_prop
    total_change_rate = high_change_rate_abs + low_change_rate_abs
    high_change_prop = high_change_rate_abs / total_change_rate
    low_change_prop = low_change_rate_abs / total_change_rate
    
    def build_matrix(high_change_prop, low_change_prop):
        mixing_matrix = jnp.array([[high_change_prop, low_change_prop]])
        mixing_matrix = jnp.repeat(mixing_matrix, 2, axis=0)
        return mixing_matrix
    
    mixing_matrix = Function(build_matrix, (high_change_prop, low_change_prop))
    activity_strat.set_flow_adjustments(
        "infection",
        {
            "High": Multiply(high_partner_change),
            "Low": Multiply(low_partner_change),
        },
    )
    activity_strat.set_mixing_matrix(mixing_matrix)
    model.stratify_with(activity_strat)
    model.request_output_for_compartments(
        "infectious",
        ["infectious"],
        save_results=False,
    )
    model.request_output_for_compartments(
        "total",
        compartments,
        save_results=False,
    )
    model.request_function_output(
        "Overall",
        DerivedOutput("infectious") / 
        DerivedOutput("total")
    )
    for stratum in ["High", "Low"]:
        model.request_output_for_compartments(
            f"infectiousX{stratum}",
            ["infectious"],
            strata={"activity": stratum},
            save_results=False,
        )
        model.request_output_for_compartments(
            f"totalX{stratum}",
            compartments,
            strata={"activity": stratum},
            save_results=False,
        )
        model.request_function_output(
            stratum,
            DerivedOutput(f"infectiousX{stratum}") / 
            DerivedOutput(f"totalX{stratum}")
        )
    model.request_output_for_flow(
        "incidence",
        "infection",
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "8_09_outputs.csv", index_col=0)
    low_change_rates = np.concatenate((np.linspace(2., 1., 21), np.linspace(0.9, 0., 6)))
    output_groups = ["Overall", "Low", "High"]
    model_results = pd.DataFrame(index=range(len(low_change_rates)), columns=output_groups)
    for i_change, low_change in enumerate(low_change_rates):
        parameters.update({"low_partner_change": low_change})
        model.run(parameters=parameters, solver="euler")
        model_results.loc[i_change] = model.get_derived_outputs_df().loc[config["end_time"], :] * 100.

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE


def build_matrix(high_prop, high_partner_change, low_partner_change, ghh):
    low_prop = 1. - high_prop
    glh = 1. - ghh
    ghl = (1. - ghh) * high_partner_change * high_prop / (low_partner_change * low_prop)
    gll = 1. - ghl
    matrix = jnp.array(
        [
            [ghh, ghl],
            [glh, gll],
        ]
    )
    return matrix.T


def test_8_14():

    config = {
        "end_time": 20. * 365.,
        "population": 1.,
        "seed": 1e-6,
    }
    parameters = {
        "contact_rate": 0.75 / 365.,
        "high_partner_change": 31.4,
        "low_partner_change": 1.4,
        "high_prop": 0.02,
    }
    updates = {
        "recovery": [0.34, 0.167, 0.097],
        "ghh": [0.0396, 0.314, 0.5884],
    }

    compartments = (
        "susceptible",
        "infectious",
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=["infectious"],
    )
    model.set_initial_population(
        distribution=
        {
            "susceptible": config["population"] - config["seed"], 
            "infectious": config["seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="susceptible", 
        dest="infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / Parameter("recovery") / 365.,
        source="infectious", 
        dest="susceptible",
    )
    activity_strat = Stratification(
        "activity",
        ["High", "Low"],
        compartments,
    )
    high_prop = Parameter("high_prop")
    low_prop = 1. - high_prop
    high_partner_change = Parameter("high_partner_change")
    low_partner_change = Parameter("low_partner_change")
    activity_strat.set_population_split(
        {
            "High": high_prop,
            "Low": low_prop,
        }
    )
    mixing_matrix = Function(
        build_matrix, 
        (high_prop, high_partner_change, low_partner_change, Parameter("ghh"))
    )
    activity_strat.set_mixing_matrix(mixing_matrix)
    activity_strat.set_flow_adjustments(
        "infection",
        {
            "High": Multiply(high_partner_change),
            "Low": Multiply(low_partner_change),
        },
    )

    model.stratify_with(activity_strat)
    model.request_output_for_compartments(
        "infectious",
        ["infectious"],
        save_results=False,
    )
    model.request_output_for_compartments(
        "total",
        compartments,
        save_results=False,
    )
    model.request_function_output(
        "Overall",
        DerivedOutput("infectious") / 
        DerivedOutput("total")
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "8_14_outputs.csv", index_col=0)
    model_names = ("More with-unlike", "Proportionate", "More with-like")
    model_results = pd.DataFrame(columns=model_names)
    for i_model, name in enumerate(model_names):
        parameters["recovery"] = updates["recovery"][i_model]
        ghh = updates["ghh"][i_model]
        parameters["ghh"] = ghh
        model.run(parameters=parameters, solver="euler")
        model_results[name] = model.get_derived_outputs_df()["Overall"]
    model_results *= 100.0

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE


def test_8_20():

    config = {
        "total_population": 1e4,
        "infectious_seed": 10.,
        "end_time": 100.,
    }
    parameters = {
        "high_prop": 0.15,
        "high_partner_change_rate": 8.,
        "low_partner_change_rate": 0.2,
        "infectious_period": 9.,
        "expectancy_at_debut": 35.,
        "aids_period": 1.,
        "contact_rate": 0.05,
    }

    compartments = (
        "Susceptible", 
        "Infectious", 
        "AIDS"
    )
    model = CompartmentalModel(
        times=(0., config["end_time"]),
        compartments=compartments,
        infectious_compartments=("Infectious",),
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["total_population"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("contact_rate"),
        source="Susceptible",
        dest="Infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("infectious_period"),
        source="Infectious", 
        dest="AIDS"
    )
    model.add_universal_death_flows(
        "non_aids_mortality",
        1. / Parameter("expectancy_at_debut"),
    )
    model.add_replacement_birth_flow(
        "recruitment",
        "Susceptible",
    )
    model.add_death_flow(
        "aids_mortality",
        1. / Parameter("aids_period"),
        "AIDS",
    )
    activity_strata = ("High", "Low")
    activity_strat = Stratification(
        "activity",
        activity_strata,
        compartments,
    )
    high_prop = Parameter("high_prop")
    low_prop = 1. - high_prop
    activity_strat.set_population_split(
        {
            "High": high_prop,
            "Low": low_prop,
        }
    )
    activity_strat.set_flow_adjustments(
        "recruitment",
        adjustments={
            "High": high_prop,
            "Low": low_prop,
        },
    )
    high_prop = Parameter("high_prop")
    high_rate = Parameter("high_partner_change_rate")
    low_rate = Parameter("low_partner_change_rate")
    high_partner_change_prop = high_rate * high_prop / (high_rate * high_prop + low_rate * (1. - high_prop))
    low_partner_change_prop = 1. - high_partner_change_prop
    
    def build_matrix(high_change_prop, low_change_prop):
        mixing_matrix = jnp.array([[high_change_prop, low_change_prop]])  # The "g" values
        mixing_matrix = jnp.repeat(mixing_matrix, 2, axis=0)  # Double up to a square array
        return mixing_matrix
    
    activity_strat.set_flow_adjustments(
        "infection",
        {
            "High": Multiply(Parameter("high_partner_change_rate")),  # Or multiply top row of matrix by this
            "Low": Multiply(Parameter("low_partner_change_rate")),  # Or multiply bottom row of matrix by this
        },
    )
    mixing_matrix = Function(build_matrix, (high_partner_change_prop, low_partner_change_prop))
    activity_strat.set_mixing_matrix(mixing_matrix)
    model.stratify_with(activity_strat)
    model.request_output_for_compartments(
        "infectious",
        ["Infectious"],
        save_results=False,
    )
    model.request_output_for_compartments(
        "total",
        compartments,
        save_results=False,
    )
    model.request_function_output(
        "Prevalence",
        DerivedOutput("infectious") / DerivedOutput("total") * 100.,
    )
    model.request_output_for_flow(
        "HIV infections",
        "infection",
    )
    model.request_function_output(
        "Incidence",
        DerivedOutput("HIV infections") / DerivedOutput("total") * 100.,
    )
    model.request_output_for_flow(
        "mortality",
        "aids_mortality",
    )
    model.request_cumulative_output(
        "Cumulative mortality",
        "mortality",
    )
    model.request_output_for_flow(
        "non_aids_mortality",
        "non_aids_mortality",
    )
    model.request_output_for_compartments(
        "hiv_number",
        ("Infectious", "AIDS",),
    )
    model.request_function_output(
        "Deaths",
        func=DerivedOutput("mortality") + (1. / Parameter("expectancy_at_debut")) * DerivedOutput("hiv_number"),
    )
    for stratum in activity_strata:
        model.request_output_for_compartments(
            f"{stratum}_number",
            compartments,
            strata={"activity": stratum},
        )
        model.request_function_output(
            f"{stratum}_prev",
            func=DerivedOutput(f"{stratum}_number") / DerivedOutput("total"),
        )
    model.request_function_output(
        "mean_partner_change",
        func=DerivedOutput("High_prev") * Parameter("high_partner_change_rate") + DerivedOutput("Low_prev") * Parameter("low_partner_change_rate"),
    )

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "8_20_outputs.csv", index_col=0)
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()

    differences = expected_results - model_results
    assert differences.abs().max().max() < TOLERANCE
