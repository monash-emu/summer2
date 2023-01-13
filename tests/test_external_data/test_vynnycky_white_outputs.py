from jax import numpy as jnp
import pandas as pd
from pathlib import Path
import numpy as np

from summer2 import CompartmentalModel, Stratification
from summer2.parameters import Parameter, DerivedOutput, Time, Function


TEST_OUTPUTS_PATH = Path(__file__).with_name("vynnycky_white_outputs")
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

    # Run for measles
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
        "infectious_seed": 1.,  # Not specified in text
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

    model.run(parameters=parameters, solver="euler")

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_04_measles_outputs.csv", index_col=0)
    model_results = pd.concat(
        (
            model.get_outputs_df()["Susceptible"] / config["total_population"], 
            model.get_outputs_df()["Immune"] / config["total_population"], 
            model.get_derived_outputs_df()["incidence"],
        ), 
        axis=1
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
        axis=1
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
        model_results[immune_prop] = model.get_derived_outputs_df()["incidence"]

    expected_results.columns = [float(col) for col in expected_results.columns]
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
    expected_results.columns = immune_props
    model_results = pd.DataFrame()
    for immune_prop in immune_props:
        parameters.update({"prop_recovered": immune_prop})
        model.run(parameters=parameters, solver="euler")
        model_results[immune_prop] = model.get_derived_outputs_df()["incidence"]

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

    # Run for measles
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

    # Run for flu
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

    r0s = np.linspace(0.99, 15., 100)
    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_12_outputs.csv", index_col=0)["0"]
    expected_results.index = r0s
    model_results = pd.Series(index=r0s, dtype=float)
    for r0 in r0s:
        parameters.update({"r0": r0})
        model.run(parameters=parameters, solver="euler")
        model_results[r0] = 1. - model.get_outputs_df().loc[config["end_time"], "susceptible"]

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

    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()["incidence"]

    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_17_outputs.csv", index_col=0)["incidence"]
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
    
    model.run(parameters=parameters, solver="euler")
    model_results = model.get_derived_outputs_df()
    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_19_outputs.csv", index_col=0)

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

    birth_rates = (0.015, 0.025, 0.04)
    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_26_outputs.csv", index_col=0)
    expected_results.columns = birth_rates
    model_results = pd.DataFrame(columns=birth_rates)
    for rate in birth_rates:
        parameters.update({"crude_birth_rate": rate})
        model.run(parameters=parameters, solver="euler")
        model_results[rate] = model.get_derived_outputs_df()["incidence_rate"]

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

    coverage_values = (0.5, 0.8, 0.9)
    expected_results = pd.read_csv(TEST_OUTPUTS_PATH / "4_29_outputs.csv", index_col=0)
    expected_results.columns = coverage_values

    model_results = pd.DataFrame(columns=coverage_values)
    for coverage in coverage_values:
        parameters.update({"vacc_coverage": coverage})
        model.run(parameters=parameters, solver="euler")
        model_results[coverage] = model.get_derived_outputs_df()["incidence_rate"]

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
    model_results.index = model_results.index / 365.
    expected_results.index = [round(i, 5) for i in expected_results.index]
    model_results.index = [round(i, 5) for i in model_results.index]

    differences = expected_results - model_results
    assert differences["incidence_rate"].abs().max() < TOLERANCE
