import pandas as pd
from pathlib import Path

from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


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
