import pandas as pd

from summer2.model import CompartmentalModel
from summer2.parameters import Parameter

TOLERANCE = 1e-5


def test_chapter_2_model():
    model_2_1 = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_2_1.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infectious_seed": 1.,
        "end_time": 1000.,
    }
    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    model = CompartmentalModel(
        times=(0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=("Infectious",),
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["tot_popn"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("R0") / ave_infous,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous, 
        source="Infectious", 
        dest="Immune"
    )
    parameters = {
        "ave_preinfous": 2.,
        "ave_infous": 2.,
        "R0": 2.,
    }
    model.run(parameters=parameters, solver="euler")
    seir_compartments = model.get_outputs_df()
    assert abs(model_2_1 - seir_compartments).max().max() < TOLERANCE


def test_chapter_2_1a_model():
    seir_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_2_1a_seir.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infectious_seed": 1.,
        "end_time": 1000.,
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
            "Susceptible": config["tot_popn"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("R0") / ave_infous,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune"
    )
    parameters = {
        "ave_preinfous": 2.,
        "ave_infous": 2.,
        "R0": 2.,
    }
    model.run(parameters=parameters, solver="euler")
    seir_values = model.get_outputs_df()
    assert abs(seir_data - seir_values).max().max() < TOLERANCE


def test_chapter_3_1_model():
    measles_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_3_1_measles.csv",
        index_col=0,
    )
    flu_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_3_1_flu.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infectious_seed": 1.,
        "end_time": 10000.,
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
            "Susceptible": config["tot_popn"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / ave_infous,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune"
    )
    parameters = {
        "r0": 13.,
        "ave_preinfous": 8.,
        "ave_infous": 7.,
    }
    model.run(parameters=parameters, solver="euler")
    measles_values = model.get_outputs_df()
    assert abs(measles_data - measles_values).max().max() < 1e-5
    parameters = {
        "r0": 2.,
        "ave_infous": 2.,
        "ave_preinfous": 2.,
    }
    model.run(parameters=parameters, solver="euler")
    flu_values = model.get_outputs_df()
    assert abs(flu_data - flu_values).max().max() < TOLERANCE


def test_chapter_3_2_model():
    demog_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_3_2.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infectious_seed": 1.,
        "end_time": 100.,
    }
    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0, config["end_time"] * 365.),
        compartments=compartments,
        infectious_compartments=["Infectious"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["tot_popn"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"],
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / ave_infous,
        source="Susceptible", 
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune"
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.
    )
    model.add_replacement_birth_flow(
        "births",
        "Susceptible"
    )
    parameters = {
        "r0": 13.,
        "ave_preinfous": 8.,
        "ave_infous": 7.,
        "life_expectancy": 70.,
    }
    model.run(parameters=parameters, solver="euler")
    demog_values = model.get_outputs_df()
    assert abs(demog_data - demog_values).max().max() < TOLERANCE


def test_chapter_4_1a_model():
    seir_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_4_1a.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infectious_seed": 1.,
        "end_time": 2000.,
        "t_step": 1.,
    }
    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
        timestep=config["t_step"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["tot_popn"] - config["infectious_seed"],
            "Infectious": config["infectious_seed"]
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / ave_infous,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune"
    )
    parameters = {
        "r0": 2.,
        "ave_preinfous": 2.,
        "ave_infous": 2.,
        "life_expectancy": 70.,
    }
    model.run(parameters=parameters, solver="euler")
    seir_values = model.get_outputs_df()
    assert abs(seir_data - seir_values).max().max() < TOLERANCE


def test_chapter_4_2_model():
    seir_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_4_2.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 5234.,
        "infous_0": 2.,
        "end_time": 200.,
        "t_step": 0.5,
        "prop_immune_0": 0.3,
    }
    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
        timestep=config["t_step"],
    )
    n_immune_0 = config["prop_immune_0"] * config["tot_popn"]
    model.set_initial_population(
        distribution={
            "Susceptible": config["tot_popn"] - config["infous_0"] - n_immune_0, 
            "Infectious": config["infous_0"],
            "Immune": n_immune_0,
        }
    )
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=Parameter("r0") / ave_infous,
        source="Susceptible",
        dest="Pre-infectious"
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious"
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune"
    )
    parameters = {
        "r0": 2.1,
        "ave_preinfous": 2.,
        "ave_infous": 2.,
        "life_expectancy": 70.,
    }
    model.run(parameters=parameters, solver="euler")
    seir_values = model.get_outputs_df()
    assert abs(seir_data - seir_values).max().max() < TOLERANCE


def test_chapter_4_3a_model():
    seir_data = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_4_3a.csv",
        index_col=0,
    )
    config = {
        "tot_popn": 1e5,
        "infous_0": 1.,
        "end_time": 18250.,
        "t_step": 1.,
        "prop_immune_0": 0.3,
    }
    compartments = ("Susceptible", "Pre-infectious", "Infectious", "Immune")
    model = CompartmentalModel(
        times=(0, config["end_time"]),
        compartments=compartments,
        infectious_compartments=["Infectious"],
        timestep=config["t_step"],
    )
    model.set_initial_population(
        distribution={
            "Susceptible": config["tot_popn"] - config["infous_0"],
            "Infectious": config["infous_0"],
        }
    )
    r0 = Parameter("r0")
    ave_infous = Parameter("ave_infous")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=r0 / ave_infous,
        source="Susceptible", 
        dest="Pre-infectious",
    )
    model.add_transition_flow(
        name="progression", 
        fractional_rate=1. / Parameter("ave_preinfous"),
        source="Pre-infectious", 
        dest="Infectious",
    )
    model.add_transition_flow(
        name="recovery", 
        fractional_rate=1. / ave_infous,
        source="Infectious", 
        dest="Immune",
    )
    model.add_universal_death_flows(
        "universal_death",
        death_rate=1. / Parameter("life_expectancy") / 365.,
    )
    model.add_replacement_birth_flow(
        "births",
        "Susceptible",
    )
    parameters = {
        "r0": 13.,
        "ave_preinfous": 8.,
        "ave_infous": 7.,
        "life_expectancy": 70.,
    }
    model.run(parameters=parameters, solver="euler")
    seir_values = model.get_outputs_df()
    assert abs(seir_data - seir_values).max().max() < TOLERANCE
    