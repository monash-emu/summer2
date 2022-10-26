import pandas as pd
from summer2.model import CompartmentalModel


def test_chapter_2_model(tolerance=1e-9):
    """
    Run the model from Chapter 2 of Vynnycky and White using summer
    and compare to the results from the worked example in the Excel spreadsheet.
    """

    # Specify parameters
    tot_popn = 100000
    ave_preinfous = 2
    ave_infous = 2
    R0 = 2
    beta = R0 / ave_infous
    infous_rate = 1 / ave_preinfous
    rec_rate = 1 / ave_infous
    infectious_seed = 1

    # Set up the SEIR model
    compartments = (
        "Susceptible", 
        "Pre-infectious", 
        "Infectious", 
        "Immune"
    )
    seir_model = CompartmentalModel(
        times=(0, 200),
        compartments=compartments,
        infectious_compartments=("Infectious",),
    )
    seir_model.set_initial_population(
        distribution={
            "Susceptible": tot_popn - infectious_seed, 
            "Infectious": infectious_seed
        }
    )
    seir_model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=beta,
        source="Susceptible",
        dest="Pre-infectious"
    )
    seir_model.add_transition_flow(
        name="progression", 
        fractional_rate=infous_rate,
        source="Pre-infectious", 
        dest="Infectious"
    )
    seir_model.add_transition_flow(
        name="recovery", 
        fractional_rate=rec_rate, 
        source="Infectious", 
        dest="Immune"
    )
    seir_model.request_output_for_flow(
        name="incidence", 
        flow_name="progression",
        raw_results=True,
    )

    # Run the model and get the compartment sizes and modelled incidence
    seir_model.run(solver="euler")
    seir_compartments = seir_model.get_outputs_df()
    modelled_incidence = seir_model.get_derived_outputs_df()["incidence"].iloc[:200]

    # Get the corresponding results from the worked Excel sheets
    model_2_1 = pd.read_csv(
        "./tests/test_external_data/vynnycky_white_examples/model_2_1.csv",
        index_col=0,
    )

    comp_diffs = model_2_1.iloc[:200, :4] - seir_compartments
    assert comp_diffs.dropna().max().max() < tolerance

    # Compare incidence
    example_incidence = model_2_1.loc[1: 200, "Incidence"].astype(float)
    example_incidence.index = example_incidence.index - 1.
    diff = example_incidence - modelled_incidence
    assert diff.max() < tolerance