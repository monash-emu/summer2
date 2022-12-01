import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from summer2 import CompartmentalModel
from summer2.parameters import Parameter


def test_full_comparison():
    config = {
        "population": 1000.0,
        "seed": 10.0,
        "start_time": 0.0,
        "end_time": 20.0,
        "time_step": 0.1,
    }

    parameters = {
        "contact_rate": 1.0,
        "recovery_rate": 0.333,
        "death_rate": 0.05,
    }

    # Get the evaluation times based on the requested parameters
    time_period = config["end_time"] - config["start_time"]
    num_steps = int(time_period / config["time_step"]) + 1
    times = np.linspace(config["start_time"], config["end_time"], num=num_steps)

    # Prepare for outputs and populate the initial conditions
    explicit_calcs = np.zeros((num_steps, 3))
    explicit_calcs[0] = [
        config["population"] - config["seed"],
        config["seed"],
        0.0,
    ]

    # Run the calculations at each modelled time step, except the first one
    for t_idx, time in enumerate(times[1:], 1):

        # Get some quantities that we'll need later
        flow_rates = np.zeros(3)
        compartment_sizes = explicit_calcs[t_idx - 1]
        n_suscept = compartment_sizes[0]
        n_infect = compartment_sizes[1]
        n_pop = compartment_sizes.sum()

        # Apply the infection process under the assumption of frequency-dependent transmission
        force_of_infection = parameters["contact_rate"] * n_infect / n_pop
        infection_flow_rate = force_of_infection * n_suscept
        flow_rates[0] -= infection_flow_rate
        flow_rates[1] += infection_flow_rate

        # Recovery of the infectious compartment
        recovery_flow_rate = parameters["recovery_rate"] * n_infect
        flow_rates[1] -= recovery_flow_rate
        flow_rates[2] += recovery_flow_rate

        # Add an infection-specific death flow to the infectious compartment
        death_flow_rate = n_infect * parameters["death_rate"]
        flow_rates[1] -= death_flow_rate

        # Calculate compartment sizes at the next time step given the calculated flow rates
        explicit_calcs[t_idx] = compartment_sizes + flow_rates * config["time_step"]

    compartments = (
        "susceptible",
        "infectious",
        "recovered",
    )
    explicit_outputs_df = pd.DataFrame(explicit_calcs, columns=compartments, index=times)

    def get_sir_model(
        model_config: dict,
    ) -> CompartmentalModel:
        """
        This is the same model as from notebook 01.

        Args:
            model_config: Values needed for model construction other than the parameter values
        Returns:
            The summer model object
        """
        compartments = (
            "susceptible",
            "infectious",
            "recovered",
        )
        infectious_compartment = [
            "infectious",
        ]
        analysis_times = (model_config["start_time"], model_config["end_time"])
        model = CompartmentalModel(
            times=analysis_times,
            compartments=compartments,
            infectious_compartments=infectious_compartment,
            timestep=model_config["time_step"],
        )
        pop = model_config["population"]
        seed = model_config["seed"]
        suscept_pop = pop - seed
        msg = "Seed larger than population"
        assert pop >= 0.0, msg
        model.set_initial_population(distribution={"susceptible": suscept_pop, "infectious": seed})
        model.add_infection_frequency_flow(
            name="infection",
            contact_rate=Parameter("contact_rate"),
            source="susceptible",
            dest="infectious",
        )
        model.add_transition_flow(
            name="recovery",
            fractional_rate=Parameter("recovery_rate"),
            source="infectious",
            dest="recovered",
        )
        model.add_death_flow(
            name="infection_death",
            death_rate=Parameter("death_rate"),
            source="infectious",
        )
        return model

    sir_model = get_sir_model(config)
    sir_model.run(parameters=parameters, solver="euler")
    compartment_values = sir_model.get_outputs_df()

    assert ((compartment_values - explicit_outputs_df).abs().max() < 1e-12).all()
