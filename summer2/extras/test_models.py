import numpy as np

from summer2 import CompartmentalModel
from summer2 import Stratification, StrainStratification
from summer2.parameters import Parameter as param
from summer2.parameters import DerivedOutput


def sir(times=[0, 100]):
    m = CompartmentalModel(times, ["S", "I", "R"], "I")
    m.set_initial_population({"S": 990.0, "I": 10.0})
    m.add_infection_frequency_flow("infection", param("contact_rate"), "S", "I")
    m.add_transition_flow("recovery", param("recovery_rate"), "I", "R")
    incidence = m.request_output_for_flow("incidence", "infection")
    m.request_function_output("notifications", incidence * param("cdr"))
    m.set_default_parameters({"contact_rate": 0.4, "recovery_rate": 0.1, "cdr": 0.2})
    return m


def sirs_parametric_age(times=[0, 100], agegroups=list(range(0, 80, 5))):
    m = CompartmentalModel(times, ["S", "I", "R"], "I")
    m.set_initial_population({"S": 99990.0, "I": 10.0})
    m.add_infection_frequency_flow("infection", param("contact_rate"), "S", "I")
    m.add_transition_flow("recovery", 1.0 / param("recovery_duration"), "I", "R")
    m.add_transition_flow("waning", 1.0 / param("waning_duration"), "R", "S")

    max_strl = len(str(agegroups[-1]))
    agegroup_keys = [str(k).zfill(max_strl) for k in agegroups]
    num_age = len(agegroup_keys)
    age_strat = Stratification("age", agegroup_keys, ["S", "I", "R"])

    # +++ SUMMER3
    # Not currently parameterizable
    rec_adj = {k: adj for k, adj in zip(agegroup_keys, np.linspace(1.5, 0.5, num_age))}
    age_strat.set_flow_adjustments("recovery", rec_adj)
    wane_adj = {k: adj for k, adj in zip(agegroup_keys, np.linspace(0.5, 1.5, num_age))}
    age_strat.set_flow_adjustments("waning", wane_adj)

    mm_base = np.linspace(1.5, 1.0, num_age).reshape((1, num_age))
    mm = (mm_base * mm_base.T) * 0.1

    age_strat.set_mixing_matrix(mm)

    pop_spread = np.linspace(2.0, 1.0, num_age)
    pop_split = pop_spread / pop_spread.sum()

    age_strat.set_population_split({k: pop_prop for k, pop_prop in zip(agegroup_keys, pop_split)})

    m.stratify_with(age_strat)

    # +++ SUMMER3
    # Arbitrary dims for output requests
    # Export to xarray instead of dataframe
    incidence = m.request_output_for_flow("incidence", "infection")
    m.request_function_output("notifications", incidence * param("cdr"))

    for k in agegroup_keys:
        m.request_output_for_flow(f"incidenceXage_{k}", "infection", source_strata={"age": k})

    m.set_default_parameters(
        {"contact_rate": 0.4, "recovery_duration": 10.0, "waning_duration": 100.0, "cdr": 0.2}
    )
    return m
