import pandas as pd
import numpy as np
from jax import numpy as jnp

from summer2 import CompartmentalModel, Stratification, population
from summer2.parameters import Function, Parameter


def test_init_with_function():
    m = CompartmentalModel([0, 100], ["S", "I", "R"], ["I"])

    age_strat = Stratification("agegroup", ["young", "old"], ["S", "I", "R"])
    m.stratify_with(age_strat)

    state_strat = Stratification("state", ["WA", "other"], ["S", "I", "R"])
    m.stratify_with(state_strat)

    imm_strat = Stratification("imm", ["vacc", "unvacc"], ["S", "I", "R"])
    m.stratify_with(imm_strat)

    state_pop_info = {
        "WA_young": 1000.0,
        "WA_old": 2000.0,
        "other_young": 10000.0,
        "other_old": 30000.0,
    }

    imm_scale = {
        "vacc_young": Parameter("vacc_young"),
        "vacc_old": Parameter("vacc_old"),
        "unvacc_young": 1.0 - Parameter("vacc_young"),
        "unvacc_old": 1.0 - Parameter("vacc_old"),
    }

    def get_init_pop(imm_scale):
        init_pop = jnp.zeros(len(m.compartments), dtype=np.float64)
        for agegroup in m.stratifications["agegroup"].strata:
            for imm in m.stratifications["imm"].strata:
                for state in m.stratifications["state"].strata:
                    q = m.query_compartments(
                        {"name": "S", "agegroup": agegroup, "imm": imm, "state": state}, as_idx=True
                    )
                    state_pinfo_str = f"{state}_{agegroup}"
                    imm_scale_str = f"{imm}_{agegroup}"
                    init_pop = init_pop.at[q].set(
                        state_pop_info[state_pinfo_str] * imm_scale[imm_scale_str]
                    )
        return init_pop

    m.init_population_with_graphobject(Function(get_init_pop, [imm_scale]))

    parameters = {"vacc_young": 0.2, "vacc_old": 0.6}

    init_pop = m.get_initial_population(parameters)

    expected = pd.Series(
        {
            "SXagegroup_youngXstate_WAXimm_vacc": 200.0,
            "SXagegroup_youngXstate_WAXimm_unvacc": 800.0,
            "SXagegroup_youngXstate_otherXimm_vacc": 2000.0,
            "SXagegroup_youngXstate_otherXimm_unvacc": 8000.0,
            "SXagegroup_oldXstate_WAXimm_vacc": 1200.0,
            "SXagegroup_oldXstate_WAXimm_unvacc": 800.0,
            "SXagegroup_oldXstate_otherXimm_vacc": 18000.0,
            "SXagegroup_oldXstate_otherXimm_unvacc": 12000.0,
        }
    )

    assert (m.get_initial_population(parameters)[expected.keys()] == expected).all()
