"""Implementation of CompartmentalModel and ModelBackend internals in Jax

This is a mess right now!
"""

from dataclasses import dataclass

import numpy as np

from jax import lax, numpy as jnp

from summer2.runner.jax import ode

# from jax.experimental import ode
from summer2.runner.jax import solvers

from summer2.adjust import Overwrite

from summer2.runner import ModelBackend

from summer2.solver import SolverType, SolverArgs

from .stratify import get_calculate_initial_pop
from .derived_outputs import build_derived_outputs_runner


def clean_compartments(compartment_values: jnp.ndarray):
    return jnp.where(compartment_values < 0.0, 0.0, compartment_values)


def get_force_of_infection(
    strain_infectious_values: jnp.array,
    strain_compartment_infectiousness: jnp.array,
    strain_category_indexer: jnp.array,
    mixing_matrix: jnp.array,
    category_populations: jnp.array,
) -> dict:
    """
    Calculate the force of infection for both frequency-dependent and density-dependent transmission assumptions.
    Considers only one strain at this stage, with the strain logic sitting outside of this function.

    Args:
        strain_infectious_values: Vector of compartment size values for the infectious compartments (relevant to the strain considered)
        strain_compartment_infectiousness: Vector of infectiousness scaling values with same length as strain_infectious_values
        strain_category_indexer: Indexer with each row containing the indices of compartments relevant to a mixing category
        mixing_matrix: The square mixing matrix with dimensions equal to the number of mixing categories
            The columns of the matrix represent the infecting categories and the rows the infected categories
        category_populations: Vector of the population sizes with length equal to the number of mixing categories
    """

    infected_values = strain_infectious_values * strain_compartment_infectiousness
    infectious_populations = jnp.sum(infected_values[strain_category_indexer], axis=-1)
    infection_density = mixing_matrix @ infectious_populations
    category_prevalence = infectious_populations / category_populations
    infection_frequency = mixing_matrix @ category_prevalence

    return {"infection_density": infection_density, "infection_frequency": infection_frequency}


def build_get_infectious_multipliers(runner, debug=False):
    population_cat_indexer = jnp.array(runner._population_category_indexer)

    # FIXME: We are hardcoding this for frequency only right now
    infect_proc_type = runner._infection_process_type

    # We use this elsewhere to enable testing simple models without an infection process
    if infect_proc_type is None:
        return None

    if infect_proc_type == "both":
        raise NotImplementedError("No support for mixed infection frequency/density")

    # FIXME: This could desparately use a tidy-up - all the indexing is a nightmare

    def get_infectious_multipliers(
        time, compartment_values, cur_graph_outputs, compartment_infectiousness
    ):
        infection_frequency = {}
        infection_density = {}

        full_multipliers = jnp.ones(len(runner.infectious_flow_indices))

        mixing_matrix = cur_graph_outputs["mixing_matrix"]
        category_populations = compartment_values[population_cat_indexer].sum(axis=1)

        per_strain_out = {}

        for strain_idx, strain in enumerate(runner.model._disease_strains):
            strain_compartment_infectiousness = compartment_infectiousness[strain]
            strain_infectious_idx = runner._strain_infectious_indexers[strain]
            strain_category_indexer = runner._strain_category_indexers[strain]

            strain_infectious_values = compartment_values[strain_infectious_idx]
            strain_values = get_force_of_infection(
                strain_infectious_values,
                strain_compartment_infectiousness,
                strain_category_indexer,
                mixing_matrix,
                category_populations,
            )
            infection_frequency[strain] = strain_values["infection_frequency"]
            infection_density[strain] = strain_values["infection_density"]

            if infect_proc_type == "freq":
                strain_ifect = strain_values["infection_frequency"]
            elif infect_proc_type == "dens":
                strain_ifect = strain_values["infection_density"]

            if debug:
                per_strain_out[strain] = strain_ifect

            # FIXME: So we produce strain infection values _per category_
            # (ie not all model compartments, not all infectious compartments)
            # We need to rebroadcast these back out the appropriate compartments somehow
            # ... hence the following bit of weird double-dip indexing

            # _infect_strain_lookup_idx is an array of size (num_infectious_comps), whose values
            # are the strain_idx of the strain to whom they belong

            strain_ifectcomp_mask = runner._infect_strain_lookup_idx == strain_idx

            # _infect_cat_lookup_idx is an array of size (num_infectious_comps), whose values
            # are the mixing category to which they belong (and therefore an index into
            # the values returned in strain_ifect above)

            # Get the strain infection values broadcast to num_infectious_comps
            strain_ifect_bcast = strain_ifect[runner._infect_cat_lookup_idx]

            # full_multipliers is the length of the infectious compartments - ie the same as the
            # above mask
            # full_multipliers = full_multipliers.at[strain_ifectcomp_mask].mul(
            #    strain_ifect_bcast[strain_ifectcomp_mask]
            # )

            full_multipliers = full_multipliers.at[strain_ifectcomp_mask].set(
                full_multipliers[strain_ifectcomp_mask] * strain_ifect_bcast[strain_ifectcomp_mask]
            )

        if debug:
            return full_multipliers, per_strain_out
        else:
            return full_multipliers

    return get_infectious_multipliers


def build_get_flow_weights(runner: ModelBackend):
    m = runner.model

    dag_keys = list(m.graph.dag)
    mvars = [k for k in dag_keys if k.startswith("model_variables.")]

    if len(mvars):
        tv_keys = list(m.graph.filter(sources=mvars).dag)
        tv_flow_map = {
            k: m._flow_key_map[k] for k in set(m._flow_key_map).intersection(set(tv_keys))
        }
    else:
        tv_flow_map = {}

    def get_flow_weights(cur_graph_outputs, static_flow_weights):
        flow_weights = jnp.copy(static_flow_weights)

        for k, v in tv_flow_map.items():
            val = cur_graph_outputs[k]
            flow_weights = flow_weights.at[v].set(val)

        return flow_weights

    return get_flow_weights


def _build_calc_computed_values(runner):
    def calc_computed_values(compartment_vals, time, parameters):
        model_variables = {"compartment_values": compartment_vals, "time": time}

        computed_values = runner.computed_values_runner(
            parameters=parameters, model_variables=model_variables
        )

        return computed_values

    return calc_computed_values


def build_get_flow_rates(runner, ts_graph_func, get_infectious_multipliers=None, debug=False):
    # calc_computed_values = build_calc_computed_values(runner)
    get_flow_weights = build_get_flow_weights(runner)

    infect_proc_type = runner._infection_process_type
    # if infect_proc_type:
    #    get_infectious_multipliers = build_get_infectious_multipliers(runner)

    population_idx = np.array(runner.population_idx)
    infectious_flow_indices = jnp.array(runner.infectious_flow_indices)

    def get_flow_rates(compartment_values: jnp.array, time, static_graph_vals, model_data):
        compartment_values = clean_compartments(compartment_values)

        sources = {
            "model_variables": {"time": time, "compartment_values": compartment_values},
            "static_inputs": static_graph_vals,
        }

        ts_graph_vals = ts_graph_func(**sources)

        # JITTED
        flow_weights = get_flow_weights(ts_graph_vals, model_data["static_flow_weights"])

        populations = compartment_values[population_idx]

        # Update for special cases (population-independent and CrudeBirth)
        if runner._has_non_pop_flows:
            populations = populations.at[runner._non_pop_flow_idx].set(1.0)
        if runner._has_crude_birth:
            populations = populations.at[runner._crude_birth_idx].set(compartment_values.sum())

        flow_rates = flow_weights * populations

        # Calculate infection flows
        if infect_proc_type:
            infect_mul = get_infectious_multipliers(
                time,
                compartment_values,
                ts_graph_vals,
                model_data["compartment_infectiousness"],
            )
            # flow_rates = flow_rates.at[infectious_flow_indices].mul(infect_mul)
            flow_rates = flow_rates.at[infectious_flow_indices].set(
                flow_rates[infectious_flow_indices] * infect_mul
            )
            # ReplacementBirthFlow depends on death flows already being calculated; update here
        if runner._has_replacement:
            # Only calculate timestep_deaths if we use replacement, it's expensive...
            # if len(runner.death_flow_indices):
            _timestep_deaths = flow_rates[runner.death_flow_indices].sum()
            # else:
            #    _timestep_deaths = 0.0
            # flow_rates = flow_rates.at[runner._replacement_flow_idx].set(0.0)  # _timestep_deaths)
            flow_rates = flow_rates.at[runner._replacement_flow_idx].set(
                flow_rates[runner._replacement_flow_idx] * _timestep_deaths
            )

        if debug:
            return flow_rates, ts_graph_vals["computed_values"], ts_graph_vals
        else:
            return flow_rates, ts_graph_vals["computed_values"]

    return get_flow_rates


def build_get_compartment_rates(runner):
    accum_maps = get_accumulation_maps(runner)

    def get_compartment_rates(compartment_values, flow_rates):
        comp_rates = jnp.zeros_like(compartment_values)

        for flow_src, comp_target in accum_maps["positive"]:
            comp_rates = comp_rates.at[comp_target].add(flow_rates[flow_src])
        for flow_src, comp_target in accum_maps["negative"]:
            comp_rates = comp_rates.at[comp_target].add(-flow_rates[flow_src])

        return comp_rates

    return get_compartment_rates


def build_get_rates(runner, ts_graph_func):
    get_infectious_multipliers = build_get_infectious_multipliers(runner)
    get_flow_rates = build_get_flow_rates(runner, ts_graph_func, get_infectious_multipliers)
    get_flow_rates_debug = build_get_flow_rates(
        runner, ts_graph_func, get_infectious_multipliers, True
    )
    get_compartment_rates = build_get_compartment_rates(runner)

    def get_rates(compartment_values, time, static_graph_vals, model_data):
        flow_rates, _ = get_flow_rates(compartment_values, time, static_graph_vals, model_data)
        comp_rates = get_compartment_rates(compartment_values, flow_rates)

        return flow_rates, comp_rates

    def get_rates_debug(compartment_values, time, static_graph_vals, model_data):
        flow_rates, _, cur_ts_vals = get_flow_rates_debug(
            compartment_values, time, static_graph_vals, model_data
        )
        comp_rates = get_compartment_rates(compartment_values, flow_rates)

        return flow_rates, comp_rates, cur_ts_vals

    return {
        "get_flow_rates": get_flow_rates,
        "get_rates": get_rates,
        "get_rates_debug": get_rates_debug,
        "get_infectious_multipliers": get_infectious_multipliers,
    }


def get_accumulation_maps(runner):
    pos_map = [mflow for mflow in runner._pos_flow_map]
    neg_map = [mflow for mflow in runner._neg_flow_map]

    def peel_flow_map(flow_map):
        targets = []
        src_idx = []
        remainder = []
        for src_flow, target in flow_map:
            if target not in targets:
                targets.append(target)
                src_idx.append(src_flow)
            else:
                remainder.append((src_flow, target))
        return np.array(src_idx), np.array(targets), remainder

    def recurse_unpeel(flow_map):
        remainder = flow_map
        full_map = []
        while len(remainder) > 0:
            sources, targets, remainder = peel_flow_map(remainder)
            full_map.append((sources, targets))
        return full_map

    return {"positive": recurse_unpeel(pos_map), "negative": recurse_unpeel(neg_map)}


def build_get_compartment_infectiousness(model):
    """
    Build a Jax function to return the compartment infectiousness (for all compartments),
    of the strain specified by strain
    """

    # This is run during prepare_dynamic
    # i.e. it is done once at the start of a model run, but
    # is parameterized (non-structural)
    def get_compartment_infectiousness(static_graph_values):
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        compartment_infectiousness = jnp.ones(len(model.compartments))

        # Apply infectiousness adjustments
        for strat in model._stratifications:
            for comp_name, adjustments in strat.infectiousness_adjustments.items():
                for stratum, adjustment in adjustments.items():
                    if adjustment:
                        is_overwrite = isinstance(adjustment, Overwrite)
                        adj_value = static_graph_values[adjustment.param._graph_key]
                        adj_comps = model.get_matching_compartments(
                            comp_name, {strat.name: stratum}
                        )
                        for c in adj_comps:
                            if is_overwrite:
                                compartment_infectiousness = compartment_infectiousness.at[
                                    c.idx
                                ].set(adj_value)
                            else:
                                orig_value = compartment_infectiousness[c.idx]
                                compartment_infectiousness = compartment_infectiousness.at[
                                    c.idx
                                ].set(adj_value * orig_value)

        strain_comp_inf = {}

        for strain in model._disease_strains:
            if "strain" in model.stratifications:
                strain_filter = {"strain": strain}
            else:
                strain_filter = {}

            # _Must_ be ordered here
            strain_infect_comps = model.query_compartments(
                strain_filter, tags="infectious", as_idx=True
            )

            strain_comp_inf[strain] = compartment_infectiousness[strain_infect_comps]

        return strain_comp_inf

    return get_compartment_infectiousness


# Returned by the one_step runner function - the complete internal state
# of the model at a given timestep
# Does not include derived outputs - these are a post-process
@dataclass
class StepResults:
    flow_rates: jnp.array
    comp_rates: jnp.array
    comp_vals: jnp.array
    static_graph_vals: dict
    ts_graph_vals: dict
    initial_population: jnp.array
    model_data: dict
    infectious_multipliers: jnp.array
    infect_mul_per_strain: dict


def build_run_model(
    runner,
    base_params=None,
    dyn_params=None,
    solver=None,
    solver_args=None,
    derived_outputs=None,
    include_full_outputs=True,
):
    if dyn_params is None:
        dyn_params = runner.model.get_input_parameters()

    dyn_params = [f"parameters.{p}" if not p.startswith("parameters.") else p for p in dyn_params]

    # dyn_params may contain parameters only used in derived outputs, that do not show up
    # in the main graph.  These need to be filtered out; they will be computed every time,
    # but it's simpler than trying to squash the graphs together
    model_graph_keys = set(runner.model.graph.dag)
    dyn_params = [k for k in dyn_params if k in model_graph_keys]

    # Graph frozen for all non-calibration parameters
    if base_params is None:
        base_params = {}

    source_inputs = {"parameters": base_params}

    ts_vars = runner.model.graph.query("model_variables")

    dyn_params = set(dyn_params).union(set(ts_vars))

    param_frozen_cg, _ = runner.model.graph.freeze(dyn_params, source_inputs)

    # static_cg = param_frozen_cg.filter(exclude=ts_vars)
    # static_graph_func = static_cg.get_callable()(parameters=base_params)

    timestep_cg, static_cg = param_frozen_cg.freeze(ts_vars)

    timestep_graph_func = timestep_cg.get_callable()
    # timestep_graph_func = timestep_cg.get_callable()
    static_graph_func = static_cg.get_callable()

    rates_funcs = build_get_rates(runner, timestep_graph_func)
    get_rates = rates_funcs["get_rates"]
    get_flow_rates = rates_funcs["get_flow_rates"]
    get_rates_debug = rates_funcs["get_rates_debug"]
    get_infectious_multipliers = rates_funcs["get_infectious_multipliers"]

    # from jax import vmap

    # get_flows_for_outputs = vmap(get_flow_rates, in_axes=(0, 0, None, None), out_axes=(0))

    def get_comp_rates(comp_vals, t, static_graph_vals, model_data):
        return get_rates(comp_vals, t, static_graph_vals, model_data)[1]

    if solver is None or solver == SolverType.SOLVE_IVP:
        solver = SolverType.ODE_INT

    if solver == SolverType.ODE_INT:
        if solver_args is None:
            # Some sensible defaults; faster than
            # the odeint defaults,
            # but accurate enough for our tests
            solver_args = SolverArgs.DEFAULT

        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return ode.odeint(
                get_comp_rates,
                initial_population,
                times,
                static_graph_vals,
                model_data,
                **solver_args,
            )

    elif solver == SolverType.RUNGE_KUTTA:

        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return solvers.rk4(
                get_comp_rates, initial_population, times, static_graph_vals, model_data
            )

    elif solver == SolverType.EULER:

        def get_ode_solution(initial_population, times, static_graph_vals, model_data):
            return solvers.euler(
                get_comp_rates, initial_population, times, static_graph_vals, model_data
            )

    else:
        raise NotImplementedError("Incompatible SolverType for Jax runner", solver)

    times = jnp.array(runner.model.times)

    calc_initial_pop = get_calculate_initial_pop(runner.model)
    get_compartment_infectiousness = build_get_compartment_infectiousness(runner.model)

    do_cg, calc_derived_outputs = build_derived_outputs_runner(
        runner.model, whitelist=derived_outputs
    )

    m = runner.model
    dag_keys = list(m.graph.dag)

    mvars = [k for k in dag_keys if k.startswith("model_variables.")]

    if len(mvars):
        tv_keys = list(m.graph.filter(sources=mvars).dag)
    else:
        tv_keys = []

    static_flow_map = {k: m._flow_key_map[k] for k in set(m._flow_key_map).difference(set(tv_keys))}

    model_times = jnp.array(m.times)

    # In the case of parameters used by derived outputs that are _not_ supplised in dyn_params,
    # we need to capture these from the original base_params
    do_params = set(
        [v.key for v in m._do_tracker_graph.get_input_variables() if v.source == "parameters"]
    )
    do_base_params = {k: v for k, v in base_params.items() if k in do_params}

    def get_flows_for_outputs(outputs, static_graph_vals, model_data):
        # Empty array to kick-start flow rates from outputs
        flow_rates_full = jnp.empty((len(m.times), len(m.flows)))

        # And likewise for computed values (but as a dict)
        cv_out = {}
        for k in m._computed_values_graph_dict:
            cv_out[k] = jnp.empty((len(m.times),))

        def f(carry, i):
            frate_full, cur_cv = carry
            t = model_times[i]
            frates, cvals = get_flow_rates(outputs[i], t, static_graph_vals, model_data)
            frate_full = frate_full.at[i].set(frates)
            for k, v in cvals.items():
                cur_cv[k] = cur_cv[k].at[i].set(v)
            return (frate_full, cur_cv), None

        (out_flows, out_cv), _ = lax.scan(
            f, (flow_rates_full, cv_out), jnp.array(range(len(m.times)))
        )

        return out_flows, out_cv

    def run_model(parameters):
        static_graph_vals = static_graph_func(parameters=parameters)
        initial_population = calc_initial_pop(static_graph_vals)

        static_flow_weights = jnp.zeros(len(runner.model.flows))
        for k, v in static_flow_map.items():
            val = static_graph_vals[k]
            static_flow_weights = static_flow_weights.at[v].set(val)

        compartment_infectiousness = get_compartment_infectiousness(static_graph_vals)
        model_data = {
            "compartment_infectiousness": compartment_infectiousness,
            "static_flow_weights": static_flow_weights,
        }

        outputs = get_ode_solution(initial_population, times, static_graph_vals, model_data)

        # # Empty array to kick-start flow rates from outputs
        # flow_rates_full = jnp.empty((len(m.times), len(m.flows)))

        # # And likewise for computed values (but as a dict)
        # cv_out = {}
        # for k in m._computed_values_graph_dict:
        #     cv_out[k] = jnp.empty((len(m.times),))

        # def f(carry, i):
        #     frate_full, cur_cv = carry
        #     t = model_times[i]
        #     frates, cvals = get_flow_rates(outputs[i], t, static_graph_vals, model_data)
        #     frate_full = frate_full.at[i].set(frates)
        #     for k, v in cvals.items():
        #         cur_cv[k] = cur_cv[k].at[i].set(v)
        #     return (frate_full, cur_cv), None

        # (out_flows, out_cv), _ = lax.scan(
        #     f, (flow_rates_full, cv_out), jnp.array(range(len(m.times)))
        # )

        out_flows, out_cv = get_flows_for_outputs(outputs, static_graph_vals, model_data)

        model_variables = {"outputs": outputs, "flows": out_flows, "computed_values": out_cv}

        do_full_params = do_base_params.copy()
        do_full_params.update(parameters)

        derived_outputs = calc_derived_outputs(
            parameters=do_full_params, model_variables=model_variables
        )

        # return {"outputs": outputs, "model_data": model_data}
        output_dict = {
            "derived_outputs": derived_outputs,
        }  # "model_data": model_data}
        if include_full_outputs:
            output_dict["outputs"] = outputs

        return output_dict

    ons_get_inf_mul = build_get_infectious_multipliers(runner, True)

    def one_step(parameters: dict = None, t: float = None, comp_vals=None):
        static_graph_vals = static_graph_func(parameters=parameters)

        if t is None:
            t = runner.model.times[0]

        if comp_vals is None:
            comp_vals = calc_initial_pop(static_graph_vals)

        initial_population = comp_vals

        static_flow_weights = jnp.zeros(len(runner.model.flows))
        for k, v in static_flow_map.items():
            val = static_graph_vals[k]
            static_flow_weights = static_flow_weights.at[v].set(val)

        compartment_infectiousness = get_compartment_infectiousness(static_graph_vals)
        model_data = {
            "compartment_infectiousness": compartment_infectiousness,
            "static_flow_weights": static_flow_weights,
        }

        flow_rates, comp_rates, ts_graph_vals = get_rates_debug(
            comp_vals, t, static_graph_vals, model_data
        )
        if get_infectious_multipliers:
            infect_mul, infect_mul_per_strain = ons_get_inf_mul(
                t,
                comp_vals,
                ts_graph_vals,
                compartment_infectiousness,
            )
        else:
            infect_mul, infect_mul_per_strain = None, None

        # return {"outputs": outputs, "model_data": model_data}
        res = {
            "flow_rates": flow_rates,
            "comp_rates": comp_rates,
            "comp_vals": comp_vals + comp_rates,
            "static_graph_vals": static_graph_vals,
            "ts_graph_vals": ts_graph_vals,
            "initial_population": initial_population,
            "model_data": model_data,
            "infectious_multipliers": infect_mul,
            "infect_mul_per_strain": infect_mul_per_strain,
        }
        return StepResults(**res)

    runner_dict = {
        "get_rates": get_rates,
        "get_flow_rates": get_flow_rates,
        "get_comp_rates": get_comp_rates,
        "calc_initial_pop": calc_initial_pop,
        "get_compartment_infectiousness": get_compartment_infectiousness,
        "get_ode_solution": get_ode_solution,
        "calc_derived_outputs": calc_derived_outputs,  # Callable for derived outputs
        "timestep_cg": timestep_cg,  # Computegraph for time-varying processes
        "timestep_graph_func": timestep_graph_func,  # Callable for timestep graph
        "static_cg": static_cg,  # Computegraph for static portion of model
        "static_graph_func": static_graph_func,  # Callable for static graph
        "one_step": one_step,  # Single (euler) step run function
        "derived_outputs_cg": do_cg,  # DerivedOutputs computegraph
        "get_rates_debug": rates_funcs["get_rates_debug"],
    }

    return run_model, runner_dict
