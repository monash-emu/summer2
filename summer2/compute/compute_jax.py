"""
Optimized 'hot' functions used by CompartmentalModel and its runners.
"""

from jax import jit, numpy as np
from functools import partial

# Use Numba to speed up the calculation of the population.
@jit
def find_sum(compartment_values: np.ndarray) -> float:
    return compartment_values.sum()


@jit
def accumulate_positive_flow_contributions(
    flow_rates: np.ndarray,
    comp_rates: np.ndarray,
    pos_flow_map: np.ndarray,
):
    """
    Fast accumulator for summing positive flow rates into their effects on compartments

    Args:
        flow_rates (np.ndarray): Flow rates to be accumulated
        comp_rates (np.ndarray): Output array of compartment rates
        pos_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
    """
    comp_rates = comp_rates.copy()
    for src, target in pos_flow_map:
        comp_rates = comp_rates.at[target].add(flow_rates[src])
    return comp_rates


@jit
def accumulate_negative_flow_contributions(
    flow_rates: np.ndarray,
    comp_rates: np.ndarray,
    neg_flow_map: np.ndarray,
):
    """Fast accumulator for summing negative flow rates into their effects on compartments

    Args:
        flow_rates (np.ndarray): Flow rates to be accumulated
        comp_rates (np.ndarray): Output array of compartment rates
        neg_flow_map (np.ndarray): Array of src (flow), target (compartment) indices
    """
    comp_rates = comp_rates.copy()
    for src, target in neg_flow_map:
        comp_rates = comp_rates.at[target].add(-flow_rates[src])
    return comp_rates


@jit
def get_strain_infection_values(
    strain_infectious_values,
    strain_compartment_infectiousness,
    strain_category_indexer,
    mixing_matrix,
    category_populations,
):
    infected_values = strain_infectious_values * strain_compartment_infectiousness
    infectious_populations = np.sum(infected_values[strain_category_indexer], axis=1)
    infection_density = mixing_matrix @ infectious_populations
    category_prevalence = infectious_populations / category_populations
    infection_frequency = mixing_matrix @ category_prevalence

    return infection_density, infection_frequency
