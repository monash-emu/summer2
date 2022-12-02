from typing import Tuple

import numpy as np
from jax import numpy as jnp

from summer2 import Compartment
from summer2.adjust import Overwrite
import summer2.flows as flows

from summer2.population import get_rebalanced_population
from summer2.utils import clean_compartment_values


class ModelBackend:
    """
    An optimized, but less accessible model runner.
    """

    def __init__(self, model):
        # Compartmental model
        self.model = model

        # Tracks total deaths per timestep for death-replacement birth flows
        self._timestep_deaths = None

        # Set our initial parameters to an empty dict - this is really just to appease tests
        self.parameters = {}

    def prepare_structural(self):
        # FIXME: Redundant
        self._iter_non_function_flows = [(i, f) for i, f in enumerate(self.model.flows)]

        self._build_compartment_category_map()

        self.infectious_flow_indices = np.array(
            [i for i, f in self._iter_non_function_flows if isinstance(f, flows.BaseInfectionFlow)],
            dtype=int,
        )
        self.death_flow_indices = np.array(
            [i for i, f in self._iter_non_function_flows if f.is_death_flow], dtype=int
        )

        # Include dummy values in population_idx to account for Entry flows
        non_func_pops = np.array(
            [f.source.idx if f.source else 0 for i, f in self._iter_non_function_flows], dtype=int
        )

        self.population_idx = non_func_pops

        # Store indices of flows that are not population dependent
        self._non_pop_flow_idx = np.array(
            [
                i
                for i, f in self._iter_non_function_flows
                if (type(f) in (flows.ReplacementBirthFlow, flows.ImportFlow, flows.AbsoluteFlow))
            ],
            dtype=int,
        )
        self._has_non_pop_flows = bool(len(self._non_pop_flow_idx))

        # Crude birth flows use population sum rather than a compartment; store indices here
        self._crude_birth_idx = np.array(
            [i for i, f in self._iter_non_function_flows if type(f) == flows.CrudeBirthFlow],
            dtype=int,
        )
        self._has_crude_birth = bool(len(self._crude_birth_idx))

        # Special indexing required for replacement flows
        self._replacement_flow_idx = np.array(
            [i for i, f in self._iter_non_function_flows if type(f) == flows.ReplacementBirthFlow],
            dtype=int,
        )
        self._has_replacement = bool(len(self._replacement_flow_idx))

        self._precompute_flow_maps()
        self._build_infectious_multipliers_lookup()

    def _build_compartment_category_map(self):
        # Create a matrix that tracks which categories each compartment is in.
        self.num_categories = ncats = len(self.model._mixing_categories)

        # Array that maps compartments to their mixing category
        self._category_lookup = np.empty(len(self.model.compartments), dtype=int)

        all_cat_idx = []
        for i, category in enumerate(self.model._mixing_categories):
            cat_idx = []
            for j, comp in enumerate(self.model.compartments):
                if all(comp.has_stratum(k, v) for k, v in category.items()):
                    cat_idx.append(j)
                    # self._category_matrix[i][j] = 1
                    self._category_lookup[j] = i
            all_cat_idx.append(np.array(cat_idx, dtype=int))

        # Indexer for whole population (compartment_values), that will return a
        # num_cats * (num_comps_per_cat) array that can be summed for category populations
        self._population_category_indexer = pop_cat_idx = np.stack(all_cat_idx)

        # Array (per-strain) of compartment indices for that strain's infectious compartments
        self._strain_infectious_indexers = {}
        # Array (per-strain) of len (strain infectious comps) that maps each
        # strain infectious compartment to its mixing category
        self._strain_category_indexers = {}
        #
        for strain in self.model._disease_strains:
            strain_filter = {"strain": strain} if "strain" in self.model.stratifications else {}
            strain_infectious_comps = self.model.query_compartments(
                strain_filter, tags="infectious", as_idx=True
            )
            self._strain_infectious_indexers[strain] = jnp.array(strain_infectious_comps)

            # Function that tests each element of an ndarray to see if is contained within the
            # current strain infectious compartments
            vcat = np.vectorize(lambda c: c in strain_infectious_comps)
            strain_cat_idx = pop_cat_idx[vcat(pop_cat_idx)]
            # Localize the compartment indices to be within strain infectious comps,
            # rather than the global compartment lookup
            # We do this by creating an inverse global map (but only fill in the compartment
            # indices we care about)
            # It's just a bit easier to read that messing around with dictionaries etc
            tlookup = np.empty(len(self.model.compartments), dtype=int)
            tlookup[strain_infectious_comps] = range(len(strain_infectious_comps))
            strain_cat_idx = tlookup[strain_cat_idx]

            # Ensure this is a 2d array
            # This necessary where there is only one compartment for each category
            strain_cat_idx = strain_cat_idx.reshape((ncats, int(strain_cat_idx.size / ncats)))
            self._strain_category_indexers[strain] = jnp.array(strain_cat_idx)

    def _precompute_flow_maps(self):
        """Build fast-access arrays of flow indices"""
        f_pos_map = []
        f_neg_map = []
        for i, f in self._iter_non_function_flows:
            if f.source:
                f_neg_map.append((i, f.source.idx))
            if f.dest:
                f_pos_map.append((i, f.dest.idx))

        self._pos_flow_map = np.array(f_pos_map, dtype=int)
        self._neg_flow_map = np.array(f_neg_map, dtype=int)

    def _build_infectious_multipliers_lookup(self):
        """Get multipliers for all infectious flows

        These are used by _get_infectious_multipliers_flat (currently experimental)

        Returns:
            np.ndarray: Array of infectiousness multipliers
        """
        lookups = []

        has_freq = False
        has_dens = False

        for i, idx in enumerate(self.infectious_flow_indices):
            f = self.model.flows[idx]
            if isinstance(f, flows.InfectionFrequencyFlow):
                has_freq = True
            elif isinstance(f, flows.InfectionDensityFlow):
                has_dens = True
            cat_idx, strain = self._get_infection_multiplier_indices(f.source, f.dest)
            strain_idx = self.model._disease_strains.index(strain)
            lookups.append([strain_idx, cat_idx])
        full_table = np.array(lookups, dtype=int)
        self._full_table = full_table.reshape(len(self.infectious_flow_indices), 2)
        self._infect_strain_lookup_idx = self._full_table[:, 0].flatten()
        self._infect_cat_lookup_idx = self._full_table[:, 1].flatten()

        self._infection_frequency_only = False
        self._infection_density_only = False

        self._infection_process_type = None

        if has_freq:
            if has_dens:
                self._infection_process_type = "both"
            else:
                self._infection_process_type = "freq"
                self._infection_frequency_only = True
        elif has_dens:
            self._infection_density_only = True
            self._infection_process_type = "dens"

    def _get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self._category_lookup[source.idx]

    def _get_infection_multiplier_indices(
        self, source: Compartment, dest: Compartment
    ) -> Tuple[str, int]:
        """Return indices for infection frequency lookups"""
        idx = self._get_force_idx(source)
        strain = dest.strata.get("strain", self.model._DEFAULT_DISEASE_STRAIN)
        return idx, strain
