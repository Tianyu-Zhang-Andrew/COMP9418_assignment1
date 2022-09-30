# Necessary libraries
import numpy as np
import pandas as pd

# combinatorics
from itertools import product

from DiscreteFactors import Factor
from Graph import Graph


class BayesNet():

    def __init__(self, graph, outcomeSpace=None, factor_dict=None):
        self.graph = graph
        self.outcomeSpace = dict()
        self.factors = dict()
        if outcomeSpace is not None:
            self.outcomeSpace = outcomeSpace
        if factor_dict is not None:
            self.factors = factor_dict

    def allEqualThisIndex(self, dict_of_arrays, **fixed_vars):
        # base index is a boolean vector, everywhere true
        first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
        index = np.ones_like(first_array, dtype=np.bool_)
        for var_name, var_val in fixed_vars.items():
            index = index & (np.asarray(dict_of_arrays[var_name]) == var_val)
        return index

    def estimateFactor(self, data, var_name, parent_names, outcomeSpace):
        """
        Calculate a dictionary probability table by ML given
        `data`, a dictionary or dataframe of observations
        `var_name`, the column of the data to be used for the conditioned variable and
        `parent_names`, a tuple of columns to be used for the parents and
        `outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
        Return a dictionary containing an estimated conditional probability table.
        """
        var_outcomes = outcomeSpace[var_name]
        parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
        # cartesian product to generate a table of all possible outcomes
        all_parent_combinations = product(*parent_outcomes)

        f = Factor(list(parent_names) + [var_name], outcomeSpace)

        for i, parent_combination in enumerate(all_parent_combinations):
            parent_vars = dict(zip(parent_names, parent_combination))
            parent_index = self.allEqualThisIndex(data, **parent_vars)
            for var_outcome in var_outcomes:
                var_index = (np.asarray(data[var_name]) == var_outcome)
                f[tuple(list(parent_combination) + [var_outcome])] = (var_index & parent_index).sum() / parent_index.sum()

        return f
