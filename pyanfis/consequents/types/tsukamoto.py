"""Consequents ithat look like a universe of different functions"""
from dataclasses import dataclass, field
from typing import Any

from pyanfis.functions import Universe


@dataclass(slots = True)
class Tsukamoto():
    """
    This class will compute the learnable parameters using the Tsukamoto approach.

    Attributes
    ----------
    universe: Universe
        Universe that will hold all the functions
    config_universe: dict[str, Any]
        Configuration of the universe
    universe_x_values: list[float]
        Range of the universe
    consequent_rules: dict[str, list[str]]
        Rules of the consequents
    precomputed_functions: dict[str, list[float]]
        Precomputed functions of the universe
    """
    universe: Universe = field(init = False)
    config_universe: dict[str, Any] = field(repr = False)
    universe_x_values: list[float] = field(init = False, repr = False)
    consequent_rules: dict[str, list[str]] = field(repr = False)
    precomputed_functions: dict[str, list[float]] = field(init = False, repr = False)

    def __post_init__(
            self
        ) -> None:
        self.universe = Universe(**self.config_universe)
        self.universe_x_values = self.linspace(
            initial_number = self.universe.minimum,
            final_number = self.universe.maximum,
            quantity_of_numbers = 200
        )
        self.precomputed_functions = self.precompute_functions_in_universe(
            rules = self.consequent_rules
        )

        del self.config_universe
        del self.consequent_rules

    def linspace(
            self,
            initial_number: int | float,
            final_number: int | float,
            quantity_of_numbers: int
        ) -> list[float]:
        """
        Create a list 'quantity_of_numbers' numbers from 'initial_number' to 'final_number'

        Parameters
        ----------
        initial_number: int
            Initial number of sequence
        final_number: int
            Final number of sequence
        quantity_of_numbers: int
            Quantity of numbers in sequence

        Returns
        -------
        list[float]:
            List of numbers
        """
        if quantity_of_numbers <= 1:
            return [initial_number]
        
        step = (final_number - initial_number) / (quantity_of_numbers - 1)
        return [
            initial_number + i * step
            for i
            in range(quantity_of_numbers)
        ]

    def precompute_functions_in_universe(
            self,
            rules: dict[str, list[str]]
        ) -> dict[str, list[float]]:
        """
        Precreates all the functions that will be intersected

        Parameters
        ----------
        rules: dict[str, list[str]]
            The functions of the universe

        Returns
        -------
        list[list[float]]:
            Precomputed functions
        """
        precomputed_rules ={}
        for rule_name, functions in rules.items():
            rule_functions = []
            for function_name in functions:
                rule_functions.append(
                    [
                        self.universe.functions[function_name](i)
                        for i
                        in self.universe_x_values
                    ]
                )
            
            if len(rule_functions) == 1:
                precomputed_rules[rule_name] = rule_functions[0]
            
            else:
                precomputed_rules[rule_name] = [
                    max(col)
                    for col
                    in zip(*rule_functions)
                ]
                
        return precomputed_rules
    
    def intersect_normalized_rules_with_precomputed_functions(
            self,
            x: dict[str, float]
        ) -> list[list[float]]:
        """
        Intersects the normalized rules with the precomputed functions, getting
        the smalles value per each point
        
        Parameters
        ----------
        x: dict[str, float]
            All the activated normalized rules
            
        Returns
        -------
        list[list[float]]:
            All the intersected funcitons with the normalized values
        """
        return [
            [
                min(value, i)
                for i
                in self.precomputed_functions[name]
            ]
            for name, value
            in x.items()
        ]
    
    def get_maximum_value_per_point(
            self,
            y: list[list[float]]
        ) -> list[float]:
        """
        For each intersected function, gets the maximum value
        of all the interscted functions in that point

        Parameters
        ----------
        y: list[list[float]]
            Intersected functions

        Returns
        -------
        list[float]:
            Maximum value of each point
        """
        return [
            max(col)
            for col
            in zip(*y)
        ]
    
    def get_weight_per_point(
            self,
            y: list[float]
        ) -> list[float]:
        """
        Returns the weight per point, the farther a point is on either
        x or y axis, the more weight it will have

        Parameters
        ----------
        y: list[list[float]]
            Maximum value of each point
        Returns
        -------
        list[float]:
            Weight of each point
        """
        return [
            i * j
            for i,j
            in zip(self.universe_x_values, y)
        ]
        
    def __call__(
            self,
            x_normalized: dict[str, float],
        ) -> float:
        """
        Forward pass of the Tsukamoto consequents

        Parameters
        ----------
        x_normalized: dict[str, float]
            Normalized dict per rule

        Returns
        -------
        float:
            Center of mass of the resulting figure
        """
        y = self.intersect_normalized_rules_with_precomputed_functions(
            x = x_normalized
        )

        y_max = self.get_maximum_value_per_point(
            y = y
        )

        y_weight = self.get_weight_per_point(
            y = y_max
        )

        added_y_max = sum(y_max)

        center_of_mass = sum(y_weight) / added_y_max

        return center_of_mass
