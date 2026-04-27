"""This class will the hold the neuron that relate fuzzy numbers given a set of rules"""
from dataclasses import dataclass, field
from typing import Any
import torch
from .intersection_algorithms import larsen, mamdani

INTERSECTIONS = {
    "larsen": larsen,
    "mamdani": mamdani,
}


@dataclass(slots = True)
class RulesNeuron():
    """
    Holds all the logic to intersect fuzzy numbers, per rule

    Attributes
    ----------
    intersection_type:str
        Name of the type of intersection
    intersection: Any
        Intersection algorithm
    """
    intersection_type:str = field(repr = False)
    intersection: Any = field(init = False)

    def __post_init__(
            self
        ) -> None:
        self.intersection = INTERSECTIONS[self.intersection_type]

        del self.intersection_type

    def __call__(
            self,
            x: dict[str, dict[str, float]],
            rules: dict[str, tuple[tuple[str, str]]]
        ) -> dict[str, float]:
        """
        Relate fuzzy numbers using rules

        Parameters
        ----------
        x: dict[str, dict[str, float]]
            Fuzzyfied numbers
        rules: dict[str, tuple[tuple[str, str]]]
            Antecedent part of the rules

        Returns
        -------
        dict[str, float]:
            Related fuzzy numbers per rule
        """
        x_intersected = {
            name: [
                x[universe][function]
                for universe, function
                in values
            ]
            for name, values
            in rules.items()
        }

        return self.intersection(x_intersected)
