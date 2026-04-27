"""Antecedents class, it is a group of Universes"""
from dataclasses import dataclass, field
from typing import Any

from pyanfis.functions import Universe


@dataclass(slots = True)
class Antecedents():
    """
    Class used to store the antecedents

    Attributes
    ----------
    config_universes: dict[str, Any]
        Configuration of the universes in the Antecedents
    universes : dict
        dict where all the universes are going to be stored
    """
    config_universes: dict[str, Any] = field(repr = False)
    universes: dict[str, Universe] = field(init = False)

    def __post_init__(
            self
        ) -> None:

        self.universes = {
            name: Universe(**values)
            for name, values
            in self.config_universes.items()
        }

        del self.config_universes

    def __call__(
            self ,
            x: dict[str, int | float]
        ) -> dict[str, dict[str, float]]:
        """
        Forward pass of antecedents, returns parsed antecedents

        Parameters
        ----------
        x: dict[str, int | float]
            Dict with inputs, per input

        Returns
        -------
        dict[str, dict[str, float]]:
            Dict with output of the antecedents
        """
        return {
                name: universe(x[name])
                for name, universe
                in self.universes.items()
        }
