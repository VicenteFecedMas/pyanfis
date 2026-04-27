"""Here you decide which type of consequents you will use"""
from dataclasses import dataclass, field
from typing import Any

from pyanfis.consequents.types import  Tsukamoto

CONSEQUENT_MAPPING: dict[str, Any] = {
    "Tsukamoto": Tsukamoto
}


@dataclass(slots = True)
class Consequents():
    """
    This class will contain all the different types of consequents.

    Attributes
    ----------
    universes : dict[str, Any]
        Universes in the consequents layer
    universes_can_have_forward_updates: dict[str, bool]
        Flags if a universe can be updated on the forward propagation
    config_consequents: dict[str, Any]
        Configuration of the universes
    consequent_rules: dict[str, list[str]]
        Rules of the consequents
    """
    universes: dict[str, Any] = field(init = False)
    config_consequents: dict[str, Any] = field(repr = False)
    consequent_rules: dict[str, list[str]] = field(repr = False)

    def __post_init__(
            self
        ) -> None:
        
        self.universes = {
            name: CONSEQUENT_MAPPING[values["type"]](
                **{"config_universe": values["parameters"], "consequent_rules": self.consequent_rules[name]}
            )
            for name, values
            in self.config_consequents.items()
        }

        del self.config_consequents
        del self.consequent_rules

    def __call__(
            self,
            x_normalized: dict[str, float],
            x: dict[str, int | float] | None,
            y: dict[str, int | float] | None,
        ) -> dict[str, float]:
        """
        Returns all parameters parsed through the respective consequents

        Parameters
        ----------
        x_normalized: dict[str, float]
            Normalized rules
        x: dict[str, int | float] | None
            Input parameters
        y: dict[str, int | float] | None
            Result of the computations over the input parameters

        Returns
        -------
        dict[str, float]:
            Result per output universe
        """
        return {
            name: universe(
                x_normalized = x_normalized
            )
            for name, universe
            in self.universes.items()
        }
    