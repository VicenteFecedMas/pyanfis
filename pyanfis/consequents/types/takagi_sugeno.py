"""Consequents that look like a polynome"""
from dataclasses import dataclass, field
from typing import Any

import numpy

from pyanfis.algorithms import LSTSQ, RLSE

ALGORITHM_MAPPING: dict[str, Any] = {
    "LSTSQ": LSTSQ,
    "RLSE":  RLSE,
}


@dataclass(slots = True)
class TakagiSugeno():
    """
    This class will compute the learnable parameters using the Takagi-Sugeno approach.

    Attributes
    ----------
    num_inputs : float
        number of inputs that the system will recive
    num_outputs : float
        number of outputs that the system will produce
    parameters_update : float
        how the system will update the parameters

    Returns
    -------
    dict
        a dictionary that will contain the prediction related to each output
    """

    algorithm: LSTSQ | RLSE = field(init = False)
    config_algorithm: dict[str, Any] = field(repr = False)

    def __post_init__(
            self
        ) -> None:
        if self.config_algorithm["algorithm"] not in ALGORITHM_MAPPING:
            raise ValueError(f"{self.config_algorithm["algorithm"]} not in {list(ALGORITHM_MAPPING.keys())}")
        
        self.algorithm = ALGORITHM_MAPPING[self.config_algorithm["algorithm"]](
            (self.config_algorithm["n_inputs"] + 1) * self.config_algorithm["n_rules"]
        )

    def __call__(
            self,
            x: dict[str, int | float],
            x_normalized: dict[str, float],
            y: dict[str, int | float] | None = None,
            
        ):
        """Forward pass of the Takagi-Sugeno consequents"""


        {
            rule_name: [
                i * value
                for i
                in x.values()
            ] + [value]
            for rule_name, value
            in x_normalized.items()
        }

        if y:
            self.algorithm.update_theta(
                x = x,
                y = y
            )
            

        output = numpy.einsum(
            'bij,jk->bik',
            x,
            self.algorithm.theta
        )

        return output
