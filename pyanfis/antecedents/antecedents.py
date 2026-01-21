"""Antecedents class, it is a group of Universes"""
from typing import Any

import torch

from pyanfis.functions import Universe

class Antecedents(torch.nn.Module):
    """
    Class used to store the antecedents

    Attributes
    ----------
    universes : dict
        dict where all the universes are going to be stored
    
    Methods
    -------
    automf(n_func)
        automatically asign equally spaced functions to all the 
        universes inside the antecedents
    """
    __slots__ = [
        "universes"
    ]

    def __init__(
            self,
            universes: dict[str, Any]
        ) -> None:
        super().__init__()
        self.universes: dict[str, Universe] = {
            name: Universe(**values)
            for name, values
            in universes.items()
        }

    def automf(
            self,
            n_func: int = 2
        ) -> None:
        """
        Automatically asign equally spaced functions to all the 
        universes inside the antecedents

        Attributes
        ----------
        n_func : int
            number of functions per universe
        """
        for key in self.universes.keys():
            self.universes[key].automf(n_func=n_func)

    def forward(
            self ,
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of antecedents, returns parsed antecedents

        Attributes
        ----------
        x : torch.Tensor
            tensor with values multipliers to variables
        """
        fuzzy = torch.cat(
            [
                universe(x[:, :, i:i+1])
                for i, universe in enumerate(self.universes.values())
            ],
            dim=2,
        )

        return torch.nan_to_num(fuzzy, nan = 1.0)
