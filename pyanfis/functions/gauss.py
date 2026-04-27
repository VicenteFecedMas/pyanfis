"""Gauss function"""
from dataclasses import dataclass
import math


@dataclass(slots = True)
class Gauss():
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : int | float | None
        Center of the gauss function
    std : int | float | None
        Width of the gauss function
    """
    mean: int | float | None = None
    std:  int | float | None = None

    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through Gauss function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the Gauss function
        """
        x = x - self.mean
        x = (x) ** 2
        x = -(x) / (2 * (self.std ** 2))
        x = math.exp(x)
        return x
