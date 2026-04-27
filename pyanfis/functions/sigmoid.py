"""Sigmoid function"""
from dataclasses import dataclass
import math


@dataclass(slots = True)
class Sigmoid():
    """
    Applies a sigmoid transformation to the incoming data.

    Attributes
    ----------
    center : int | float | None
        Center of the sigmoid function
    width : int | float | None
        Width of the transition area
    """
    center: int | float | None = None
    width: int | float | None = None

    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through Sigmoid function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the Sigmoid function
        """
        x = x - self.center
        x = x / (- self.width)
        x = math.exp(x)
        x = x + 1
        x = 1 / x
        return x
