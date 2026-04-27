"""Linear Z function"""
from dataclasses import dataclass


@dataclass(slots = True)
class LinearZ():
    """
    Applies a linear Z transformation to the incoming data.

    Attributes
    ----------
    foot : int | float | None
        foot of the linear Z function
    shoulder : int | float | None
        shoulder of the linear Z function
    """
    shoulder: int | float | None = None
    foot: int | float | None = None

    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through LinearZ function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the LinearZ function
        """
        x = self.shoulder - x
        x = x / (self.foot - self.shoulder)
        x = x + 1
        x = min(x, 1)
        x = max(x, 0)
        return x
