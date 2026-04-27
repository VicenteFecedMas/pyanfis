"""Linear S function"""
from dataclasses import dataclass


@dataclass(slots = True)
class LinearS():
    """
    Applies a linear S transformation to the incoming data.

    Attributes
    ----------
    foot : int | float | None
        Foot of the linear S function
    shoulder : int | float | None
        Shoulder of the linear S function
    """
    shoulder: int | float | None = None
    foot: int | float | None = None
    
    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through LinearS function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the LinearS function
        """
        x = x - self.foot
        x = x / (self.shoulder - self.foot)
        x = max(x, 0)
        x = min(x, 1)
        return x
