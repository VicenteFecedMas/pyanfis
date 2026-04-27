"""Bell function"""
from dataclasses import dataclass


@dataclass(slots = True)
class Bell():
    """
    Applies a bell transformation to the incoming data.

    Attributes
    ----------
    width : int | float | None
        Width of the bell function
    shape : int | float | None
        Shape of the transition area of the bell function
    center : int | float | None
        Center of the bell function
    """
    width: int | float | None = None
    shape: int | float | None = None
    center: int | float | None = None
        
    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through Bell function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the Bell function
        """
        x = x - self.center
        x = x / self.width
        x = abs(x) ** (2 * self.shape)
        x = x + 1
        x = 1 / x
        return x
