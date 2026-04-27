"""Triangular function"""
from dataclasses import dataclass


@dataclass(slots = True)
class Triangular():
    """
    Applies a sigmoid transformation to the incoming data.
s
    Attributes
    ----------
    left_foot : float | torch.Tensor | None
        Left foot of the triangular function
    peak : float | torch.Tensor | None
        Peak of the triangular function
    right_foot : float | torch.Tensor | None
        Right foot of the triangular function
    """
    left_foot: int | float | None = None
    peak: int | float | None = None
    right_foot: int | float | None = None

    def __call__(
            self,
            x: int | float
        ) -> float:
        """
        Returns input parsed through Triangular function
        
        Parameters
        ----------
        x: int | float
            Number to be transformed
            
        Returns
        -------
        float:
            Parsed parameters through the Triangular function
        """
        term1 = (x - self.left_foot) / (self.peak - self.left_foot)
        term2 = (self.right_foot - x) / (self.right_foot - self.peak)
        min_term = min(term1, term2)
        x = max(min_term, 0)
        return x
