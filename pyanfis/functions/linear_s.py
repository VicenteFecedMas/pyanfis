"""Linear S function"""
from typing import Optional, Union
import torch

from .utils import init_parameter

class LinearS(torch.nn.Module):
    """
    Applies a linear S transformation to the incoming data.

    Attributes
    ----------
    foot : float | torch.Tensor | None
        foot of the linear S function
    shoulder : float | torch.Tensor | None
        shoulder of the linear S function

    Returns
    -------
    torch.Tensor
        a tensor of equal size to the input tensor
    """
    __slots__ = ["foot", "shoulder"]
    def __init__(
            self,
            foot: Optional[Union[float, torch.Tensor]] = None,
            shoulder:Optional[Union[float, torch.Tensor]] = None
        ) -> None:
        super().__init__() # type: ignore
        self.shoulder: torch.Tensor = init_parameter(shoulder)
        self.foot: torch.Tensor = init_parameter(foot)
    def __setattr__(self, name: str, value: Optional[Union[int, float, torch.Tensor]]): # type: ignore
        """Item setter dunder method"""        
        super().__setattr__(name, init_parameter(value))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns input parsed through Linear S function"""
        x = x - self.foot
        x = x / (self.shoulder - self.foot)
        x = torch.maximum(x, torch.tensor(0))
        x = torch.minimum(x, torch.tensor(1))
        return x
