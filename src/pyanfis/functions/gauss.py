import torch
from .utils import init_parameter

class Gauss(torch.nn.Module):
    """
    Applies a gauss transformation to the incoming data.

    Attributes
    ----------
    mean : float
        center of the gauss function
    std : float
        width of the gauss function

    Returns
    -------
    torch.tensor
        a tensor of equal size to the input tensor
    """
    def __init__(self, mean:float = None, std:float = None) -> None:
        super().__init__()
        
        
        #if not std and not isinstance(std, float):
        #    raise ValueError(f"Expected std to be a float number but got {type(std)} instead")
        #elif std <= 0.0:
        #    raise ValueError(f"std value must be greater than 0.0 but got {std}")
    
        #if not mean and not isinstance(mean, float):
        #    raise ValueError(f"Expected mean to be a float number but got {type(std)} instead")
        
        self.mean = init_parameter(mean)
        self.std = init_parameter(std)
    
    def get_center(self) -> torch.Tensor:
        """Util to get the center of the function"""
        return self.mean
    
    def __setitem__(self, key, value):
        if key == "mean":
            self.mean = init_parameter(value)
        elif key == "std":
            self.std = init_parameter(value)
        else:
            raise KeyError(f"Invalid key: {key}")

    def forward(self, x) -> torch.Tensor:
        x = x - self.mean
        x = (x)** 2
        x = -(x)/ (2 * (self.std ** 2))
        x = torch.exp(x)
        return x