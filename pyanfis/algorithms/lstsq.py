"""lstsq algorithm to compute a set of parameters given the input and output variables"""
import torch

class LSTSQ(torch.nn.Module):
    """
    Computes the vector x that approximately solves the equation a @ x = b.

    Attributes
    ----------
    n_vars : int
        number of variables.
    alpha : float
        alpha value for updating theta.
    driver: str
        driver for the lstsq algorithm.
    """
    __slots__ = [
        "theta",
        "alpha",
        "driver"
    ]
    def __init__(
            self,
            n_vars: int,
            alpha: float = 0.001,
            driver: str = 'gels'
        ) -> None:
        super().__init__()
        self.theta = torch.zeros((n_vars, 1))
        self.alpha = alpha
        self.driver = driver

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor|None = None
        ) -> None:
        """
        Compute a new value of 'theta'.

         Attributes
        ----------
        x : torch.Tensor
            tensor with values multipliers to variables
        y : torch.Tensor
            tensor with values that are results
        """
        for _ in range(x.size(0)):
            new_theta = torch.linalg.lstsq(x, y, driver=self.driver).solution

            if new_theta.dim() > 2:
                new_theta = new_theta.mean(dim=0)
                
            self.theta = (1 - self.alpha) * self.theta + self.alpha * new_theta

        return None
