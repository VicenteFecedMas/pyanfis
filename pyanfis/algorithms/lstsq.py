"""lstsq algorithm to compute a set of parameters given the input and output variables"""
import torch

@dataclass(slots=True)
class LSTSQ(torch.nn.Module):
    """
    Computes the vector x that approximately solves the equation a @ x = b

    Attributes
    ----------
    n_vars : int
        number of variables
    alpha : float
        alpha value for updating theta
    driver: str
        driver for the lstsq algorithm
    """
    n_vars: int
    aplha: float = 0.001
    driver: str = "gels"
    theta: torch.Tensor = field(
        init = False,
        repr = False
    )

    def __post_init__(self):
        super().__init__()
        self.theta = torch.zeros(
            (
                self.n_vars,
                1
            )
        )

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor
        ) -> None:
        """
        Compute a new value of 'theta'

         Attributes
        ----------
        x : torch.Tensor
            tensor with values multipliers to variables
        y : torch.Tensor
            tensor with values that are results
        """
        new_theta = torch.linalg.lstsq(
            x,
            y,
            driver = self.driver
        ).solution
            
        self.theta = (1 - self.alpha) * self.theta + self.alpha * new_theta

        return None
