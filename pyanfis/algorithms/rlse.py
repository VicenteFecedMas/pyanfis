"""rlse algorithm to compute a set of parameters given the input and output variables"""
import torch

class RLSE(torch.nn.Module):
    """
    Computes the vector x that approximately solves the equation a @ x = b
    using a recursive approach

    Attributes
    ----------
    n_vars : float
        length of the "x" vector
    initial_gamma : float
        big number to initialise the "S" matrix
    """
    __slots__ = [
        "s",
        "theta",
        "gamma"
    ]

    def __init__(
            self,
            n_vars: int,
            gamma: float = 0.99
        ):
        super().__init__()
        self.s = torch.eye(
            n_vars,
            dtype = torch.float32,
            requires_grad = False
        ) * gamma

        self.theta = torch.nn.Parameter(
            torch.zeros(
                (n_vars, 1),
                dtype = torch.float32
            ),
            requires_grad = False
        )

        self.gamma = gamma

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor
        ) -> None:
        """
        RLSE by minimizing the weighted least squares cost function
        using the Kalman filter framework.

        Attributes
        ----------
        x : torch.Tensor
            tensor with values multipliers to variables
        y : torch.Tensor
            tensor with values that are results
        """
        input_rows = x.reshape(x.size(0), -1).float()

        for row, col in zip(input_rows, y):

            # Unsqueeze for easier computation
            i = row.unsqueeze(-1) #5, 1

            # Kalman Gain
            s_x = self.s @ i
            denominator = self.gamma + i.T @ s_x
            K = s_x / denominator # 5, 5
            
            # Update theta
            error = col -  self.theta.T @ i # 1, 1
            self.theta.add_(
                K * error # 5, 1
            )
            
            # Update P
            numerator = self.s - (K @ i.T @ self.s)
            self.s = (numerator) * self.gamma

        return None
