"""rlse algorithm to compute a set of parameters given the input and output variables"""
from dataclasses import dataclass, field

import numpy


@dataclass(slots = True)
class RLSE():
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
    s: list[list[int | float]] = field(init = False)
    theta: list[int | float] = field(init = False)
    gamma: int | float
    n_vars: int = field(repr = False)

    def __post_init__(self) -> None:
        
        self.s = [
            [
                self.gamma
                if i == j
                else 0
                for j
                in range(self.n_vars)
            ]
            for
            i in range(self.n_vars)
        ]

        self.theta = [
            0
            for _
            in range(self.n_vars)
        ]

        del self.n_vars

    def transpose(m):
        return list(
              map(
                    list,
                    zip(*m)
                )
            )

def multiply_matrix_by_vector(
            self,
            matrix: list[list[int | float]],
            vector: list[float]
        ) -> list[list[int | float]]:
        return [
            [
                a * b
                for a, b
                in zip(row, vector)
            ]
            for row
            in matrix
        ]
    
    def update_theta(
            self,
            x: dict[str, int | float],
            y: dict[str, int | float]
        ) -> None:
        """
        RLSE by minimizing the weighted least squares cost function
        using the Kalman filter framework.
        """
        #input_rows = x.reshape(x.size(0), -1).float()
        #for row, col in zip(input_rows, y):

        # Unsqueeze for easier computation
        x_array = [i for i in x.values() ]


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
