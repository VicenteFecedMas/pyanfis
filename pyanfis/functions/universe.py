"""Universe class, it encapsulates several functions related to a variable"""
from dataclasses import dataclass, field
from typing import Any

from .sigmoid import Sigmoid
from .gauss import Gauss
from .bell import Bell
from .linear_z import LinearZ
from .linear_s import LinearS
from .triangular import Triangular

FUNCTION_MAPPING = {
    "Sigmoid": Sigmoid,
    "Gauss": Gauss,
    "Bell": Bell,
    "LinearZ": LinearZ,
    "LinearS": LinearS,
    "Triangular": Triangular
}


@dataclass(slots = True)
class Universe():
    """
    A Universe encapsulate several functions related to an input

    Attributes
    ----------
    config_functions: dict[str, Any]
        Configuration of the universe
    functions: dict[str, Any]
        Functions inside the universe
    maximum: float
        Maximum boundary of the universe
    minimum: float
        Minimum boundary of the universe
    """
    config_functions: dict[str, Any] = field(repr = False)
    functions: dict[str, Any] = field(init = False)
    maximum: float
    minimum: float

    def __post_init__(
            self
        ) -> None:

        self.functions = {
            name: self._load_function(
                values["type"],
                values["parameters"]
            )
            for name, values
            in self.config_functions.items()
        }

        del self.config_functions

    def _load_function(
            self,
            function_type: str,
            function_parameters: dict[str, Any]
        ) -> Any:
        """
        Returns a function given a name and its parameters

        Parameters
        ----------
        function_type: str
            Type of function to be loaded
        function_parameters: dict[str, Any]
            Parameters to load into into the function

        Returns
        -------
        Any:
            Any given function valid for the ANFIS
        """
        if function_type not in FUNCTION_MAPPING:
            raise ImportError(f"Class {function_type} not found in the 'functions' folder.")

        return FUNCTION_MAPPING[function_type](**function_parameters)

    def __call__(
            self,
            x: int | float
        ) -> dict[str, float]:
        """
        Returns input parsed through a Universe
        
        Parameters
        ----------
        x: int | float
            Value to be transformed
            
        Returns
        -------
        dict[str, float]:
            Parsed parameters through a Universe
        """
        if not self.functions:
            raise ValueError("Forward pass impossible, universe contains no functions")

        return {
            name: function(x)
            for name, function
            in self.functions.items()
        }
