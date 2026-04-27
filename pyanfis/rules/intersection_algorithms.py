"""Intersection algorithms to relate memberships from fuzzy functions"""
import math


def mamdani(
        related_inputs_per_rule: dict[str, list[float]]
    ) -> dict[str, float]:
    """
    Perform product over related fuzzy numbers

    Parameters
    ----------
    related_inputs_per_rule: dict[str, list[float]]
        Related fuzzy numbers

    Returns
    -------
    dict[str, float]:
        Intersected fuzzy numbers
    """
    return {
        name: math.prod(values)
        for name, values
        in related_inputs_per_rule.items()
    }

def larsen(
        related_inputs_per_rule: dict[str, list[float]]
    ) -> dict[str, float]:
    """
    Perform minimum over related fuzzy numbers

    Parameters
    ----------
    related_inputs_per_rule: dict[str, list[float]]
        Related fuzzy numbers

    Returns
    -------
    dict[str, float]:
        Intersected fuzzy numbers
    """
    return {
        name: min(values)
        for name, values
        in related_inputs_per_rule.items()
    }
