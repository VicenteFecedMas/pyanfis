"""Main class of ANFIS"""
from dataclasses import dataclass, field
from typing import Any

from pyanfis.antecedents import Antecedents
from pyanfis.rules import RulesBase, RulesNeuron
from pyanfis.consequents import Consequents


@dataclass(slots = True)
class ANFIS():
    """
    Main ANFIS class, used to make predictions
    
    Attributes
    ----------
    antecedents: Antecedents
        Antecedents part of the ANFIS
    rules_base: RulesBase
        Place where all the rules are going to be stored
    rules_neuron: RulesNeuron
        Computations performed in the rules layer
    consequents: Consequents
        Consequents part of the ANFIS
    config_antecedents: dict[str, Any]
        Configuration of the Antecedents
    config_rules: dict[str, Any]
        Configuration of the Rules
    config_consequents: dict[str, Any]
        Configuration of the Consequents
    """
    antecedents: Antecedents = field(init = False)
    rules_base: RulesBase = field(init = False)
    rules_neuron: RulesNeuron = field(init = False)
    consequents: Consequents = field(init = False)
    config_antecedents: dict[str, Any] = field(repr = False)
    config_rules: dict[str, Any] = field(repr = False)
    config_consequents: dict[str, Any] = field(repr = False)

    def __post_init__(self) -> None:
        self.antecedents = Antecedents(
            self.config_antecedents
        )
        self.rules_base = RulesBase(
            rules_list = self.config_rules["rules"],
            config_antecedents = self.config_antecedents,
            config_consequents = self.config_consequents

        )
        self.rules_neuron = RulesNeuron(
            intersection_type = self.config_rules["intersection_type"]
        )
        self.consequents = Consequents(
            config_consequents = self.config_consequents,
            consequent_rules = self.rules_base.rules["Consequents"]
        )

        del self.config_antecedents
        del self.config_rules
        del self.config_consequents

    def normalize(
            self,
            rules: dict[str, float]
        ) -> dict[str, float]:
        """
        Normalize a set of related rules
        
        Parameters
        ----------
        rules: dict[str, float]
            Related rules
            
        Returns
        -------
        dict[str, float]:
            Normalized rules
        """
        denominator = sum([
            number for number in rules.values()
        ])

        return {
            name: value / denominator
            for name, value
            in rules.items()
        }

    def __call__(
            self,
            x: dict[str, int | float],
            y: dict[str, int | float] | None = None
        ) -> dict[str, int | float]:
        """
        Forward pass but with the added preprocessing

        Parameters
        ----------
        x: dict[str, int | float]
            Input parameters
        y: dict[str, int | float] | None
            Result of the inputs parameters

        Returns
        -------
        dict[str, int | float]:
            Output of the ANFIS
        """
        fuzzyfied = self.antecedents(x)
        print(fuzzyfied)
        related = self.rules_neuron(
            x = fuzzyfied,
            rules = self.rules_base.rules["Antecedents"]
        )
        print(related)
        normalized = self.normalize(related)
        print(normalized)
        output = self.consequents(
            x_normalized = normalized,
            x = x,
            y = y
        )
        return output
