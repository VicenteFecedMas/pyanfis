"""This class will hold the rules of a system"""
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(slots = True)
class RulesBase():
    """
    Holds all the rules, both for the antecedents and the consequents

    Attributes
    ----------
    rules: dict[str, Any]
        Dictionary with rules, separated by antecedents and consequents
    rules_list: list[str]
        List with all the rules
    config_antecedents: dict[str, Any]
        Configuration of the antecedents
    config_consequents: dict[str, Any]
        The configuration of the consequents
    """
    rules: dict[str, Any] = field(init = False)
    rules_list: list[str] = field(repr = False)
    config_antecedents: dict[str, Any] = field(repr = False)
    config_consequents: dict[str, Any] = field(repr = False)
    
    def __post_init__(
            self
        ) -> None:
        universes_keys_antecedents, functions_keys_antecedents = self.get_keys(self.config_antecedents)
        universes_keys_consequents, functions_keys_consequents = self.get_keys(self.config_consequents)
        self.rules = self.extract_rules(
            rules = self.rules_list,
            antecedents_pattern = self.get_pattern(
                universes_keys = universes_keys_antecedents,
                functions_keys = functions_keys_antecedents
            ),
            consequents_pattern = self.get_pattern(
                universes_keys = universes_keys_consequents,
                functions_keys = functions_keys_consequents
            )
        )

        del self.rules_list
        del self.config_antecedents
        del self.config_consequents

    def get_keys(
            self,
            config: dict[str, Any]
        ) -> tuple[list[str], list[list[str]]]:
        """
        Retrieves a series of universe and function keys from a configuration

        Parameters
        ----------
        config: dict[str, Any]
            Configuration of Antecedents or Consequents

        Returns
        -------
        tuple[list[str], list[list[str]]]:
            The keys that we are looked for
        """ 
        universes_keys, functions_keys = [], []
        for key, item in config.items():
            universes_keys.append(key)
            if "config_functions" in item:
                functions_keys.append(list(item["config_functions"].keys()))
            else:
                functions_keys.append(list(item["parameters"]["config_functions"].keys()))

        return universes_keys, functions_keys

    def get_pattern(
            self,
            universes_keys: list[str],
            functions_keys: list[list[str]]
        ) -> str:
        """
        Retrieves a pattern from a sequence of keys

        Parameters
        ----------
        universes_keys: list[str]
            Keys of the universes
        functions_keys: list[list[str]]
            Keys of the functions

        Returns
        -------
        str:
            The desired patterns
        """
        patterns = [
            f"{universe} is ({"|".join(functions)})"
            for universe, functions
            in zip(universes_keys, functions_keys)
        ]

        return f"^({"|".join(patterns)})$"
    
    def extract_rules(
            self,
            rules: list[str],
            antecedents_pattern: str,
            consequents_pattern: str
        ) -> dict[str, Any]:
        """
        Extract the rule value-pairs from a list of rules

        Parameters
        ----------
        rules: list[str]
            List of all the rules
        antecedents_pattern: str
            Patter for how to extract the rules in the antecedents
        consequents_pattern: str
            Patter for how to extract the rules in the consequents

        Returns
        -------
        dict[str, Any]:
            Dict with all the rule value-pairs divided by antecedents and consequents
        """
        processed_rules = {
            "Antecedents": {},
            "Consequents": {}
        }
        
        for i, rule in enumerate(rules):
            if not rule.startswith("If"):
                raise ValueError(f"Rule must start with 'If ... Got: '{rule[0: 10]}...'")
            
            rule = rule.replace("If ", "")

            if " then " not in rule:
                raise ValueError(f"Expected ' then ' to be present in '{rule}'.")
            
            antecedent, consequent = rule.split(" then ")

            found = self.find_universe_function_tuples(
                sentence=antecedent,
                pattern=antecedents_pattern
            )
            processed_rules["Antecedents"][f"Rule {i}"] = found

            found = self.find_universe_function_tuples(
                sentence = consequent,
                pattern = consequents_pattern
            )

            for universe, function in found:
                if universe not in processed_rules["Consequents"]:
                    processed_rules["Consequents"][universe] = {}

                if f"Rule {i}" not in processed_rules["Consequents"][universe]:
                    processed_rules["Consequents"][universe][f"Rule {i}"] = []

                processed_rules["Consequents"][universe][f"Rule {i}"].append(function)

        return processed_rules

    def find_universe_function_tuples(
            self,
            sentence: str,
            pattern: str
        ) -> tuple[tuple[str, str]]:
        """
        Find the universe - function pairs that are mentioned in the rules

        Parameters
        ----------
        sentence: str
            The sentence to be parsed
        pattern: str
            The names of universes and functions to look for

        Returns
        -------
        tuple[tuple[str, str]]:
            All the possible universe - function combinations
        """
        tuples = []
        for i in sentence.split(" and "):
            if "is" not in i:
                raise ValueError(f"A 'Universe name - Function name' tuple must have 'is' between them. Got: '{i}'")
            
            matches = list(re.finditer(pattern, i))
            if not matches:
                raise ValueError(f"No match for: '{i}'")

            for m in matches:
                universe_function_tuple = tuple(m.group().split(" is "))
                if universe_function_tuple in tuples:
                    raise ValueError(f"Found repeated 'Universe name - Function name' tuple: {universe_function_tuple}")
                tuples.append(universe_function_tuple)

        return tuple(tuples)