import torch

from .intersection_algorithms import larsen, mamdani

INTERSECTIONS = {
    'larsen': larsen,
    'mamdani': mamdani,
}


class Rules(torch.nn.Module):
    """
    This class will contain all the rules of the system,
    it will dictate how each one of the antecedent functions
    relate with each other.

    Attributes
    ----------
    intersection : str
        intersection algorithm that is going to be used

    Methods
    -------
    generate_rules(n_membership_functions_per_universe)
        generate the rules of the universe
    relate_fuzzy_numbers(fuzzy_numbers_matrix)
        parse each input through the set of established rules

    Returns
    -------
    torch.tensor
        a tensor of size [n_batches, n_lines, n_functions]

    Examples
    --------
    """
    def __init__(self, intersection:str = 'larsen'):
        super(). __init__()
        self.active_antecedents_rules = None
        self.active_consequents_rules = None
        self.intersection = INTERSECTIONS[intersection]
    
    def relate_fuzzy_numbers(self, fuzzy_numbers_matrix):
        '''
        INPUT: FN es Funny numbers matrix y R es rules matrix
        OUTPUT: FA Fuzzy And matrix
        '''
        rules_per_universe = torch.zeros((fuzzy_numbers_matrix.size(0), fuzzy_numbers_matrix.size(1), self.active_antecedents_rules.size(0), fuzzy_numbers_matrix.size(2)))
        for b, _ in enumerate(fuzzy_numbers_matrix):
            for i, _ in enumerate(fuzzy_numbers_matrix[b, :, :]):
                rules_per_universe[b, i, :, :] = fuzzy_numbers_matrix[b, i, :] * self.active_antecedents_rules
        return rules_per_universe

    def binarice(self, binary_list: torch.Tensor) -> str:
        return str(int(''.join(str(int(i)) for i in binary_list), 2))
    
    def forward(self, x):
        x = self.intersection(self.relate_fuzzy_numbers(x)) # This is a 4D tensor
        return x[:, :, :, 0]