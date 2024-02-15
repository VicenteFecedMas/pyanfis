import torch

from functions import Universe


class Tsukamoto(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, parameters_update) -> None:
        super().__init__()
        self.universes = {f"Output {i+1}": Universe() for i in range(num_outputs)}
        self.parameters_update = parameters_update
        self.num_outputs = num_outputs
        self.active_rules = None

    def forward(self, f):
        outputs = {"Output 1": torch.zeros(f.size(0), f.size(1), 1)}
        for ith_output, (key, universe) in enumerate(self.universes.items()):
            X = torch.linspace(universe.min, universe.max, 200)
            functions_list = []
            debug = []
            for name, function in universe.functions.items():
                functions_list.append(function(X))
                debug.append(name)

            function_rules = None
            for rule in self.active_rules:
                main_function = None
                for i, num in enumerate(rule):
                    if num == 1:
                        main_function = functions_list[i].unsqueeze(0) if main_function is None else torch.max(main_function, functions_list[i]).unsqueeze(0)

                function_rules = main_function if function_rules is None else torch.cat((function_rules, main_function), dim=0)

            for b, batch in enumerate(f):
                for i, row in enumerate(batch):
                    Y = torch.min(function_rules, row.view(-1, 1))
                    outputs[key][b, i, 0] = torch.sum(X * Y) / torch.sum(Y)
                        


        return outputs