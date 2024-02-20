import torch

from .algorithm import Algorithm 


class TakagiSugeno(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, parameters_update) -> None:
        super().__init__()
        self.universes = {f"Output {i+1}" :  Algorithm(num_inputs, parameters_update) for i in range(num_outputs)}
        self.parameters_update = parameters_update
        self.num_outputs = num_outputs
        
    def forward(self, X, f, Y):

        ones = torch.ones(X.shape[:-1] + (1,), dtype=X.dtype)
        X = torch.cat([X, ones], dim=-1)

        x_b, x_i, _ = X.size()

        output = {f"Output {i+1}": torch.zeros((x_b, x_i , 1)) for i in range(self.num_outputs)}
        X = torch.einsum('bri, brj -> brij', f, X).view(x_b, x_i, -1)

        for i, (key, algorithm) in enumerate(self.universes.items()):
            if self.training:
                algorithm.training = self.training
                algorithm(X, Y[:, :, i:i+1])
            output[key] = output[key] + torch.einsum('bij, jk -> bik', X, algorithm.theta)
   
        return output 