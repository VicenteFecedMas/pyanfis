import torch

def init_parameter(number=None):
    if not isinstance(number, (int, float)) and number is not None:
        raise TypeError(f"Expected an int, a float, or None but got {type(number)}")


    if number is None:
        return torch.nn.Parameter(None, requires_grad=False)
    elif isinstance(number, torch.Tensor):
        return torch.nn.Parameter(number, requires_grad=True)
    else:
        return torch.nn.Parameter(torch.tensor(number, dtype=float), requires_grad=True)