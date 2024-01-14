import torch
from antecedents.functions import *
import matplotlib.pyplot as plt

class Universe(torch.nn.Module):
    
    def __init__(self, x: torch.Tensor, name:str, merge:bool, heaviside:bool, universe:dict={}, atol:float=0.0, rtol:float=0.1):
        super(Universe, self).__init__()
        self.min, self.max = (torch.min(x) ,torch.max(x))
        self.name = name

        self.merge = merge

        self.heaviside = heaviside

        self.universe = universe

        self.atol = atol
        self.rtol = rtol

    def display(self) -> None:
        plt.title(self.name)
        for key, function in self.universe.items():
            plt.plot(torch.linspace(self.min, self.max, 1000), function(torch.linspace(self.min, self.max, 1000)).detach().numpy(), label=key)

        plt.legend()
        plt.show()

    def rename(self, old_names: list, new_names: list) -> None:
        for old, new in zip(old_names, new_names):
            self.universe[new] = self.universe.pop(old)


    def get_centers_and_intervals(self, n_func: int) -> tuple:
        
        interval = (self.max - self.min)/ (n_func - 1)

        return [float(self.min + interval * i) for i in range(n_func)], [float(interval) for _ in range(n_func)]

    def automf(self, n_func):
        centers, intervals = self.get_centers_and_intervals(n_func=n_func)
        
        self.universe = {f"Gauss_{i}": Gauss(mean=center, std=interval) for i, (center, interval) in enumerate(zip(centers, intervals))}

        
    def forward(self , x: torch.Tensor) -> torch.Tensor:
        fuzzy = torch.empty(x.size(0), x.size(1), len(self.universe))

        for i, (key, function) in enumerate(self.universe.items()):
            if i == 0 and self.heaviside == True:
                fuzzy[:, :, i:i+1] = Heaviside(right_equation=function)(x)
            elif i == len(self.universe) and self.heaviside == True:
                fuzzy[:, :, i:i+1] = Heaviside(left_equation=function)(x)
            else:
                fuzzy[:, :, i:i+1] = function(x)

        return fuzzy

class Antecedents(torch.nn.Module):
    '''
    Responsible for storing all the universes in 1 place
    '''
    def __init__(self, x: torch.Tensor, merge:bool=False, heaviside:bool=False):
        super(Antecedents, self).__init__()
        self.x = x
        self.n_vars = list(range(x.size(-1)))

        self.merge = merge

        self.heaviside = heaviside

        self.universes = {str(i): Universe(x=x[:,:,i],
                                           merge=self.merge,
                                           heaviside=self.heaviside,
                                           name=str(i)) for i, _ in enumerate(self.n_vars)}

    def rename(self, old_names: list, new_names: list) -> None:
        for old, new in zip(old_names, new_names):
            self.universes[new] = self.universes.pop(old)
            self.universes[new].name = new

    def display(self, universe_list: list) -> None:

        for universe in universe_list:
            plt.title(universe)
            self.universes[universe].display()

    def automf(self, n_func: int=100) -> None:
        for i, _ in enumerate(self.n_vars):
            self.universes[str(i)].automf(n_func=n_func)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        width = len([function for key, universe in self.universes.items() for key, function in universe.universe.items()])
        fuzzy = torch.zeros(x.size(0), x.size(1), width)
        
        for i, (key, universe) in enumerate(self.universes.items()):
            fuzzy[:, :, i*len(universe.universe):(i+1)*len(universe.universe)] = universe(x[:,:,i:i+1])
        
        fuzzy[torch.isnan(fuzzy)] = 1
        return fuzzy