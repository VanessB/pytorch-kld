import torch
import torchfd
import math


class BasicDenseT(torchfd.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=100) -> None:
        super().__init__()
        
        self.linear_1 = torch.nn.Linear(X_dim + Y_dim, inner_dim)
        self.linear_2 = torch.nn.Linear(inner_dim, inner_dim)
        self.linear_3 = torch.nn.Linear(inner_dim, 1)
        
        self.activation = torch.nn.LeakyReLU()
        

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:       
        layer = torch.cat((x, y), dim=1)
        
        # First layer.
        layer = self.linear_1(layer)
        layer = self.activation(layer)
        
        # Second layer.
        layer = self.linear_2(layer)
        layer = self.activation(layer)
        
        # Third layer.
        layer = self.linear_3(layer)
        
        return layer