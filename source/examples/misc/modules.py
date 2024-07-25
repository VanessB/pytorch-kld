import torch
import torchkld
import math


class BasicDenseT(torchkld.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=100) -> None:
        super().__init__()
        
        self.linear_1 = torch.nn.Linear(X_dim + Y_dim, inner_dim)
        self.linear_2 = torch.nn.Linear(inner_dim, inner_dim)
        self.linear_3 = torch.nn.Linear(inner_dim, 1)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)
        
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


class BasicSeparableT(torchkld.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=100) -> None:
        super().__init__()
        
        self.linear_x = torch.nn.Linear(X_dim, inner_dim)
        self.linear_y = torch.nn.Linear(Y_dim, inner_dim)
        self.linear_prod = torch.nn.Linear(inner_dim, inner_dim)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)
        
        # First layer.
        x = self.activation(self.linear_x(x))
        y = self.activation(self.linear_y(y))
        
        # Second layer.
        x = self.linear_prod(x)
        
        return torch.sum(x * y, dim=-1)


class BasicConv2dT(torchkld.mutual_information.MINE):
    def __init__(self, X_size: int, Y_size: int,
                 n_filters: int=16, hidden_dimension: int=128) -> None:
        super().__init__()
        
        log2_remaining_size = 2
        
        # Convolution layers.
        X_convolutions_n = int(math.floor(math.log2(X_size))) - log2_remaining_size
        self.X_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(1, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(X_convolutions_n - 1)])
            
        Y_convolutions_n = int(math.floor(math.log2(Y_size))) - log2_remaining_size
        self.Y_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(1, n_filters, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(n_filters, n_filters, kernel_size=3, padding='same') for index in range(Y_convolutions_n - 1)])
            
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense layer.
        remaining_dim = n_filters * 2**(2*log2_remaining_size)
        self.linear_1 = torch.nn.Linear(remaining_dim + remaining_dim, hidden_dimension)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_3 = torch.nn.Linear(hidden_dimension, 1)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)
            
        # Convolution layers.
        for conv2d in self.X_convolutions:
            x = conv2d(x)
            x = self.maxpool2d(x)
            x = self.activation(x)
            
        for conv2d in self.Y_convolutions:
            y = conv2d(y)
            y = self.maxpool2d(y)
            y = self.activation(y)
            
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        
        layer = torch.cat((x, y), dim=1)
        
        # First dense layer.
        layer = self.linear_1(layer)
        layer = self.activation(layer)
        
        # Second dense layer.
        layer = self.linear_2(layer)
        layer = self.activation(layer)
        
        # Third dense layer.
        layer = self.linear_3(layer)
        
        return layer