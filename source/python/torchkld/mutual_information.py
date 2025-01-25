import torch

from collections.abc import Callable


class MINE(torch.nn.Module):
    """
    Base class for neural network that computes T-statistics for MINE.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    def get_mutual_information(
        self,
        dataloader,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device,
        marginalize: bool="permute",
        transform=None
    ) -> float:
        """
        Mutual information estimation.
        
        Parameters
        ----------
        dataloader
            Data loader. Must yield tuples (x,y,z).
        loss : callable
            Mutual information neural estimation loss.
        device
            Comoutation device.
        marginalize : str, optional
            Method of marginalizing the joint distribution.
            Is either "permute" or "product".
        """
        
        # Disable training.
        was_in_training = self.training
        self.eval()
        
        sum_loss = 0.0
        total_elements = 0
        
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                x, y = batch
                batch_size = x.shape[0]

                x, y = x.to(device), y.to(device)

                if not (transform is None):
                    x, y = transform(x, y)
            
                T_joined   = self(x.to(device), y.to(device))
                T_marginal = self(x.to(device), y.to(device), marginalize=marginalize)
                
                sum_loss += loss(T_joined, T_marginal).detach().cpu().item() * batch_size
                total_elements += batch_size
                
        mutual_information = (-1 if loss.is_lower_bound else 1) * sum_loss / total_elements
                
        # Enable training if was enabled before.
        self.train(was_in_training)
        
        return mutual_information

    def forward(self, x: torch.Tensor, y: torch.Tensor, marginalize: bool=False) -> torch.Tensor:
        if isinstance(marginalize, bool):
            if marginalize:
                marginalize = "permute"
            else:
                return x, y
                
        elif not isinstance(marginalize, str):
            raise ValueError("`marginalize` must be either `bool` or `str`")
        elif marginalize not in ["permute", "product"]:
            raise ValueError("`marginalize` must be either 'permute' or 'product'")
        
        if marginalize == "permute":
            x = x[torch.randperm(x.shape[0])]
        elif marginalize == "product":
            x_repeat_shape = [1 for i in range(len(x.shape))]
            x_repeat_shape[0] = y.shape[0]

            y = y.repeat_interleave(x.shape[0], dim=0)
            x = x.repeat(x_repeat_shape)

        return x, y