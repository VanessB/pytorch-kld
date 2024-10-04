import math
import torch

from .utils import BaseVariationalBoundLoss


class InfoNCELoss(BaseVariationalBoundLoss):
    """
    Noise-Contrastive Estimation variational lower bound for
    the mutual information.

    References
    ----------
    .. [1] Oord A., Li Y. and Vinyals O. "Representation Learning with
           Contrastive Predictive Coding". arXiv:1807.03748
    """
    
    def __init__(self):
        super().__init__()

        self.is_lower_bound = True

    def forward(self, T_joined: torch.tensor, T_product: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_joined : torch.tensor
            Critic network value on all samples from the batch.
        T_product : torch.tensor
            Critic network value on all possible pairs of samples from the batch.
        """

        batch_size = T_joined.shape[0]
        T_product = T_product.view((batch_size, batch_size))

        return torch.mean(torch.logsumexp(T_product - T_joined, dim=-1)) - math.log(batch_size)