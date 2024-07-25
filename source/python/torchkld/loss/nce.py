import math
import torch

from .utils import BaseVariationalBoundLoss


class InfoNCE(BaseVariationalBoundLoss):
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

    def forward(self, T_product: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_product : torch.tensor
            Critic network value on all pairs of samples from the batch.
        """

        batch_size = math.isqrt(T_product.shape[0])
        T_product = T_product.view((batch_size, batch_size))

        log_softmax = torch.nn.functional.log_softmax(T_product, dim=0)
        