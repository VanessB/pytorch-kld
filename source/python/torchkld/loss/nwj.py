import torch

from .utils import BaseVariationalBoundLoss


class NWJLoss(BaseVariationalBoundLoss):
    """
    Nguyen-Wainwright-Jordan variational lower bound for
    the Kullback-Leibler divergence.

    References
    ----------
    .. [1] Nguyen, X., Wainwright, M. J., and Jordan, M. I. "Estimating
           divergence functionals and the likelihood ratio by convex risk
           minimization". IEEE Transactions on Information Theory, 56
           (11):5847–5861, 2010.
    """
    
    def __init__(self):
        super().__init__()

        self.is_lower_bound = True

    def forward(self, T_p: torch.tensor, T_q: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_p : torch.tensor
            Critic network value on a batch from the first distribution.
        T_q : torch.tensor
            Critic network value on a batch from the second distribution.
        """
        
        return -torch.mean(T_p) + torch.mean(torch.exp(T_q - 1))