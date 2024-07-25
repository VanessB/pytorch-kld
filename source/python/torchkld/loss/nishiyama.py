import torch

from .utils import BaseVariationalBoundLoss


class NishiyamaLoss(BaseVariationalBoundLoss):
    """
    Nishiyama lower bound for the Kullback-Leibler divergence.

    References
    ----------
    .. [1] Nishiyama, T. "A new lower bound for kullback-leibler divergence
           based on hammersley-chapman-robbins bound". arXiv:1907.00288
    """
    
    def __init__(self, biased=True):
        super().__init__()

        if not biased:
            raise NotImplementedError("Unbiased loss is not implemented yet")
        self.biased = biased
        
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
        
        E_p = torch.mean(T_p)
        V_p = torch.var(T_p)

        E_q = torch.mean(T_q)
        V_q = torch.var(T_q)

        A = (E_p - E_q)**2 + V_p + V_q
        D = torch.sqrt(A**2 - 4.0 * V_p * V_q)

        return -( (A - 2.0 * V_p) * torch.atanh(D / A) / D + 0.5 * torch.log(V_p / V_q) )