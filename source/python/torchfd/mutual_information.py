import torch

from collections.abc import Callable
from typing import Any

# TODO:
# from .loss.utils import BaseVariationalBoundLoss


class Marginalizer(torch.nn.Module):
    """
    Base class for modules which marginalize joint distributions.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        function: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class PermutationMarginalizer(Marginalizer):
    """
    Permutation-based marginalizer, as described in [1].

    References
    ----------
    .. [1] M. I. Belghazi et al., "Mutual Information Neural Estimation".
           Proc. of ICML 2018.
    """

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        function: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_permuted = x[torch.randperm(x.shape[0])]
        
        return function(x, y), function(x_permuted, y)

class OuterProductMarginalizer(Marginalizer):
    """
    Outer-product-based marginalizer, as described in [1].

    References
    ----------
    .. [1] Oord A., Li Y. and Vinyals O. "Representation Learning with
           Contrastive Predictive Coding". arXiv:1807.03748
    """

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        function: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        x_repeat_shape = [1 for i in range(len(x.shape))]
        x_repeat_shape[0] = y.shape[0]

        x_repeated = x.repeat(x_repeat_shape)
        y_repeated = y.repeat_interleave(x.shape[0], dim=0)

        T_marginal = function(x_repeated, y_repeated)
        T_joined = torch.diag(T_marginal.view((batch_size, batch_size))) # A shortcut to avoid needless computation.
        
        return T_joined, T_marginal


class MINE(torch.nn.Module):
    """
    Base class for neural network that computes T-statistics for MINE [1]
    and similar variational methods.

    Parameters
    ----------
    marginalizer : Marginalizer
        An instance of a `Marginalizer` base class which is used to convert
        samples from a joint distribution to samples from a product of
        marginal distributions.

    References
    ----------
    .. [1] M. I. Belghazi et al., "Mutual Information Neural Estimation".
           Proc. of ICML 2018.
    """
    
    def __init__(
        self, 
        marginalizer: Marginalizer=None
    ) -> None:
        super().__init__()

        self.marginalizer = marginalizer
        if self.marginalizer is None:
            self.marginalizer = PermutationMarginalizer()
        
    def get_mutual_information(
        self,
        dataloader,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device,
        clip: float=None,
    ) -> float:
        """
        Mutual information estimation.
        
        Parameters
        ----------
        dataloader
            Data loader. Must yield tuples (x,y).
        loss : Callable
            Mutual information neural estimation loss.
        device
            Comoutation device.
        clip : float, optional
            Clipping treshold for SMILE [1]. No clipping if None.

        References
        ----------
        .. [1] Song, J. and Ermon, S. "Understanding the Limitations of
               Variational Mutual Information Estimators". Proc. of ICLR 2020.
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
            
                T_joined, T_marginal = self(x.to(device), y.to(device))

                if not (clip is None):
                    T_marginal = torch.clamp(T_marginal, -clip, clip)
                
                sum_loss += loss(T_joined, T_marginal).detach().cpu().item() * batch_size
                total_elements += batch_size
                
        mutual_information = (-1 if loss.is_lower_bound else 1) * sum_loss / total_elements
                
        # Enable training if was enabled before.
        self.train(was_in_training)
        
        return mutual_information

    #@staticmethod
    def marginalized(
        function: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]:
        def wrapped(self, x, y):
            return self.marginalizer(x, y, lambda x, y : function(self, x, y))

        return wrapped

    @marginalized
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError