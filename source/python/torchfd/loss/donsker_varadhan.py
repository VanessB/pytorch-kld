import math
import torch

from .utils import BaseVariationalBoundLoss


class DonskerVaradhanLoss(BaseVariationalBoundLoss):
    """
    Donsker-Varadhan variational lower bound for
    the Kullback-Leibler divergence.

    References
    ----------
    .. [1] Donsker, M. and Varadhan, S. "Asymptotic evaluation of certain
           markov process expectations for large time".
           Communications on Pure and Applied Mathematics 36(2):183?212, 1983
    """

    is_lower_bound = True
    
    class DVLossSecondTerm(torch.autograd.Function):
        """
        Implementation of the second term of DonskerVaradhanLoss, unbiased gradient.
        """
    
        # Denominator regularization.
        EPS = 1e-6

        @staticmethod
        def forward(context, T_q: torch.tensor, moving_average: float) -> torch.tensor:
            """
            Forward pass.
            
            Parameters
            ----------
            T_q : torch.tensor
                Critic network value on a batch from the second distribution.
            moving_average : float
                 Current moving average of expectation of exp(T),
                 which is used for back propagation.
            """

            logmeanexp_T_q = torch.logsumexp(T_q, dim=0) - math.log(T_q.shape[0])
            
            # Needed for the gradient computation.
            # moving_average is not required for the forward pass.
            context.save_for_backward(T_q, logmeanexp_T_q.detach() if moving_average is None else moving_average)
            
            return logmeanexp_T_q
    
        @staticmethod
        def backward(context, grad_output: torch.tensor) -> torch.tensor:
            """
            Backward pass.
            
            Parameters
            ----------
            grad_output : torch.tensor
                Output value gradient.
            """
    
            T_q, moving_average = context.saved_tensors
            grad = grad_output * T_q.exp().detach() / (moving_average * T_q.shape[0] + DonskerVaradhanLoss.DVLossSecondTerm.EPS)
            
            return grad, None

    def __init__(
        self,
        biased: bool=False,
        ema_multiplier: float=1.0e-2,
    ):
        """
        Create an instance of `DonskerVaradhanLoss` class.

        Parameters
        ----------
        biased : bool, optional
            Do not use exponential moving average estimate of the
            partition function. Default: False.
        ema_multiplier : float, optional
            Exponential moving average multiplier. Default: 0.01.
        """
        
        super().__init__(ema_multiplier)

        if not isinstance(biased, bool):
            raise TypeError("Parameter `biased' has to be boolean")

        self.biased = biased
        
        self.ema_meanexp_T_q = None

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

        mean_T_p = torch.mean(T_p)

        if self.biased:
            logmeanexp_T_q = torch.logsumexp(T_q, dim=0) - math.log(T_q.shape[0])
        else:
            logmeanexp_T_q = DonskerVaradhanLoss.DVLossSecondTerm.apply(T_q, self.ema_meanexp_T_q)
            with torch.no_grad():
                self.update_exponential_moving_average("ema_meanexp_T_q", torch.exp(logmeanexp_T_q.detach()))

        return -mean_T_p + logmeanexp_T_q