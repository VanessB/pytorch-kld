import torch


class BaseVariationalBoundLoss(torch.nn.Module):
    """
    Base class for Kullback-Leibler variational bound loss.
    """
    
    def __init__(self, ema_multiplier: float=1.0e-2):
        if not isinstance(ema_multiplier, float):
            raise TypeError("Parameter `ema_multiplier' has to be float")
            
        if not (0.0 <= ema_multiplier <= 1.0):
            raise ValueError("Parameter `ema_multiplier' has to be within the range [0; 1]")
        
        super().__init__()

        self.ema_multiplier = ema_multiplier


    def update_exponential_moving_average(self, attribute_name: str, value):
        """
        Update moving average.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to be updated.
        value
            Value to be used to update the attribute.
        """

        current_value = getattr(self, attribute_name)
        if current_value is None:
            setattr(self, attribute_name, value)
        else:
            setattr(self, attribute_name, (1.0 - self.ema_multiplier) * current_value + self.ema_multiplier * value)