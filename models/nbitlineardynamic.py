'''

Applies N-bit min-max quantization to both activations and weights for dynamic PTQ

@vla, 06/12/2024

'''

from torch import nn, Tensor
import torch.nn.functional as F

# Observers compute quantization parameters (scaling factor and zero-point)
# TODO: experiment with different observers
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver


class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding

    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale




def quant(x: Tensor, num_bits, obs):
    
    # Q_low = -2 ** (num_bits - 1)
    # Q_high = 2 ** (num_bits - 1) - 1
    
    # pass input to observer for metric computing
    obs(x)
    
    # computed quantization parameters
    s,z = obs.calculate_qparams()
    
    # quantize 
    result = ((x / s) + z).round()
    
    # --> dequantize
    result = (result - z) * s
    
    return result
    

class NBitLinearDynamic(nn.Linear):
    """
    Custom linear layer with N-bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """
    
    def __init__(self,
                 *kargs,
                 weight_bits=8,
                 activation_bits=8,
                 **kwargs
    ):
    
        super(NBitLinearDynamic, self).__init__(*kargs, **kwargs)
        self.weight_bits     = weight_bits
        self.activation_bits = activation_bits
        
        Q_low = -2 ** (self.weight_bits - 1)
        Q_high = 2 ** (self.weight_bits - 1) - 1
        self.weight_observer = MinMaxObserver(quant_min=Q_low, quant_max=Q_high)
        
        Q_low = -2 ** (self.activation_bits - 1)
        Q_high = 2 ** (self.activation_bits - 1) - 1
        self.activation_observer = MovingAverageMinMaxObserver(quant_min=Q_low, quant_max=Q_high, is_dynamic=True, averaging_constant=1)
        
        
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the NBitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)

        # STE (Straight-through estimator) trick using detach, not really necessary for just PTQ inference
        x_quant = x_norm + (quant(x_norm, self.activation_bits, self.activation_observer) - x_norm).detach()
        w_quant = w + (quant(w, self.weight_bits, self.weight_observer) - w).detach()
        y = F.linear(x_quant, w_quant)
        
        return y
