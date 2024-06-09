'''

Applies N-bit abs-max quantization to both activations and weights

@vla, 06/08/2024

'''

from torch import nn, Tensor
import torch.nn.functional as F

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

# def activation_quant(x: Tensor):
#     """Per token quantization to 8bits. No grouping is needed for quantization

#     Args:
#         x (Tensor): _description_

#     Returns:
#         _type_: _description_
#     """
#     scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
#     y = (x * scale).round().clamp_(-128, 127) / scale
#     return y

def quant(x:Tensor, num_bits=8):
    '''
        Per token quantization to num_bits precision
    '''
    # dtype = x.dtype
    # x = x.float()
    Q_low = -2 ** (num_bits - 1)
    Q_high = 2 ** (num_bits - 1) - 1
    s = Q_high / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Q_low, Q_high) / s
    return result
    # return result.type(dtype)   


# def weight_quant(w: Tensor):
#     scale = w.abs().mean()
#     e = w.mean()
#     u = (w - e).sign() * scale
#     return u


class NBitLinear(nn.Linear):
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
    
        super(NBitLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits     = weight_bits
        self.activation_bits = activation_bits

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

        # STE (Straight-through estimator) trick using detach
        x_quant = x_norm + (quant(x_norm, self.activation_bits) - x_norm).detach()
        w_quant = w + (quant(w, self.weight_bits) - w).detach()
        y = F.linear(x_quant, w_quant)
        
        return y
