# Part of the Chebysjev Implementation came from
# https://github.com/SpaceLearner/JacobiKAN/blob/main/ChebyKANLayer.py
# Inspiration from https://arxiv.org/pdf/2406.02917

import torch
import torch.nn as nn

# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(
        self, 
        input_dim : int, 
        output_dim: int, 
        degree: int = 4
    ):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class cKANArchCore(nn.Module):
    def __init__(
        self,
        # KAN data,
        layers_hidden : list  = [2,2],
        degree: int = 4
    ):
        super().__init__()
        
        self.degree = degree
        
        self.layers = nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLayer(
                    input_dim  = in_features,
                    output_dim = out_features,
                    degree = self.degree
                ),
                nn.LayerNorm(out_features) # To avoid gradient vanishing caused by tanh
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def get_weight_list(self):
        weights = [param for param in self.parameters()]
        biases = []
        
        return weights, biases

class cKANArch(Arch):
    def __init__(
        self,
        # Modulus data
        input_keys : List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        # KAN data,
        layers_hidden : list  = [2,2],
        degree: int = 4
    ):
        super().__init__(
            input_keys  = input_keys,
            output_keys = output_keys,
            detach_keys = detach_keys,
            periodicity = periodicity,
        )

        if self.periodicity is not None:
            in_features = sum(
                [
                    x.size
                    for x in self.input_keys
                    if x.name not in list(periodicity.keys())
                ]
            ) + sum(
                [
                    2 * x.size
                    for x in self.input_keys
                    if x.name in list(periodicity.keys())
                ]
            )
        else:
            in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        # ===== KAN PART =============
        self.layers_hidden = [
            in_features,
            *layers_hidden,
            out_features
        ]
        
        self._impl = cKANArchCore(
            layers_hidden = self.layers_hidden ,
            degree = degree
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity = self.periodicity,
            input_dict  = self.input_key_dict,
            dim = -1,
        )
        #x = self._impl(x.float())
        x = self._impl(x.to(torch.float32))
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict = self.detach_key_dict,
            dim = -1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)
