# Part of the Chebysjev Implementation came from
# https://github.com/SpaceLearner/JacobiKAN/blob/main/ChebyKANLayer.py
# Inspiration from https://arxiv.org/pdf/2406.02917

import torch
import torch.nn as nn

# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    def __init__(
        self, 
        input_dim : int, 
        output_dim: int, 
        degree : int = 4,
        a : float = 1.0, 
        b : float = 1.0
    ):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim   = output_dim
        self.a        = a
        self.b        = b
        self.degree   = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0: ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a-self.b) + (self.a+self.b+2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k  = (2*i+self.a+self.b)*(2*i+self.a+self.b-1) / (2*i*(i+self.a+self.b))
            theta_k1 = (2*i+self.a+self.b-1)*(self.a*self.a-self.b*self.b) / (2*i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            theta_k2 = (i+self.a-1)*(i+self.b-1)*(2*i+self.a+self.b) / (i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class jKANArchCore(nn.Module):
    def __init__(
        self,
        # KAN data,
        layers_hidden : list  = [2,2],
        degree : int = 4,
        a : float = 1.0, 
        b : float = 1.0
    ):
        super().__init__()
        
        self.degree = degree
        
        self.layers = nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                JacobiKANLayer(
                    input_dim  = in_features,
                    output_dim = out_features,
                    degree = self.degree,
                    a = a, 
                    b = b
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

class jKANArch(Arch):
    def __init__(
        self,
        # Modulus data
        input_keys : List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        # KAN data,
        layers_hidden : list  = [2,2],
        degree : int = 4,
        a_fact : float = 1.0, 
        b_fact : float = 1.0
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
        
        self._impl = jKANArchCore(
            layers_hidden = self.layers_hidden ,
            degree = degree,
            a = a_fact,
            b = b_fact
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
