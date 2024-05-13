from typing import Optional, Dict, Tuple, Union, List
from modulus.sym.key import Key

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from modulus.sym.models.layers import Activation, get_activation_fn
#from modulus.sym.models.activation import Activation, get_activation_fn
from modulus.sym.models.arch import Arch

class KANLinear(torch.nn.Module):
    """
    Efficient KAN Linear Layer, from 
        An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN).
        Blealtan Cao (@Blealtan)
        https://github.com/Blealtan/efficient-kan/tree/master
        
    For modulus, we need to rewrite the KAN linear layer to be capable of working without the Batching dimension.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size    : int = 5,
        spline_order : int = 3,
        scale_noise  : float = 0.1,
        scale_base   : float = 1.0,
        scale_spline : float = 1.0,
        enable_standalone_scale_spline : bool = True,
        base_activation = torch.nn.SiLU,
        grid_eps : float  = 0.02,
        grid_range : list = [-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        # == torch parameters =======
        self.base_weight   = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        # == additional args =======
        self.scale_noise  = scale_noise
        self.scale_base   = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(-1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        is_modulus = False
        # Added for Modulus
        # if there is no batch. mimic batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_modulus = True

        assert x.dim() == 2 and x.size(-1) == self.in_features, f"Error in KANLinear;\nx.dim() == 2 and x.size(1) == self.in_features\ngot\nx.dim() = {x.dim()}, x.size(1): {x.size(-1)} ; self.in_features: {self.in_features}"
        
            
        base_output   = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_output + spline_output

        if is_modulus:
            out = out.squeeze(0)
        return out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(-1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KANArchCore(nn.Module):
    def __init__(
        self,
        # KAN data,
        layers_hidden : list  = [2,2],
        grid_size     : int   = 5,
        spline_order  : int   = 3,
        scale_noise   : float = 0.1,
        scale_base    : float = 1.0,
        scale_spline  : float = 1.0,
        grid_eps      : float = 0.02,
        grid_range    : list  = [-1, 1],
        base_activation=torch.nn.SiLU,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
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

class KANArch(Arch):
    def __init__(
        self,
        # Modulus data
        input_keys : List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        # KAN data,
        layers_hidden : list  = [2,2],
        grid_size     : int   = 5,
        spline_order  : int   = 3,
        scale_noise   : float = 0.1,
        scale_base    : float = 1.0,
        scale_spline  : float = 1.0,
        grid_eps      : float = 0.02,
        grid_range    : list  = [-1, 1],
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
        
        self._impl = KANArchCore(
            layers_hidden = self.layers_hidden ,
            grid_size = grid_size    ,
            spline_order = spline_order ,
            scale_noise  = scale_noise ,
            scale_base   = scale_base ,
            scale_spline = scale_spline ,
            grid_eps   = grid_eps,
            grid_range = grid_range,
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
