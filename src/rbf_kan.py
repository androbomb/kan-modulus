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

from rbf_layer import RBFKANLayer

class RBFKANArchCore(nn.Module):
    def __init__(
        self,
        # data,
        layers_hidden   : list  = [2,2],
        num_centers     : int = 100,
        radial_function : str = 'gaussian',
        p_norm          : float = 2.0 ,
        normalization   : bool = True,
        centers_scale   : float = 1.0 ,
        # 
        add_layernorm : bool = False
    ):
        super().__init__()

        self.num_centers   = num_centers
        self.centers_scale = centers_scale
        self.p_norm = p_norm
        self.normalization = normalization
        self.radial_function = radial_function
        
        self.layers = nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                RBFKANLayer(
                    input_dim  = in_features,
                    output_dim = out_features,
                    #
                    num_centers   = num_centers,
                    radial_function = radial_function ,
                    centers_scale = centers_scale,
                    normalization = normalization,
                    p_norm = p_norm,
                )
            )
            if add_layernorm:
                self.layers.append(
                    nn.LayerNorm(out_features) # To avoid gradient vanishing caused by tanh
                )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


    def get_weight_list(self):
        weights = [param for param in self.parameters()]
        biases = []
        
        return weights, biases

class RBFKANArch(Arch):
    def __init__(
        self,
        # Modulus data
        input_keys : List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        # KAN data,
        layers_hidden   : list  = [2,2],
        num_centers     : int = 100,
        radial_function : str = 'gaussian',
        p_norm          : float = 2.0 ,
        normalization   : bool = True,
        centers_scale   : float = 1.0 ,
        # 
        add_layernorm : bool = False ,
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
        
        self._impl = RBFKANArchCore(
            layers_hidden = self.layers_hidden ,
            num_centers   = num_centers,
            radial_function = radial_function ,
            centers_scale = centers_scale,
            normalization = normalization,
            p_norm = p_norm,
            add_layernorm = add_layernorm,
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
