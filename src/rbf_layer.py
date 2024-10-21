from typing import Optional, Dict, Tuple, Union, List, Callable
from modulus.sym.key import Key

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from modulus.sym.models.layers import Activation, get_activation_fn
#from modulus.sym.models.activation import Activation, get_activation_fn
from modulus.sym.models.arch import Arch


class RBFLayer(nn.Module):
    """
    Defines a Radial Basis Function Layer

    An RBF is defined by 5 elements:
        1. A radial kernel phi
        2. A positive shape parameter epsilon
        3. The number of kernels N, and their relative
           centers c_i, i=1, ..., N
        4. A norm ||.||
        5. A set of weights w_i, i=1, ..., N

    The output of an RBF is given by
    y(x) = sum_{i=1}^N a_i * phi(eps_i * ||x - c_i||)

    For more information check [1,2]

    [1] https://en.wikipedia.org/wiki/Radial_basis_function
    [2] https://en.wikipedia.org/wiki/Radial_basis_function_network

    Parameters
    ----------
        in_features_dim: int
            Dimensionality of the input features
        num_centers: int
            Number of centers to use
        out_features_dim: int
            Dimensionality of the output features
        radial_function: Callable[[torch.Tensor], torch.Tensor]
            A radial basis function that returns a tensor of real values
            given a tensor of real values
        p_norm: float
            Order of p-norm.
        normalization: bool, optional
            if True applies the normalization trick to the rbf layer
    """

    def __init__(
        self,
        input_dim  : int,
        output_dim : int,
        #
        num_centers     : int = 100,
        radial_function : str = 'gaussian',
        p_norm          : float = 2.0 ,
        normalization   : bool = True,
        centers_scale   : float = 1.0
    ):
        super(RBFLayer, self).__init__()

        self.in_features_dim  = input_dim
        self.num_centers      = num_centers
        self.out_features_dim = output_dim
        self.centers_scale = centers_scale
        self.p_norm = p_norm
        self.normalization = normalization

        self.list_of_radial_functions = [
            'gaussian',
            'multiquadric',
            'inverse_quadratic',
            'inverse_multiquadric'
        ]
        if radial_function not in self.list_of_radial_functions:
            raise Exception(f"Inserted function {radial_function} is not implemented")

        if radial_function   == 'gaussian':
            self.radial_function = RBFLayer.rbf_gaussian
        elif radial_function == 'multiquadric':
            self.radial_function = RBFLayer.rbf_multiquadric
        elif radial_function == 'inverse_quadratic':
            self.radial_function = RBFLayer.rbf_inverse_quadratic
        elif radial_function == 'inverse_multiquadric':
            self.radial_function = RBFLayer.rbf_inverse_multiquadric

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        self.weights = nn.Parameter(
            torch.zeros(
                self.out_features_dim,
                self.num_centers,
                dtype=torch.float32
            )
        )

        # Initialize kernels' centers
        self.kernels_centers = nn.Parameter(
            torch.zeros(
                self.num_centers,
                self.in_features_dim,
                dtype=torch.float32
            )
        )

        # Initialize shape parameter
        self.log_shapes = nn.Parameter(
                torch.zeros(self.num_centers, dtype=torch.float32)
        )

        self.reset(upper_bound_kernels = self.centers_scale)

    def reset(
        self,
        upper_bound_kernels: float = 1.0,
        std_shapes  : float = 0.1,
        gain_weights: float = 1.0
    ) -> None:
        """
        Resets all the parameters.

        Parameters
        ----------
            upper_bound_kernels: float, optional
                Randomly samples the centers of the kernels from a uniform
                distribution U(-x, x) where x = upper_bound_kernels
            std_shapes: float, optional
                Randomly samples the log-shape parameters from a normal
                distribution with mean 0 and std std_shapes
            gain_weights: float, optional
                Randomly samples the weights used to linearly combine the
                output of the kernels from a xavier_uniform with gain
                equal to gain_weights
        """
        nn.init.uniform_(
            self.kernels_centers,
            a = - upper_bound_kernels,
            b = + upper_bound_kernels
        )

        nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the ouput of the RBF layer given an input vector, 
        allowing the trick for being used with Nvidia Modulus

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of size B x Fin, where B is the batch size,
                and Fin is the feature space dimensionality of the input

        Returns
        ----------
            out: torch.Tensor
                Output tensor of size B x Fout, where B is the batch
                size of the input, and Fout is the output feature space
                dimensionality
        """
        is_modulus = False
        # Added for Modulus
        # if there is no batch. mimic batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_modulus = True

        out = self._tensor_forward(x)
        
        if is_modulus:
            out = out.squeeze(0)
        return out

    def _tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard RBF forward

        It works this way:
            1. Compute distance betweem x and ξ (kernel centers)
                x : [B, d_in]
                ξ : [N_c, d_in]
                    -> r : [B, N_c]
            2. Compute φ( ε r ), where ε : [N_c] so that 
                φ( ε r ) : [B, N_c]
            3. Normalise (if requested)
            4. Compute out via multiplication of 
                w : [N_out, N_c] 
                φ( ε r ) : [B, N_c]
                  -> φ( ε r ) @ w.T
            
        """
        # 1.
        centers = self.kernels_centers
        r = RBFLayer.compute_distance(x, centers, p=self.p_norm)
        # 2. 
        log_shapes = self.log_shapes
        #kernel = log_shapes.unsqueeze(0).expand(r.shape[0], r.shape[1]) * r
        kernel = log_shapes * r
        phi =  self.radial_function(
            kernel
        )
        # 3. 
        if self.normalization:
            phi = phi / (1e-9 + phi.sum(dim=-1)).unsqueeze(-1)
        # 4. 
        out = self.weights @ phi.T
        return out.T
        
    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """ Returns the linear combination weights """
        return self.weights.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return torch.exp(self.log_shapes.detach() )

    # Radial basis functions
    @staticmethod
    def rbf_gaussian(x: torch.Tensor) -> torch.Tensor:
        return torch.exp( - x**2 )
    
    @staticmethod
    def rbf_linear(x: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def rbf_multiquadric(x: torch.Tensor) -> torch.Tensor:
        return (1 + x.pow(2)).sqrt()
    
    @staticmethod
    def rbf_inverse_quadratic(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + x.pow(2))
    
    @staticmethod
    def rbf_inverse_multiquadric(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + x.pow(2)).sqrt()

    # distance function
    @staticmethod
    def compute_distance(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
        """
        Static method to compute the Lp distance between two tensors of shapes
            x : [B, d]
            y : [N, d]

        returns the matrix of duistances as 
            r : [B, N]
        """
        # Batch size
        x_size = x.size(0)
        # Number of centers
        y_size = y.size(0)
        # in space dim
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim) = (B, 1, d)
        y = y.unsqueeze(0) # (1, y_size, dim) = (1, N, d)
        tiled_x = x.expand(x_size, y_size, dim) # (B, N, d)
        tiled_y = y.expand(x_size, y_size, dim) # (B, N, d)
        r = (tiled_x - tiled_y).abs().pow(p).sum(-1).pow(1/p) # (B, N)
        return r

##########################################################################################
#                       RBF KAN Layer                                                    #
##########################################################################################

class RBFKANLayer(nn.Module):
    """
    Defines a Radial Basis Function KAN Layer.

    Similarly to standard RBF Layer, it is defined by 5 elements:
        1. A radial kernel phi
        2. A positive shape parameter epsilon
        3. The number of kernels N, and their relative
           centers c_i, i=1, ..., N
        4. A norm ||.||
        5. A set of weights w_i, i=1, ..., N

    But the output of an RBFKAN is given by
        RBFKAN(x_{bi})_{bj} = sum_{i=1}^{d_in} \Phi_i  sum_{n=1}^{N_c} w_{jn} * phi(eps_{in} * |x_{bi} - c_{in}|)
    where i \in d_in (the input dim), n\in N_c (the number of centers), 
    is the batch idx,  w_{jn} are the RBF wights, \Phi_i the KAN weights, c_{in} the n-centers for each i input, eps_{in} are the shape params. 

    For more information check [1,2]

    [1] https://en.wikipedia.org/wiki/Radial_basis_function
    [2] https://en.wikipedia.org/wiki/Radial_basis_function_network
    [3] https://arxiv.org/abs/2405.06721
    [4] https://github.com/sidhu2690/RBF-KAN

    Parameters
    ----------
        in_features_dim: int
            Dimensionality of the input features
        num_centers: int
            Number of centers to use
        out_features_dim: int
            Dimensionality of the output features
        radial_function: Callable[[torch.Tensor], torch.Tensor]
            A radial basis function that returns a tensor of real values
            given a tensor of real values
        p_norm: float
            Order of p-norm.
        normalization: bool, optional
            if True applies the normalization trick to the rbf layer
    """

    def __init__(
        self,
        input_dim  : int,
        output_dim : int,
        #
        num_centers     : int = 100,
        radial_function : str = 'gaussian',
        p_norm          : float = 2.0 ,
        normalization   : bool = True,
        centers_scale   : float = 1.0
    ):
        super(RBFKANLayer, self).__init__()

        self.in_features_dim  = input_dim
        self.num_centers      = num_centers
        self.out_features_dim = output_dim
        self.centers_scale = centers_scale
        self.p_norm = p_norm
        self.normalization = normalization

        self.list_of_radial_functions = [
            'gaussian',
            'multiquadric',
            'inverse_quadratic',
            'inverse_multiquadric'
        ]
        if radial_function not in self.list_of_radial_functions:
            raise Exception(f"Inserted function {radial_function} is not implemented")

        if radial_function   == 'gaussian':
            self.radial_function = RBFLayer.rbf_gaussian
        elif radial_function == 'multiquadric':
            self.radial_function = RBFLayer.rbf_multiquadric
        elif radial_function == 'inverse_quadratic':
            self.radial_function = RBFLayer.rbf_inverse_quadratic
        elif radial_function == 'inverse_multiquadric':
            self.radial_function = RBFLayer.rbf_inverse_multiquadric

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialise KAN weights
        self.kan_weights = nn.Parameter(
            torch.zeros(
                self.in_features_dim,
                dtype=torch.float32
            )
        )
        # Initialize linear combination weights
        self.weights = nn.Parameter(
            torch.zeros(
                self.out_features_dim,
                self.num_centers,
                dtype=torch.float32
            )
        )

        # Initialize kernels' centers
        self.kernels_centers = nn.Parameter(
            torch.zeros(
                self.num_centers,
                self.in_features_dim,
                dtype=torch.float32
            )
        )

        # Initialize shape parameter
        self.log_shapes = nn.Parameter(
                torch.zeros(self.num_centers, dtype=torch.float32)
        )

        self.reset(upper_bound_kernels = self.centers_scale)

    def reset(
        self,
        upper_bound_kernels: float = 1.0,
        std_shapes  : float = 0.1,
        gain_weights: float = 1.0
    ) -> None:
        """
        Resets all the parameters.

        Parameters
        ----------
            upper_bound_kernels: float, optional
                Randomly samples the centers of the kernels from a uniform
                distribution U(-x, x) where x = upper_bound_kernels
            std_shapes: float, optional
                Randomly samples the log-shape parameters from a normal
                distribution with mean 0 and std std_shapes
            gain_weights: float, optional
                Randomly samples the weights used to linearly combine the
                output of the kernels from a xavier_uniform with gain
                equal to gain_weights
        """
        nn.init.uniform_(
            self.kernels_centers,
            a = - upper_bound_kernels,
            b = + upper_bound_kernels
        )

        nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        nn.init.xavier_uniform_(self.weights, gain=gain_weights)

        nn.init.uniform_(self.kan_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the ouput of the RBF layer given an input vector, 
        allowing the trick for being used with Nvidia Modulus

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of size B x Fin, where B is the batch size,
                and Fin is the feature space dimensionality of the input

        Returns
        ----------
            out: torch.Tensor
                Output tensor of size B x Fout, where B is the batch
                size of the input, and Fout is the output feature space
                dimensionality
        """
        is_modulus = False
        # Added for Modulus
        # if there is no batch. mimic batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_modulus = True

        out = self._tensor_forward(x)
        
        if is_modulus:
            out = out.squeeze(0)
        return out

    def _tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard RBF forward

        It works this way:
            1. Compute distance betweem x and ξ (kernel centers)
                x : [B, d_in]
                ξ : [N_c, d_in]
                    -> r : [B, d_in, N_c] 
                Which is the main difference betweem standard RDF with RDFKAN.
            2. Compute φ( ε r ), where ε : [N_c, d_in] so that 
                φ( ε r ) : [B, d_in,  N_c]
            3. Normalise (if requested)
            4. Compute out N_c via multiplication of 
                w : [d_out, N_c] 
                φ( ε r ) : [B, d_in N_c]
                  -> φ( ε r ) @ w.T : [B, d_in, d_out]
            5. Compute out d_in via multiplication with Φ [d_in]
                Φ [d_in]
                (φ( ε r ) @ w.T : [B, d_in]
                  -> (φ( ε r ) @ w.T) @ Φ [B, d_out]
            
        """
        # 1.
        centers = self.kernels_centers
        r = self.compute_kan_distance(x, centers) # [B, d_in, N]
        # 2. 
        log_shapes = self.log_shapes # [N]
        kernel = log_shapes * r
        phi =  self.radial_function(
            kernel
        ) # [B, d_in, N]
        # 3. 
        if self.normalization:
            phi = phi / (1e-9 + phi.sum(dim=-1)).unsqueeze(-1) # [B, d_in, N]
        # 4. 
        out = phi @ self.weights.T  # [B, d_in, N] @ [d_out, N_c].T = [B, d_in, N] @ [N_c, d_out] = [B, d_in, d_out]
        out = out.moveaxis(-2, -1)  # [B, d_out, d_in]
        # 5
        out = out @ self.kan_weights # [B, d_out, d_in] @ [d_in] = [B. d_out]
        return out
        
    @property
    def get_kernels_centers(self):
        """ Returns the centers of the kernels """
        return self.kernels_centers.detach()

    @property
    def get_weights(self):
        """ Returns the linear combination weights """
        return self.weights.detach()

    @property
    def get_shapes(self):
        """ Returns the shape parameters """
        return torch.exp(self.log_shapes.detach() )

    # Radial basis functions
    @staticmethod
    def rbf_gaussian(x: torch.Tensor) -> torch.Tensor:
        return torch.exp( - x**2 )
    
    @staticmethod
    def rbf_linear(x: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def rbf_multiquadric(x: torch.Tensor) -> torch.Tensor:
        return (1 + x.pow(2)).sqrt()
    
    @staticmethod
    def rbf_inverse_quadratic(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + x.pow(2))
    
    @staticmethod
    def rbf_inverse_multiquadric(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + x.pow(2)).sqrt()

    # distance function
    @staticmethod
    def compute_kan_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Static method to compute the Lp distance between two tensors of shapes
            x : [B, d]
            y : [N, d]

        returns the matrix of distances as 
            r : [B, d, N]
        """
        # Batch size
        x_size = x.size(0)
        # Number of centers
        y_size = y.size(0)
        # in space dim
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim) = (B, 1, d)
        y = y.unsqueeze(0) # (1, y_size, dim) = (1, N, d)
        tiled_x = x.expand(x_size, y_size, dim) # (B, N, d)
        tiled_y = y.expand(x_size, y_size, dim) # (B, N, d)
        r = (tiled_x - tiled_y).abs() # (B, N, d)
        return r.moveaxis(-2, -1) # (B, d, N)
        