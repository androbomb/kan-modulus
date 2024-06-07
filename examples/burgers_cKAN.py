import os
import warnings

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

from sympy import Symbol, Eq, Abs, And, Or, Xor, Function, Number
from sympy import atan2, pi, sin, cos

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.layers import Activation
#from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.eq.pde import PDE

# Import KANArch
import sys
sys.path.append('../src/')
from kan import KANArch
from chebyshev_kan import cKANArch
from jacobi_kan    import jKANArch

class Burgers(PDE):
    """
    Burgers equation 1D
        u_tt + u u_x - D^2 u_xx = 0

    Parameters
    ==========
    D : float, string
        Diffusion coefficient. If a string then the
        Diffusion  is input into the equation.
    """

    name = "Burgers"

    def __init__(self, u: str = 'u', D: float = 0.5):
        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # make u function
        u = Function("u")(*input_variables)

        # wave speed coefficient
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # set equations
        self.equations = {}
        self.equations["burgers"] = u.diff(t, 1) + u*u.diff(x,1) - (D**2 * u.diff(x)).diff(x)
        
def get_model(is_jacobi: bool = False):
    if is_jacobi:
        flow_net = jKANArch(
            input_keys = [Key("x"), Key("t")],
            output_keys= [Key("u")],
            # KAN Arch Specs
            layers_hidden = [2,2], 
            degree = 8,
            a_fact = 1.0, 
            b_fact = 1.0,
        )
        return flow_net
        
    flow_net = cKANArch(
        input_keys = [Key("x"), Key("t")],
        output_keys= [Key("u")],
        # KAN Arch Specs
        layers_hidden = [2,2], 
        degree = 8
    )
    
    return flow_net

@modulus.sym.main(config_path="conf", config_name="config_burgers")
def run(cfg: ModulusConfig) -> None:
    # MACRO PARAMS
    _ell = 1.0
    _diffusion_coefficient = np.sqrt( 0.01/float(np.pi) )
    _t_f = 1.0
    # ====== PDE ===========================
    # make list of nodes to unroll graph on
    pde = Burgers(u="u",  D =_diffusion_coefficient)

    # ====== MODEL ===========================
    flow_net = get_model()
    # make nodes
    nodes = pde.make_nodes() + [flow_net.make_node(name="flow_network")]
    # ====== Geometry ===========================
    # vars
    x, t_symbol = Symbol("x"), Symbol("t")
    time_range = {t_symbol: (0, _t_f)}
    # geo
    geo_1D = Line1D(point_1 = -_ell, point_2 = +_ell)
    # ====== Domain ===========================
    # make diamond domain
    domain = Domain()   # <====== DOMAIN instance =======
    # Interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_1D,
        outvar={"burgers": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "burgers": Symbol("sdf"),
        },
        parameterization = time_range,
    )
    domain.add_constraint(interior, "interior")
    # BC
    BC = PointwiseBoundaryConstraint(
        nodes    = nodes,
        geometry = geo_1D,
        outvar   = {"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization = time_range,
    )
    domain.add_constraint(BC, "BC")
    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_1D,
        outvar={"u": - sin(pi*x)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")    
      
    # ====== inferencer ===========================
    # add inferencer data
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(-_ell, +_ell, deltaX)
    t = np.arange(0, _t_f, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    invar_numpy  = {"x": X, "t": T}
    
    grid_inference = PointwiseInferencer(
        nodes = nodes,
        invar = invar_numpy,
        output_names = ["u"],
        batch_size = 1024,
        plotter    = InferencerPlotter(),
    )
    domain.add_inferencer(grid_inference, "inf_data")
    
    # ====== Solver ===========================
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
