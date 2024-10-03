import numpy as np
from sympy import Symbol, Function, Number

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pde import PDE

class LaplaceEquation(PDE):
    """
    Laplace Equation

    Parameters
    ==========
    u : str

    dim :

    Examples
    ========
    >>> we = 
    >>> we.
    """
    name = "LaplaceEquation"

    def __init__(self, u="u", dim=3):
        # Set params
        self.u = u
        self.dim = dim

        # Coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # Make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # Scalar function
        assert type(u) == str
        u = Function(u)(*input_variables)

        # Set equations
        self.equations = {}
        self.equations["laplace_equation"] = (
            u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        )




@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    lp = LaplaceEquation(dim=2)

    # Create network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = lp.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Make geometry
    height = 1
    width = 0.5
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # Make domain
    Rect = Domain()

    # No slip condition
    no_slip = PointwiseBoundaryConstraint(
        nodes = nodes
    )