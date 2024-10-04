import numpy as np
from sympy import Symbol, Function, Number, Eq, Abs

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
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

import os
import warnings
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer

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

    def __init__(self, dim=2):
        # Set params
        self.dim = dim

        # Coordinates
        x, y = Symbol("x"), Symbol("y")

        # Make input variables
        input_variables = {"x": x, "y": y}
        if self.dim == 1:
            input_variables.pop("y")

        # Velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)

        # Set equations
        self.equations = {}
        self.equations["continuity"] = (
            u.diff(x) + v.diff(y)
        )
        self.equations["irrotational"] = (
            v.diff(x) - u.diff(y)
        )


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    lp = LaplaceEquation(dim=2)

    # Create network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],    
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = lp.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Make geometry
    height = 1
    width = 0.5
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # Make domain
    Rect_domain = Domain()

    # No slip condition
    no_slip1 = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(x, width / 2),
    )
    Rect_domain.add_constraint(no_slip1, "no_slip1")
    
    no_slip2 = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(x, -width / 2),
    )
    Rect_domain.add_constraint(no_slip2, "no_slip2")

    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0.0, "v": 1.0},
        batch_size=cfg.batch_size.Inlet,
        # lambda_weighting={"u": 1.0 - 2 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(y, -height/2),
    )
    Rect_domain.add_constraint(Inlet, "inlet")
    
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "irrotational": 0}, 
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "irrotational": Symbol("sdf"),
        },
    )
    Rect_domain.add_constraint(interior, "interior")

    # Make solver
    slv = Solver(cfg, Rect_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()