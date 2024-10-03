import numpy as np
from sympy import Symbol, Function, Number, Eq

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
    no_slip = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        outvar={"u": 0, "v": 1.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(x, width / 2),
    )
    Rect_domain.add_constraint(no_slip, "no_slip")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"laplace_equation": 0},
        batch_size=cfg.batch_size.Interior,
    )
    Rect_domain.add_constraint(interior, "interior")

    # add validator
    file_path = "openfoam/cavity_uniformVel0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] += -width / 2  # center OpenFoam data
        openfoam_var["y"] += -height / 2  # center OpenFoam data
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
        Rect_domain.add_validator(openfoam_validator)

        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            output_names=["u", "v", "p"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        Rect_domain.add_inferencer(grid_inference, "inf_data")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )

    # Make solver
    slv = Solver(cfg, Rect_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()