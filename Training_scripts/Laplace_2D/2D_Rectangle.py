import numpy as np
import os

from sympy import Symbol, Function, Number, Eq, Abs, cos, pi

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE

from modulus.sym.utils.io import ValidatorPlotter, csv_to_dict
from modulus.sym.domain.validator import PointwiseValidator

from Laplace_EQ import LaplaceEquation

pi = float(pi)

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    lp = LaplaceEquation()

    # Create network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],    
        output_keys=["phi"],
        cfg=cfg.arch.fully_connected,
    )
    nodes = lp.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Make geometry
    height = 3.0
    width = 2.0
    x, y = Symbol("x"), Symbol("y")
    rec1 = Rectangle((-width/2, -height/2), (width/2, height/2))
    
    # Make domain
    Rect_domain = Domain()

    # Define our conditions for the inlet pipe
    # No penetration condition
    no_pen_right = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"normal_x": 0.0},
        batch_size=cfg.batch_size.Wall,
        criteria= Eq(x, width / 2), 
    )
    Rect_domain.add_constraint(no_pen_right, "no_pen_right")
    
    no_pen_left = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"normal_x": 0},
        batch_size=cfg.batch_size.Wall,
        criteria= Eq(x, -width / 2),
    )
    Rect_domain.add_constraint(no_pen_left, "no_pen_left")

    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"phi__y": 1.0},
        batch_size=cfg.batch_size.Inlet,
        criteria= Eq(y, -height/2),
    )
    Rect_domain.add_constraint(Inlet, "inlet")
    
    # Outlet
    Outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"phi__y": 1.0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(y, height/2),
    )
    Rect_domain.add_constraint(Outlet, "outlet")

    # Interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"continuity": 0},
        batch_size=cfg.batch_size.Interior,
    )
    Rect_domain.add_constraint(interior, "interior")
    
    # Add data path
    data_path = f"/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data/2D/"
    # data_path = f"/home/madshh7/PINN_Bachelor/Data/2D"
    
    
    ## Add validator
    # Find the validation data
    val_df = os.path.join(data_path, "LaplaceRect2D.csv")
    mapping = {"u": "phi__x", "v": "phi__y", "x": "x", "y": "y"}
    val_var = csv_to_dict(to_absolute_path(val_df), mapping)
    
    val_invar_numpy = {
        key: value for key, value in val_var.items() if key in ["x", "y"]
    }
    val_outvar_numpy = {
        key: value for key, value in val_var.items() if key in ["phi__x", "phi__y"]
    }
    
    # Construct the validator
    validator = PointwiseValidator(
        nodes=nodes,
        invar=val_invar_numpy,
        true_outvar=val_outvar_numpy,
        batch_size=1024,
        plotter=ValidatorPlotter(),    
    )
    Rect_domain.add_validator(validator)
    
    # Make solver
    slv = Solver(cfg, Rect_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()