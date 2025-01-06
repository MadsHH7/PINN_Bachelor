import numpy as np
import os
from sympy import Symbol, Function, Number, Eq, Abs, cos, pi, And, Or

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
from Laplace_EQ import LaplaceEquation

from modulus.sym.utils.io import ValidatorPlotter, csv_to_dict
from modulus.sym.domain.validator import PointwiseValidator


pi = float(pi)

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    lp = LaplaceEquation(dim=2)

    # Create network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],    
        output_keys=[Key("phi")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = lp.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Make geometry 
    x, y = Symbol("x"), Symbol("y")
    
    # Define and cut our circle
    center = (0.0, 0.0)     # Center of the circle
    radius = 3.0            # Radius of the circle
    circ1 = Circle(center = center, radius = radius)
    cut_rect = Rectangle((0.0, 0.0), (3.0, 3.0))    

    circle = circ1 & cut_rect

    # Make the bend on the inside corner
    rec_inner = Rectangle((0.0, 0.0), (1.0, 1.0))
    cut_circ = Circle(center = (0.0, 0.0), radius = 1.0)

    bend = rec_inner - (rec_inner - cut_circ)

    # Add all the geometries
    Pipe = circle - bend

    # Make domain
    Pipe_domain = Domain()

    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"phi__y": 1.0},
        batch_size=cfg.batch_size.Inlet,
        criteria= Eq(y, 0.0),
    )
    Pipe_domain.add_constraint(Inlet, "inlet")
    
    # Outlet
    Outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"phi__x": -1.0},
        batch_size=cfg.batch_size.Outlet,
        criteria= Eq(x, 0.0),
    )
    Pipe_domain.add_constraint(Outlet, "outlet")
    
    # Define the boundaries for our bend
    Outer_bend = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar= {"normal_circle_outer": 0},
        batch_size=cfg.batch_size.Wall,
        criteria= Or(And((x > 0.0), (y > 1.0)), And(x > 1.0, y > 0.0)),
    )
    Pipe_domain.add_constraint(Outer_bend, "outer_bend")

    # Define the boundaries for our bend
    Inner_bend = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar= {"normal_circle_inner": 0},
        batch_size=cfg.batch_size.Wall,
        criteria= And((x < 1.0), (y < 1.0)),
    )
    Pipe_domain.add_constraint(Inner_bend, "inner_bend")
    
    # Define the interior points
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"continuity": 0}, 
        batch_size=cfg.batch_size.Interior,
    )
    Pipe_domain.add_constraint(interior, "interior")
    
    # Add data path
    data_path = f"/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data/2D/"
    # data_path = f"/home/madshh7/PINN_Bachelor/Data/2D"
    
    ## Add validator
    # Find the validation data
    val_df = os.path.join(data_path, "LaplaceBend2D.csv")
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
    Pipe_domain.add_validator(validator)
    
    # Make solver
    slv = Solver(cfg, Pipe_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()
    # for paraview
    # u*iHat + v*jHat