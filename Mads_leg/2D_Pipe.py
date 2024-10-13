import numpy as np
from sympy import Symbol, Eq, Abs, And, sin, cos, pi
import math

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

pi = float(pi)

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
    height = 1.0
    width = 0.5
    x, y = Symbol("x"), Symbol("y")
    
    # Define inlet pipe
    rec1 = Rectangle((0.0, 0.0), (0.5, 1.0))
    
    # Define our outlet pipe, swap height and width
    rec2 = Rectangle((0.5, 1.0), (1.5, 1.5))
    
    # Define and cut our circle
    center = (0.5, 1.0)     # Center of the circle
    radius = 0.5            # Radius of the circle
    circ1 = Circle(center = center, radius = radius)
    cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))    # Define our cut rectangle to be the point where our other pipe ends, and the upper right hand corner we wish to end at

    circle = circ1 & cut_rect

    # Make the bend on the inside corner
    rec_inner = Rectangle((0.5, 0.875), (0.625, 1.0))
    cut_circ = Circle(center = (0.625, 0.875), radius = 0.125)

    bend = rec_inner - cut_circ

    # Add all the geometries
    Pipe = rec1 + circle + rec2 #+ bend

    # Make domain
    Pipe_domain = Domain()

    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0.0, "v": 1.0},
        batch_size=cfg.batch_size.Inlet,
        lambda_weighting={"u": 1.0, "v": 1.0 - cos(2*x*pi)**2},  # weight edges to be zero
        criteria= Eq(y, 0.0),
    )
    Pipe_domain.add_constraint(Inlet, "inlet")
    
    # Define no penetration in inlet pipe
    Inlet_pipe_left = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(x, 0.0),
    )
    Pipe_domain.add_constraint(Inlet_pipe_left, "IP_left")
    
    Inlet_pipe_right = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= Eq(x, 0.5) # y: (0, 0.875)
    )
    Pipe_domain.add_constraint(Inlet_pipe_right, "IP_right")

    # Define the boundaries for our bend
    Inner_bend = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar= {"normal_circle": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= And((x<=0.5), (y >= 1.0))
    )
    Pipe_domain.add_constraint(Inner_bend, "inner_bend")
    
    # Define the boundary for the outlet pipe
    Outlet_pipe_upper = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= rec2,
        outvar= {"v": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(y, 1.5)
    )
    Pipe_domain.add_constraint(Outlet_pipe_upper, "OP_upper")
    
    Outlet_pipe_lower = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= rec2,
        outvar= {"v": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= Eq(y, 1.0)
    )
    Pipe_domain.add_constraint(Outlet_pipe_lower, "OP_lower")
    
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"continuity": 0, "irrotational": 0, "bernoulli": 0}, 
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "irrotational": Symbol("sdf"), 
            "bernoulli": Symbol("sdf")
        },
    )
    Pipe_domain.add_constraint(interior, "interior")
    
    # Make solver
    slv = Solver(cfg, Pipe_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()
    # for paraview
    # u*iHat + v*jHat