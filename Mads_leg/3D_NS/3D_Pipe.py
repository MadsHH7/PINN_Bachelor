from sympy import Eq, And, Symbol, sqrt, cos, sin, pi
import os

import modulus.sym
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint
)

from modulus.sym.key import Key

from PINN_Helper import get_data

from pipe_bend_parameterized_geometry import *
import numpy as np


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    ns = NavierStokes(nu = 0.00002, rho = 1.0, dim = 3, time = False)
    
    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name = "flow_network")]
    
    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    # Make domain
    Cylinder_domain = Domain()
    
    # test other geometry 
    bend_angle_range=(1.323541349, 1.323541349)
    radius_pipe_range=(0.1, 0.1) 
    radius_bend_range=(0.1, 0.1)
    inlet_pipe_length_range=(0.2, 0.2) # 0.2 m rigtige tal
    outlet_pipe_length_range=(1.0, 1.0) # 1.0 m rigtige tal
    
    theta = bend_angle_range[1]
    radius = radius_bend_range[1]
    outlet_pipe_length = outlet_pipe_length_range[1]

    Pipe = PipeBend(bend_angle_range, 
                    radius_pipe_range, 
                    radius_bend_range,
                    inlet_pipe_length_range, 
                    outlet_pipe_length_range,
    )

    # Make constraints
    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.inlet_pipe,
        outvar = {"u": 0.0, ("v"): 0.1, ("w"): 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = (x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 <= radius**2
    )
    Cylinder_domain.add_constraint(Inlet, "Inlet")
    
    # Outlet
    direction = (outlet_pipe_length * cos(theta + pi / 2), outlet_pipe_length * sin(theta + pi / 2))
    
    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.outlet_pipe,
        outvar = {"p": 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = ((x - Pipe.outlet_center[0])**2 + (y - Pipe.outlet_center[1])**2 + z**2 <= radius**2)
    )
    Cylinder_domain.add_constraint(Outlet, "Outlet")

    # Boundary    
    
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.geometry,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.NoSlip,
        criteria = And(((x - Pipe.outlet_center[0])**2 + (y - Pipe.outlet_center[1])**2 + z**2 > radius**2),
                       ((x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 > radius**2),
        )
    )
    Cylinder_domain.add_constraint(Walls, "Walls")
    
    # Interior
    Interior = PointwiseInteriorConstraint(
        nodes = nodes, 
        geometry = Pipe.geometry,
        outvar = {"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0, "momentum_z": 0.0},
        batch_size = cfg.batch_size.Interior,
        lambda_weighting = {
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        }
    )
    Cylinder_domain.add_constraint(Interior, "Interior")
    
    # Integral constraint
    # all_planes = Pipe.inlet_pipe_planes + Pipe.bend + Pipe.bend_planes + Pipe.outlet_pipe_planes + Pipe.outlet

    # integral = IntegralBoundaryConstraint(
    #     nodes = nodes,
    #     geometry = Pipe.geometry,
    #     outvar = {"u": 0.0, "v": 0.1, "w": 0.0},
    #     batch_size = 11,
    #     integral_batch_size = 100,
    # )
    # Cylinder_domain.add_constraint(integral, "Integral")
    
    input, output, df = get_data(
        df_path= os.path.join()
    )
    
    # Make solver
    slv = Solver(cfg, Cylinder_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat

    # DTU HPC interactive
    # sxm2sh