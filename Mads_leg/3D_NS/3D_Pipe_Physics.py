from sympy import Eq, And, Symbol, sqrt, cos, sin, pi, Max
import os

import numpy as np
import torch

import modulus.sym
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)

from modulus.sym.key import Key

from PINN_Helper import get_data

from pipe_bend_parameterized_geometry import *
import numpy as np

from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor

from modulus.sym.utils.io import csv_to_dict


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equations

    # ze = ZeroEquation(nu = 0.00002, dim = 3, time = False, max_distance = Max(Symbol("sdf")))
    # ns = NavierStokes(nu = ze.equations["nu"], rho = 500, dim = 3, time = False)
    ns = NavierStokes(nu=0.002, rho=25, dim=3, time=False)

    # Setup things for the integral continuity condition
    normal_dot_vel = NormalDotVec()

    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )

    # Create nodes
    nodes = ns.make_nodes() + normal_dot_vel.make_nodes() + [flow_net.make_node(name = "flow_network")]

    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    # Make domain
    Pipe_domain = Domain()
    
    # Setup the geometry
    bend_angle_range=(1.323541349, 1.323541349)
    radius_pipe_range=(0.1, 0.1)        # 0.1 m rigtige tal
    radius_bend_range=(0.2, 0.2)        # 0.2 m rigtige tal
    inlet_pipe_length_range=(0.2, 0.2)  # 0.2 m rigtige tal
    outlet_pipe_length_range=(1.0, 1.0) # 1.0 m rigtige tal    
    
    in_vel = 0.1

    radius = radius_pipe_range[1]

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
        outvar = {"u": 0.0, "v": in_vel, "w": 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = (x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 <= radius**2,
    )
    Pipe_domain.add_constraint(Inlet, "Inlet")
    
    # Outlet
    
    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.outlet_pipe,
        outvar = {"p": 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = (x - Pipe.outlet_center[0])**2 + (y - Pipe.outlet_center[1])**2 + z**2 <= radius**2,
    )
    Pipe_domain.add_constraint(Outlet, "Outlet")

    # Boundary    
    
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.geometry,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.NoSlip,
        criteria = And(
            ((x - Pipe.outlet_center[0])**2 + (y - Pipe.outlet_center[1])**2 + z**2 > radius**2),
            ((x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 > radius**2),
        )
    )
    Pipe_domain.add_constraint(Walls, "Walls")
    
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
    Pipe_domain.add_constraint(Interior, "Interior")
    
    # Integral constraint
    Volumetric_flow = pi * radius**2 * in_vel
    all_planes = Pipe.inlet_pipe_planes + Pipe.bend_planes + Pipe.outlet_pipe_planes
    for i, plane in enumerate(all_planes):
        integral = IntegralBoundaryConstraint(
            nodes = nodes,
            geometry = plane,
            outvar = {"normal_dot_vel": Volumetric_flow},
            batch_size = 1,
            integral_batch_size = 250,
        )
        Pipe_domain.add_constraint(integral, f"Integral{i}")
    
    # # Lastly add inferencer
    
    data_path = f"/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data"
    # data_path = f"/home/madshh7/PINN_Bachelor/Data"
    key = "pt1"
    
    val_df = os.path.join(data_path, f"U0{key}_Laminar_validation.csv")
    mapping = {"Velocity[i] (m/s)": "u", "Velocity[j] (m/s)": "v", "Velocity[k] (m/s)": "w", "X (m)": "x", "Y (m)": "y", "Z (m)": "z"}
    val_var = csv_to_dict(to_absolute_path(val_df), mapping)
    
    val_invar_numpy = {
        key: value for key, value in val_var.items() if key in ["x", "y", "z"]
    }
    

    inf = PointwiseInferencer(
        nodes = nodes,
        invar = val_invar_numpy,
        output_names = {"u", "v", "w", "p"},
        batch_size = 1024,
    )
    Pipe_domain.add_inferencer(inf, "vtk_inf")


    # Make solver
    slv = Solver(cfg, Pipe_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat
    # "Velocity[i] (m/s)" * iHat + "Velocity[j] (m/s)" * jHat + "Velocity[k] (m/s)" * kHat

    # DTU HPC interactive
    # sxm2sh