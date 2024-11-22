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
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor

from modulus.sym.utils.io import csv_to_dict, ValidatorPlotter


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equations

    ze = ZeroEquation(nu = 0.00002, dim = 3, time = False, max_distance = Max(Symbol("sdf")))
    ns = NavierStokes(nu = ze.equations["nu"], rho = 500, dim = 3, time = False)

    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )

    # Create nodes
    nodes = ze.make_nodes() + ns.make_nodes() + [flow_net.make_node(name = "flow_network")]

    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    # Make domain
    Pipe_domain = Domain()
    
    # test other geometry 
    bend_angle_range=(1.323541349, 1.323541349)
    radius_pipe_range=(0.1, 0.1) 
    radius_bend_range=(0.2, 0.2)
    inlet_pipe_length_range=(0.2, 0.2) # 0.2 m rigtige tal
    outlet_pipe_length_range=(1.0, 1.0) # 1.0 m rigtige tal
    
    in_vel = 0.1

    theta = bend_angle_range[1]
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
    
    data_path = f"/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data"
    # data_path = f"/home/madshh7/PINN_Bachelor/Data"
    key = "pt1"

    input, output, nr_points = get_data(
        df_path= os.path.join(data_path, f"U0{key}_Laminar_train.csv"),
        desired_input_keys=["x", "y", "z"],
        original_input_keys=["X (m)", "Y (m)", "Z (m)"],
        desired_output_keys=["u", "v", "w", "p"],
        original_output_keys=["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"],
    )
    
    flow_data = np.full((nr_points, 1), fill_value=100.0)
    
    flow = PointwiseConstraint.from_numpy(
        nodes = nodes,
        invar = input,
        outvar = output,
        batch_size = nr_points,
        lambda_weighting={
            "u": flow_data,
            "v": flow_data,
            "w": flow_data,
            "p": flow_data,
        }
    )
    Pipe_domain.add_constraint(flow, "flow_data")
    
    ## Add validator
    # Find the validation data
    val_df = os.path.join(data_path, f"U0{key}_Laminar_validation.csv")
    mapping = {"Velocity[i] (m/s)": "u", "Velocity[j] (m/s)": "v", "Velocity[k] (m/s)": "w", "X (m)": "x", "Y (m)": "y", "Z (m)": "z"}
    val_var = csv_to_dict(to_absolute_path(val_df), mapping)
    
    val_invar_numpy = {
        key: value for key, value in val_var.items() if key in ["x", "y", "z"]
    }
    val_outvar_numpy = {
        key: value for key, value in val_var.items() if key in ["u", "v", "w"]
    }
    
    validator = PointwiseValidator(
        nodes=nodes,
        invar=val_invar_numpy,
        true_outvar=val_outvar_numpy,
        batch_size=1024,
        plotter=None    
    )
    Pipe_domain.add_validator(validator)
    
    # Lastly add inferencer
    n_pts = int(5e4)
    inference_pts = Pipe.geometry.sample_interior(nr_points=n_pts)
    
    xs = inference_pts["x"]
    ys = inference_pts["y"]
    zs = inference_pts["z"]
    
    inf = PointwiseInferencer(
        nodes = nodes,
        invar = {"x": xs, "y": ys, "z": zs},
        output_names = {"u", "v", "w", "p"},
        batch_size = n_pts
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