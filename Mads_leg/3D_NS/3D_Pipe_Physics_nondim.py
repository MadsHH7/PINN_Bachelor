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

from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Physical quantities
    nu = quantity(0.00002, "m^2/s")
    rho = quantity(500, "kg/m^3")
    inlet_u = quantity(0.0, "m/s")
    inlet_v = quantity(0.1, "m/s")
    inlet_w = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    noslip_w = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_v
    density_scale = rho
    # length_scale = quantity(0.2, "m") # Diameter of pipe
    length_scale = quantity(1.0, "m")
    
    nd = NonDimensionalizer(
        length_scale=length_scale,
        mass_scale=density_scale * (length_scale**3),
        # time_scale=length_scale / velocity_scale
    )
    
    # Define the geometry
    bend_angle_range=(quantity(1.323541349, "rad"), quantity(1.323541349, "rad"))
    radius_pipe_range=(quantity(0.1, "m"), quantity(0.1, "m"))        # 0.1 m rigtige tal
    radius_bend_range=(quantity(0.2, "m"), quantity(0.2, "m"))        # 0.2 m rigtige tal
    inlet_pipe_length_range=(quantity(0.2, "m"), quantity(0.2, "m"))  # 0.2 m rigtige tal
    outlet_pipe_length_range=(quantity(1.0, "m"), quantity(1.0, "m")) # 1.0 m rigtige tal   
    
    # Non-dimensionalization
    bend_angle_range_nd=tuple(map(lambda x: nd.ndim(x), bend_angle_range))
    radius_pipe_range_nd=tuple(map(lambda x: nd.ndim(x), radius_pipe_range))
    radius_bend_range_nd=tuple(map(lambda x: nd.ndim(x), radius_bend_range))
    inlet_pipe_length_range_nd=tuple(map(lambda x: nd.ndim(x), inlet_pipe_length_range))
    outlet_pipe_length_range_nd=tuple(map(lambda x: nd.ndim(x), outlet_pipe_length_range))
    
    Pipe = PipeBend(
        bend_angle_range=bend_angle_range_nd,
        radius_bend_range=radius_bend_range_nd,
        radius_pipe_range=radius_pipe_range_nd,
        inlet_pipe_length_range=inlet_pipe_length_range_nd,
        outlet_pipe_length_range=outlet_pipe_length_range_nd,        
    )
    
    radius = radius_bend_range_nd[-1]
    
    # Make equations
    # ze = ZeroEquation(nu = nd.ndim(nu), dim = 3, time = False, max_distance = Max(Symbol("sdf")))
    # ns = NavierStokes(nu = ze.equations["nu"], rho = nd.ndim(rho), dim = 3, time = False)
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    
    # Setup things for the integral continuity condition
    normal_dot_vel = NormalDotVec()

    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )

    # Create nodes
    nodes = (
        ns.make_nodes() 
        # + ze.make_nodes()
        + normal_dot_vel.make_nodes() 
        + [flow_net.make_node(name = "flow_network")]
        + Scaler(
            ["u", "v", "w", "p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s", "m/s", "m^2/s^2"],
            nd
        ).make_node()
    )

    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    # Make domain
    Pipe_domain = Domain()
    
    # Make constraints
    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.inlet_pipe,
        outvar = {"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v), "w": nd.ndim(inlet_w)},
        batch_size = cfg.batch_size.Inlet,
        criteria = (x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 <= radius**2,
    )
    Pipe_domain.add_constraint(Inlet, "Inlet")
    
    # Outlet
    
    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.outlet_pipe,
        outvar = {"p": nd.ndim(outlet_p)},
        batch_size = cfg.batch_size.Inlet,
        criteria = (x - Pipe.outlet_center[0])**2 + (y - Pipe.outlet_center[1])**2 + z**2 <= radius**2,
    )
    Pipe_domain.add_constraint(Outlet, "Outlet")

    # Boundary    
    
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe.geometry,
        outvar = {"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v), "w": nd.ndim(noslip_w)},
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
    Volumetric_flow = pi * radius**2 * nd.ndim(inlet_v)
    all_planes = Pipe.inlet_pipe_planes + Pipe.bend_planes + Pipe.outlet_pipe_planes
    for i, plane in enumerate(all_planes):
        integral = IntegralBoundaryConstraint(
            nodes = nodes,
            geometry = plane,
            outvar = {"normal_dot_vel": Volumetric_flow},
            batch_size = 1,
            integral_batch_size = 100,
        )
        Pipe_domain.add_constraint(integral, f"Integral{i}")
    
    # Lastly add inferencer
    n_pts = int(5e4)
    inference_pts = Pipe.geometry.sample_interior(nr_points=n_pts)
    
    xs = inference_pts["x"]
    ys = inference_pts["y"]
    zs = inference_pts["z"]
    
    inf = PointwiseInferencer(
        nodes = nodes,
        invar = {"x": xs, "y": ys, "z": zs},
        output_names = {"u_scaled", "v_scaled", "w_scaled", "p_scaled"},
        batch_size = n_pts
    )
    Pipe_domain.add_inferencer(inf, "vtk_inf_scaled")

    inf2 = PointwiseInferencer(
        nodes = nodes,
        invar = {"x": xs, "y": ys, "z": zs},
        output_names = {"u", "v", "w", "p"},
        batch_size = n_pts,
    )
    Pipe_domain.add_inferencer(inf2, "vtk_inf")

    # Make solver
    slv = Solver(cfg, Pipe_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat
    # u_scaled*iHat+v_scaled*jHat+w_scaled*kHat
    # "Velocity[i] (m/s)" * iHat + "Velocity[j] (m/s)" * jHat + "Velocity[k] (m/s)" * kHat

    # DTU HPC interactive
    # sxm2sh