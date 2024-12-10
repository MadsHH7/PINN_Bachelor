import os
import torch
import numpy as np
import modulus.sym
from sympy import Eq, And, Symbol, sqrt, cos, sin, pi, Max

from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.key import Key

from modulus.sym.geometry.primitives_2d import Polygon

from modulus.sym.hydra import (
    instantiate_arch, 
    ModulusConfig, 
    to_absolute_path
)

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)


from PINN_Helper import get_data

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io import csv_to_dict, ValidatorPlotter


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equations
    ns = NavierStokes(nu = 1.85508e-5, rho = 1.18415, dim = 2, time = False)
    # mu er 1.85508e-5 (Dynamic viscosity). Unit: pa*s = Newton*s/m^2 = kg/(m*s)
    # rho: 1.18415 # Density. Unit: kg/m^3
    
    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y")],
        output_keys = [Key("u"), Key("v"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )

    # Create nodes
    nodes = ns.make_nodes() + [flow_net.make_node(name = "flow_network")]

    # Make geometry
    x, y = Symbol("x"), Symbol("y")
    
    # Make domain
    Pipe_domain = Domain()

    ### GEOMETRY ###
    point0 = (213.12714072, 403.12714072)
    point1 = (613.12714072, 403.12714072)
    point2 = (613.12714072, 243.13183868)
    point3 = (447.83179646, -34.70465573)
    point4 = (169.99530204, -200.00)
    point5 = (0.00, -200.00)
    point6 = (0.00, 200.00)
    point7 = (60.00469796, 200.00)
    point8 = (156.00999255, 257.11714818)
    point9 = (213.12714072, 353.12244276)
 
    geo = Polygon(
        points=[
            point0,
            point1,
            point2,
            point3,
            point4,
            point5,
            point6,
            point7,
            point8,
            point9,
        ]
    )
 
    scaling = 1 / 1000
 
    geo = geo.scale(scaling)
    
    # Make constraints
    in_vel = 0.01
    
    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar = {"u": 0.0, "v": in_vel},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(x, 0.0),
    )
    Pipe_domain.add_constraint(Inlet, "Inlet")
    
    # Outlet
    y_max = 403.12714072 * scaling
    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar = {"p": 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(y, y_max),
    )
    Pipe_domain.add_constraint(Outlet, "Outlet")

    # Boundary    
    
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,
        outvar = {"u": 0.0, "v": 0.0},
        batch_size = cfg.batch_size.NoSlip,
        criteria = And(
            x > 0.0,
            y < y_max,
        )
    )
    Pipe_domain.add_constraint(Walls, "Walls")
    
    # Interior
    Interior = PointwiseInteriorConstraint(
        nodes = nodes, 
        geometry = geo,
        outvar = {"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0},
        batch_size = cfg.batch_size.Interior,
    )
    Pipe_domain.add_constraint(Interior, "Interior")
    
    # data_path = f"/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data/2D/bend_data_mvel001.csv/"
    data_path = f"/home/madshh7/PINN_Bachelor/Data/2D/bend_data_mvel001.csv/"
    if os.path.exists(to_absolute_path(data_path)):
        input, output, nr_points = get_data(
            df_path= to_absolute_path(data_path),
            desired_input_keys=["x", "y"],
            original_input_keys=["X (m)", "Y (m)"],
            desired_output_keys=["u", "v", "p"],
            original_output_keys=["Velocity[i] (m/s)", "Velocity[j] (m/s)"],
        )
        # flow_data = np.full((nr_points, 1), fill_value=100.0)
        
        flow = PointwiseConstraint.from_numpy(
            nodes = nodes,
            invar = input,
            outvar = output,
            batch_size = nr_points,
        )
        Pipe_domain.add_constraint(flow, "flow_data")
        
        ## Add validator
        # Find the validation data
        val_df = data_path
        mapping = {"Pressure (Pa)": "p", "X (m)": "x", "Y (m)": "y", "Velocity[j] (m/s)": "v", "Velocity[i] (m/s)": "u"}
        val_var = csv_to_dict(to_absolute_path(val_df), mapping)
        
        val_invar_numpy = {
            key: value for key, value in val_var.items() if key in ["x", "y"]
        }
        val_outvar_numpy = {
            key: value for key, value in val_var.items() if key in ["u", "v", "p"]
        }
        
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
    # Paraview
    # u*iHat + v*jHat
    # "Velocity[i] (m/s)" * iHat + "Velocity[j] (m/s)" * jHat

    # DTU HPC interactive
    # sxm2sh