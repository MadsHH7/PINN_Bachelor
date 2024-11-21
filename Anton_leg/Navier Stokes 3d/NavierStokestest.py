from sympy import Eq, And, Symbol, sqrt

import modulus.sym

from modulus.sym.geometry.primitives_3d import Cone, Plane
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.inferencer import PointwiseInferencer, PointVTKInferencer

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym import quantity

from modulus.sym.key import Key

from pipe_bend_parameterized_geometry import PipeBend
from modulus.sym.geometry import Parameterization
from sympy import pi, cos, sin
from PINN_Helper import get_data
import numpy as np
import os
import torch
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter)
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:


    inlet_vel_initial = quantity(0.1, "m/s")
    bend_angle_range_intial = (1.323541349,1.323541349)
    radius_pipe_range_inital = (quantity(0.1,"m"), quantity(0.1,"m"))
    radius_bend_range_inital = (quantity(0.2,"m"),quantity(0.2,"m"))
    inlet_pipe_length_range_initial = (quantity(0.2,"m"),quantity(0.2,"m"))
    outlet_pipe_length_range_intial = (quantity(1,"m"),quantity(1,"m"))



    length_scale = quantity(1, "m")
    velocity_scale = inlet_vel_initial
    nu = quantity(0.00002, "kg/(m*s)")
    rho = quantity(500, "kg/m^3")

    inlet_u = quantity(0,"m/s")
    inlet_v = quantity(0.1, "m/s")
    inlet_w = quantity(0,"m/s")
    density_scale = rho


    nd = NonDimensionalizer(
    length_scale=length_scale,
    mass_scale=density_scale * (length_scale**3),
    # time_scale= length_scale/velocity_scale,
    )

    # Scaled geometry
    inlet_vel_nd =  nd.ndim(inlet_u)
    bend_angle_nd = bend_angle_range_intial # I did not add any dimension to the bend
    radius_pipe_nd = tuple(map(lambda x: nd.ndim(x), radius_pipe_range_inital))
    radius_bend_nd = tuple(map(lambda x: nd.ndim(x), radius_bend_range_inital))
    inlet_pipe_length_nd = tuple(map(lambda x: nd.ndim(x), inlet_pipe_length_range_initial))
    outlet_pipe_length_nd = tuple(map(lambda x: nd.ndim(x), outlet_pipe_length_range_intial))

  


    # Make equation
    ze = ZeroEquation(nu = nd.ndim(nu), max_distance=0.1 , dim = 3, time = False)
    ns = NavierStokes(nu = ze.equations["nu"], rho = nd.ndim(rho), dim = 3, time = False)
    # ns = NavierStokes(nu = 0.00002, rho = 500.0, dim = 3, time = False)
    # max_distance kan bruge signed_distance fields i stedet for radius
    # Betyder at hvert punkt i dit rum får en afstand til geometriens overfflade. Fortæller om du er inde eller uden for geomtrien, og hvor langt fra den du er.
    # Kan bruge sympy til at definere max distance sim max sdf


   

    normal_dot_vel = NormalDotVec()

    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )
    
    # nodes = ns.make_nodes() + [flow_net.make_node(name = "flow_network")] + normal_dot_vel.make_nodes()
    nodes = (ns.make_nodes()
    +ze.make_nodes()
    + normal_dot_vel.make_nodes()
    + [flow_net.make_node(name = "flow_network")]
    + Scaler(["u","v","w","p"],
             ["u_scaled","v_scaled","w_scaled","p_scaled"],
             ["m/s","m/s","m/s","kg/(m*s^2)"],
             nd).make_node()
    )
    
    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    radius = radius_pipe_nd[0]
    

    Pipe = PipeBend(bend_angle_range= bend_angle_nd,
                    radius_pipe_range=radius_pipe_nd,
                    radius_bend_range=radius_bend_nd,
                    inlet_pipe_length_range=inlet_pipe_length_nd,
                    outlet_pipe_length_range=outlet_pipe_length_nd)
        # Make domain
    pr = Pipe.geometry.parameterization
    Pipe_domain = Domain()

    # Boundary

    # PointwiseConstrain
    # fromnumpy
    # Er en turbulence model, og den er meget simpel, og derfor god at bruge.
    # Lokal turbulence, men ikke noget global turbulence.
    # Han har ikke brugt ZeroEquation(),men syntes at det ville være en god ide

    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.inlet_pipe,
        outvar = {"u": 0.0, "v": inlet_vel_nd, "w": 0.0},
        batch_size= cfg.batch_size.Inlet,
        # criteria=Eq(y,Pipe.inlet_center[1]),
        criteria= (x - Pipe.inlet_center[0])**2 + (y - Pipe.inlet_center[1])**2 + z**2 <= radius**2
    )

    Pipe_domain.add_constraint(Inlet,"Inlet")

    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.outlet_pipe,
        outvar = {"p": 0.0,},
        batch_size= cfg.batch_size.Inlet,
        criteria= ((x - Pipe.outlet_center[0]) ** 2 + (y - Pipe.outlet_center[1]) ** 2 + z**2 <= radius**2),
    )

    Pipe_domain.add_constraint(Outlet,"Outlet")

    ## Boundary conditions
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.geometry,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.Walls,
        criteria=And(
            ((x - Pipe.inlet_center[0]) ** 2 + (y - Pipe.inlet_center[1]) ** 2 + z**2 > radius**2),
            ((x - Pipe.outlet_center[0]) ** 2 + (y - Pipe.outlet_center[1]) ** 2 + z**2 > radius**2),
    )
    )

    Pipe_domain.add_constraint(Walls,"Walls")
    
    Interior = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry= Pipe.geometry,
        outvar={"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0, "momentum_z": 0.0,},
        batch_size= cfg.batch_size.Interior,
        lambda_weighting = {
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        } 
    )

    Pipe_domain.add_constraint(Interior,"Interior")

    ## Attempt 2 at integral conditions
    # Integral constraints bruges mest hvis vi ikke har data.



    all_planes = Pipe.inlet_pipe_planes + Pipe.bend_planes + Pipe.outlet_pipe_planes
    mass_flow_rate = pi*radius**2 * inlet_vel_nd # Unit is m^2/s

    for i,plane in enumerate(all_planes):
        
        integral_continuity = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=plane,
            outvar={"normal_dot_vel": mass_flow_rate},
            batch_size=1,
            integral_batch_size=100,
            lambda_weighting= {"normal_dot_vel": 1},
        )
        # Pipe_domain.add_constraint(integral_continuity, f"integral_plane_{i}")

    # data_path = f"/zhome/e3/5/167986/Desktop/PINN_Bachelor/Data"
    # key = "pt1"
    # angle = (pi / 2) + bend_angle
    # rot_matrix = (
    #     [float(cos(angle)), float(-sin(angle)), 0],
    #     [float(sin(angle)), float(cos(angle)), 0],
    #     [0, 0, 1]
    # )

    # translate= ([
    #     0,
    #     inlet_pipe_length_range[-1],
    #     0
    # ])


    # input, output, nr_points = get_data(
    #     df_path= os.path.join(data_path, f"U0{key}_Laminar.csv"),
    #     desired_input_keys=["x", "y", "z"],
    #     original_input_keys=["X (m)", "Y (m)", "Z (m)"],
    #     desired_output_keys=["u", "v", "w", "p"],
    #     original_output_keys=["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"],
    #     rotation_matrix= rot_matrix,
    #     translation=translate
    # )
    
    # # flow_data = np.full((nr_points, 1))
    
    # flow = PointwiseConstraint.from_numpy(
    #     nodes = nodes,
    #     invar = input,
    #     outvar = output,
    #     batch_size = nr_points,
    # )
    # Pipe_domain.add_constraint(flow, "flow_data")




    nr_points=int(1e4)
    
    
    inference_pts = Pipe.geometry.sample_interior(nr_points=nr_points)
    
    xs = inference_pts["x"]
    ys = inference_pts["y"]
    zs = inference_pts["z"]

    inference = PointwiseInferencer(
            nodes=nodes,
            invar={"x": xs, "y": ys, "z": zs},
            output_names=["u", "v","w", "p"],
            batch_size=nr_points,
        )
    Pipe_domain.add_inferencer(inference, "Inference")

    slv = Solver(cfg, Pipe_domain)
    
    # Start solver
    slv.solve()
    

    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat

# Integral continuty planes
# Tilføj dem for at tvinge flow gennem røret
# laver tværsnit i røret som siger at integralet igennem snittet skal flow være det samme.
# Lav en constrint der integrer (summer for at gøre det diskret), det skal altid være det samme.

# class: Integralboundaryconstraint

# Husk at fjerne disconuitet ved inlet.
# Man kan ændre learning rate ved at stoppe trænningen og ændre navnet på optime_checkpoint
# Den vil køre videre med samme model, men bruge den nye learning rate.