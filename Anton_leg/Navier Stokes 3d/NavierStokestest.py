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

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    ns = NavierStokes(nu = 0.01, rho = 500.0, dim = 3, time = False)

    # normal_dot_vel = NormalDotVec(["u", "v","w"])
    vel = Symbol("vel")
    parameters ={"vel":(5,30)}
    pr = Parameterization(parameters)

    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes()+normal_dot_vel.make_nodes() + [flow_net.make_node(name = "flow_network")]
    
    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    bend_angle_range = (1.323541349,1.323541349)
    radius_pipe_range = (0.1,0.1)
    radius_bend_range = (0.2,0.2)
    inlet_pipe_length_range = (0.2,0.2)
    outlet_pipe_length_range = (1,1)

    bend_angle = bend_angle_range[0]
    radius = radius_pipe_range[0]
    inlet_length = inlet_pipe_length_range[0]
    outlet_pipe_length = outlet_pipe_length_range[0]

    Pipe = PipeBend(bend_angle_range= bend_angle_range,
                    radius_pipe_range=radius_pipe_range,
                    radius_bend_range=radius_bend_range,
                    inlet_pipe_length_range=inlet_pipe_length_range,
                    outlet_pipe_length_range=outlet_pipe_length_range)
        # Make domain
    Pipe_domain = Domain()
    Pipe.bend_planes_centers[-1]
    # Make outlet Geometry

    # geom_outlet = Pipe.outlet_pipe & Pipe.outlet_pipe_planes[-1]

    # Make constraints

    # Boundary

    # PointwiseConstrain
    # fromnumpy
    # Er en turbulence model, og den er meget simpel, og derfor god at bruge.
    # Lokal turbulence, men ikke noget global turbulence.
    # Han har ikke brugt ZeroEquation(),men syntes at det ville være en god ide

    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.inlet_pipe,
        outvar = {"u": 0.0, "v": 1.0, "w": 0.0},
        batch_size= cfg.batch_size.Inlet,
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
        criteria=(x - Pipe.inlet_center[0]) ** 2 + (y - Pipe.inlet_center[1]) ** 2 + z**2 > radius**2,   
    )

    Pipe_domain.add_constraint(Interior,"Interior")

    ## Attempt 2 at integral conditions

    normal_dot_vel = NormalDotVec()

    for i,plane in enumerate(Pipe.inlet_pipe_planes):
        # mass_flow_rate = vel * radius # Unit is m^2/s
        integral_continuity = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=plane,
            outvar={"normal_dot_vel": 0.1},
            batch_size=1,
            integral_batch_size=100,
            # lambda_weighting={"normal_dot_vel": cfg.custom.continuity_weight},
            parameterization=pr,
            )
        Pipe_domain.add_constraint(integral_continuity, f"integral_plane_{i}")

 # Add integral planes to help the network learn the flow.
    # The planes tell the network how much fluid is moved through each plane on average.

    # centers = []

    # for elem in (Pipe.inlet_pipe_planes_centers):
    #     centers.append(elem)

    # for elem in Pipe.bend_planes_centers:
    #     centers.append(elem)
    
    # for elem in Pipe.outlet_pipe_planes_center:
    #     centers.append(elem)

    # planes = []
    # for elem in Pipe.inlet_pipe_planes:
    #     planes.append(elem)
    # for elem in Pipe.bend_planes:
    #     planes.append(elem)
    # for elem in Pipe.outlet_pipe_planes:
    #     planes.append(elem)
    
    # print(centers[0])
    # lengths = []
    # lengths.append(sqrt( (Pipe.inlet_center[0]-centers[0][0])**2
    #                     + (Pipe.inlet_center[1]-centers[0][1])**2) )
    # for i in range(len(planes)-1):
    #     lengths.append(sqrt( (centers[i][0]-centers[i+1][0])**2
    #                          + (centers[i][1]-centers[i+1][1])**2))
    
    # # lengths.append(sqrt( (Pipe.outlet_center[0]-centers[-1][0])**2
    #                     # + (Pipe.outlet_center[1]-centers[-1][1])**2) )

    # print("Length og lengths: ", len(lengths))
    # print("Length of planes: ", len(planes))

    # for i, (plane, length) in enumerate(zip(planes, lengths)):
    #     mass_flow_rate = vel * length # Unit is m^2/s
    #     print( )
    #     print(mass_flow_rate)
    #     print()
    #     print(plane)

    #     integral_continuity = IntegralBoundaryConstraint(
    #         nodes=nodes,
    #         geometry=plane,
    #         outvar={"normal_dot_vel": mass_flow_rate},
    #         batch_size=1,
    #         integral_batch_size=cfg.batch_size.IntegralContinuity,
    #         # lambda_weighting={"normal_dot_vel": cfg.custom.continuity_weight},
    #         parameterization=pr,
    #     )
    #     Pipe_domain.add_constraint(integral_continuity, f"integral_plane_{i}")



    data_path = f"/zhome/e3/5/167986/Desktop/PINN_Bachelor/Data"
    key = "pt1"
    angle = (pi / 2) + bend_angle
    rot_matrix = (
        [float(cos(angle)), float(-sin(angle)), 0],
        [float(sin(angle)), float(cos(angle)), 0],
        [0, 0, 1]
    )

    translate= ([
        0,
        inlet_pipe_length_range[-1],
        0
    ])

    input, output, nr_points = get_data(
        df_path= os.path.join(data_path, f"U0{key}_Laminar.csv"),
        desired_input_keys=["x", "y", "z"],
        original_input_keys=["X (m)", "Y (m)", "Z (m)"],
        desired_output_keys=["u", "v", "w", "p"],
        original_output_keys=["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"],
        rotation_matrix= rot_matrix,
        translation=translate
    )
    
    # flow_data = np.full((nr_points, 1))
    
    flow = PointwiseConstraint.from_numpy(
        nodes = nodes,
        invar = input,
        outvar = output,
        batch_size = nr_points,
    )
    Pipe_domain.add_constraint(flow, "flow_data")




    nr_points=int(1e5)
    
    
    inference_pts = Pipe.geometry.sample_interior(nr_points=nr_points)
    
    xs = inference_pts["x"]
    ys = inference_pts["y"]
    zs = inference_pts["z"]

    inference = PointwiseInferencer(
            nodes=nodes,
            invar={"x": xs, "y": ys, "z": zs},
            output_names=["u", "v", "p"],
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