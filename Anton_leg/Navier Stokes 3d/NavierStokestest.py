from sympy import Eq, And, Symbol, sqrt

import modulus.sym

from modulus.sym.geometry.primitives_3d import Cone, Plane
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.turbulence_zero_eq import ZeroEquation

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)

from modulus.sym.key import Key

from pipe_bend_parameterized_geometry import PipeBend
from modulus.sym.geometry import Parameterization
from sympy import pi, cos, sin

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    ns = NavierStokes(nu = 0.01, rho = 1.0, dim = 3, time = False)
    
    # Create network
    flow_net = instantiate_arch(
        input_keys = [Key("x"), Key("y"), Key("z")],
        output_keys = [Key("u"), Key("v"), Key("w"), Key("p")],
        cfg = cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name = "flow_network")]
    
    # Make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    
    bend_angle_range = (1.323541349,1.323541349)
    radius_pipe_range = (1,1)
    radius_bend_range = (1,1)
    inlet_pipe_length_range = (5,5)
    outlet_pipe_length_range = (5,5)

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
    pr = Pipe.geometry.parameterization
    Pipe_domain = Domain()
    
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
        criteria=Eq(y,Pipe.inlet_center[1]),
    )

    Pipe_domain.add_constraint(Inlet,"Inlet")

    direction = (outlet_pipe_length * cos(bend_angle + pi / 2), outlet_pipe_length * sin(bend_angle + pi / 2))
    

    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.outlet_pipe,
        outvar = {"p": 0.0,},
        batch_size= cfg.batch_size.Inlet,
        criteria=Eq( direction[0] * (x-Pipe.outlet_center[0]) + direction[1] * (y-Pipe.outlet_center[1]),0 ),
    )

    Pipe_domain.add_constraint(Outlet,"Outlet")

    ## Boundary conditions
    epsilon = 10**(-4)
    scaler = 1-epsilon
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.geometry,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.Walls,
        criteria=And( direction[0] * (x-Pipe.outlet_center[0]*scaler) + direction[1] * (y-Pipe.outlet_center[1]*scaler) - epsilon < 0,
                     y > Pipe.inlet_center[1]),
    )

    Pipe_domain.add_constraint(Walls,"Walls")
    
    Interior = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry= Pipe.geometry,
        outvar={"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0, "momentum_z": 0.0,},
        batch_size= cfg.batch_size.Interior,
    )

    Pipe_domain.add_constraint(Interior,"Interior")

    # Make solver
    slv = Solver(cfg, Pipe_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat
