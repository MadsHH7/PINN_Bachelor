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
from sympy import pi

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
    
    bend_angle = (pi/2,pi/2)
    radius_pipe = (1,1)
    radius_bend = (1,1)
    inlet_pipe_length = (5,5)
    outlet_pipe_length = (5,5)


    Pipe = PipeBend(bend_angle_range= bend_angle,
                    radius_pipe_range=radius_pipe,
                    radius_bend_range=radius_bend,
                    inlet_pipe_length_range=inlet_pipe_length,
                    outlet_pipe_length_range=outlet_pipe_length)
        # Make domain
    pr = Pipe.geometry.parameterization
    Pipe_domain = Domain()
    

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

    Pipe.outlet_center[2]
    Pipe_domain.add_constraint(Inlet,"Inlet")

    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.outlet_pipe,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size= cfg.batch_size.Inlet,
        criteria=Eq(y,Pipe.outlet_center[1]),
    )
   
    Pipe_domain.add_constraint(Outlet,"Outlet")

    ## Boundary conditions
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= Pipe.geometry,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.Walls,
    )

    Pipe_domain.add_constraint(Walls,"Walls")
    


    # Make solver
    slv = Solver(cfg, Pipe_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat
