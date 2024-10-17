from sympy import Eq, And, Symbol, sqrt

import modulus.sym

from modulus.sym.geometry.primitives_3d import Cylinder
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.key import Key

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
    
    center = (0, 0, 0)
    radius = 1
    height = 10

    Pipe = Cylinder(center, radius, height)
    
    # Make domain
    Cylinder_domain = Domain()
    
    # Make constraints
    
    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar = {"u": 0.0, ("v"): 0.0, ("w"): 1.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(z, -5.0),
        lambda_weighting = {"u": 1.0, "v": 1.0, "w": 1.0 - sqrt(x**2 + y**2)},
    )
    Cylinder_domain.add_constraint(Inlet, "Inlet")
    
    # Outlet
    Outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar = {"p": 0.0},
        batch_size = cfg.batch_size.Inlet,
        criteria = Eq(z, 5.0)
    )
    Cylinder_domain.add_constraint(Outlet, "Outlet")
    
    # Boundary
    Walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar = {"u": 0.0, "v": 0.0, "w": 0.0},
        batch_size = cfg.batch_size.NoSlip,
        # criteria = And(z < 5.0, z > -5.0)
        criteria = z <= 5.0
    )
    Cylinder_domain.add_constraint(Walls, "Walls")
    
    # Interior
    Interior = PointwiseInteriorConstraint(
        nodes = nodes, 
        geometry = Pipe,
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
    
    # Make solver
    slv = Solver(cfg, Cylinder_domain)
    
    # Start solver
    slv.solve()
    
    
if __name__ == "__main__":
    run()
    # Paraview
    # u*iHat + v*jHat + w*kHat