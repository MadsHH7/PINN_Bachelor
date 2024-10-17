import numpy as np
from sympy import Symbol, Function, Number, Eq, Abs, cos, pi

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig, to_absolute_path
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE

pi = float(pi)

class LaplaceEquation(PDE):
    """
    Laplace Equation

    Parameters
    ==========
    p : float describing pressure [Pa]
    
    rho : float describing density of the fluid

    c : float describing the constant pressure we wish to apply

    dim : int describing dimension    
    
    name = "LaplaceEquation"
    """
    
    def __init__(self, rho=1, c=1.5, dim=2):
        # Set params
        self.dim = dim

        # Coordinates
        x, y = Symbol("x"), Symbol("y")

        # Make input variables
        input_variables = {"x": x, "y": y}
        if self.dim == 1:
            input_variables.pop("y")

        # Velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        
        # Pressure component
        p = Function("p")(*input_variables)
        
        # Set equations
        self.equations = {}
        self.equations["continuity"] = (
            u.diff(x, 1) + v.diff(y, 1)
        )
        self.equations["irrotational"] = (
            v.diff(x, 1) - u.diff(y, 1)
        )
        self.equations["bernoulli"] = (
            ((u**2 + v**2)**(0.5) / 2) + p/rho - c
        )


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Make equation
    lp = LaplaceEquation(dim=2)

    # Create network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],    
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = lp.make_nodes() + [flow_net.make_node(name="flow_network")]

    # Make geometry
    height = 1.0
    width = 0.5
    x, y = Symbol("x"), Symbol("y")
    
    # Define inlet pipe
    rec1 = Rectangle((0.0, 0.0), (0.5, 1.0))
    
    # Make domain
    Rect_domain = Domain()

    # Define our conditions for the inlet pipe
    # No slip condition
    no_slip_right = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= Eq(x, width), 
    )
    Rect_domain.add_constraint(no_slip_right, "no_slip_right")
    
    no_slip_left = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= Eq(x, 0.0),
    )
    Rect_domain.add_constraint(no_slip_left, "no_slip_left")

    # Inlet
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0.0, "v": 1.0},
        batch_size=cfg.batch_size.Inlet,
        criteria= Eq(y, 0.0),
    )
    Rect_domain.add_constraint(Inlet, "inlet")
    
    # Outlet
    Outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"p": 0},
        batch_size=cfg.batch_size.Inlet,
        criteria= Eq(y, 1.0),
    )
    Rect_domain.add_constraint(Outlet, "outlet")
    
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"continuity": 0, "irrotational": 0, "bernoulli": 0}, 
        batch_size=cfg.batch_size.Interior,
        # lambda_weighting={
        #     "continuity": Symbol("sdf"),
        #     "irrotational": Symbol("sdf"), 
        #     "bernoulli": Symbol("sdf")
        # },
    )
    Rect_domain.add_constraint(interior, "interior")
    
    # Make solver
    slv = Solver(cfg, Rect_domain)

    # Start solver
    slv.solve()

if __name__ == "__main__":
    run()