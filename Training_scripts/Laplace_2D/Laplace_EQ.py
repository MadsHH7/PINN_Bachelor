from modulus.sym.eq.pde import PDE
from sympy import Symbol, Function


class LaplaceEquation(PDE):
    """
    Laplace Equation

    Parameters
    ==========    
    rho : float describing density of the fluid

    dim : int describing dimension    

    """
    name = "LaplaceEquation"

    def __init__(self, rho=1, dim=2):
        # Set params
        self.dim = dim

        # Coordinates
        x, y = Symbol("x"), Symbol("y")

        # Make input variables
        input_variables = {"x": x, "y": y}
        if self.dim == 1:
            input_variables.pop("y")

        # Velocity components
        phi = Function("phi")(*input_variables)
        u = phi.diff(x)
        v = phi.diff(y)
        ux = u.diff(x)      
        vy = v.diff(y)
        
        # Set equations
        self.equations = {}

        self.equations["continuity"] = (
            ux + vy
        )
        self.equations["normal_x"] = (
            u
        )
        self.equations["normal_y"] = (
            v
        )
        self.equations["normal_circle_outer"] = (
            -(u * (-x/3.0) + v * (-y/3.0))
        )
        self.equations["normal_circle_inner"] = (
            (u * (-x) + v * (-y)) 
        )