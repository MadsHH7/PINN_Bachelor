from modulus.sym.eq.pde import PDE
from sympy import Symbol, Function


class LaplaceEquation(PDE):
    """
    Laplace Equation

    Parameters
    ==========
    p : float describing pressure [Pa]
    
    rho : float describing density of the fluid

    dim : int describing dimension    

    Examples
    ========
    >>> we = 
    >>> we.
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
        # u = Function("u")(*input_variables)
        # v = Function("v")(*input_variables)
        u = phi.diff(x)
        v = phi.diff(y)
        ux = u.diff(x)      
        # uy = u.diff(y)
        vy = v.diff(y)     
        # vx = v.diff(x)
        
        # Pressure component
        # p = Function("p")(*input_variables)
        
        # Set equations
        self.equations = {}

        self.equations["continuity"] = (
            ux + vy
        )
        # self.equations["irrotational"] = (
        #     vx - uy
        # )
        self.equations["normal_circle_outer"] = (
            (u * (-x/3.0) + v * (-y/3.0))
        )
        self.equations["normal_circle_inner"] = (
            -(u * (-x) + v * (-y)) 
        )