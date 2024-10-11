from modulus.sym.eq.pde import PDE
from sympy import Symbol, Function


class LaplaceEquation(PDE):
    """
    Laplace Equation

    Parameters
    ==========
    p : float describing pressure [Pa]
    
    rho : float describing density of the fluid

    c : float describing the constant pressure we wish to apply

    dim : int describing dimension    

    Examples
    ========
    >>> we = 
    >>> we.
    """
    name = "LaplaceEquation"

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
        self.equations["normal_circle"] = (
            ((u**2 + v**2)**(0.5) / 2) * x
        )
