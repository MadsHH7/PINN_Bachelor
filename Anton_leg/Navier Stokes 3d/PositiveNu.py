from sympy import Symbol, Function, Number
from modulus.sym.eq.pde import PDE

class PositiveNu(PDE):
    # A equation the enforces postive nu values in the inverse problem.

    # Parameters
    # nu: The current value of nu


    name = "PositiveNu"

    def __init__(self,nu):

        self.equations = {}

        self.equations["positive_nu"] = min(nu,0)**2