from sympy import Symbol, Number, Function
from modulus.sym.eq.pde import PDE
class Laplace2D(PDE):


    name = "Laplace2D"
    def __init__(self, C= "C", rho = 1):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x,"y": y}

        # make u function
        # phi = Function("phi")(*input_variables)
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        p = Function("p")(*input_variables)

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            u.diff(x,1)+v.diff(y,1) 
        )
        self.equations["irrotational"] = (
            v.diff(x,1)-u.diff(y,1)
        )

        self.equations["Bernoulli"] = (
           ( (u**2+v**2)**(0.5)  / 2 ) + p/rho-C
        )

    # The version which uses phi
        # self.equations = {}
        # self.equations["continuity"] = (
        #     phi.diff(x,2)+phi.diff(y,2) 
        # )
        # self.equations["irrotational"] = (
        #     phi.diff(y,1).diff(x,1)-phi.diff(x,1).diff(y,1)
        # )

        # self.equations["Bernoulli"] = (
        #     sqrt(phi.diff(x,1)**2+phi.diff(y,1)**2)/2+p/rho-C
        # )



