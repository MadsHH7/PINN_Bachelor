from modulus.sym.geometry.primitives_2d import Rectangle, Circle, Line
from modulus.sym.geometry.parameterization import Parameterization, Parameter
# from modulus.geometry import Parameterization, Paramter
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from sympy import Symbol, pi, cos, sin, Eq, And
pi = float(pi)

from modulus.sym.utils.io.vtk import var_to_polyvtk

n_points = 5000

# Step 1: Define the Circle
circle = Circle(center=(0.0, 0.0), radius=1.0)

# Step 2: Define the Line (for upper half-circle y ≥ 0)
line = Line(
    normal=(0, -1),   # Normal vector pointing downwards
    point_1=(0.0, 0.0),  # The line passes through the origin
    point_2=(0.0, 1.0)
)

# Step 3: Create the Half-Circle
half_circle = circle & line

var_to_polyvtk(half_circle, "Rectangle/Half_Circ")