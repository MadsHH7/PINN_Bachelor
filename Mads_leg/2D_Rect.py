from modulus.sym.geometry.primitives_2d import Rectangle, Line, Polygon
from modulus.sym.geometry.parameterization import Parameterization, Parameter
# from modulus.geometry import Parameterization, Paramter
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from geo_test import Geometry
from sympy import Symbol, pi, cos, sin, Eq, And
pi = float(pi)

from modulus.sym.utils.io.vtk import var_to_polyvtk

n_points = 5000

# # Define the rectangle parameters
# bottom_left_corner = (0.0, 0.0)  # Bottom left corner of the rectangle
# top_right_corner = (0.5, 1.0)      # Top right corner of the rectangle

# center = (0.5, 1.0)     # Center of the circle
# radius = 0.5            # Radius of the circle

# # Create the rectangle
# rectangle = Rectangle(
#     bottom_left_corner,
#     top_right_corner
# )

# # Create the rectangle
# rectangle2 = Rectangle(
#     (0.5, 1.0),
#     (1.5, 1.5)
# )

# circle = Circle(
#     center,
#     radius
# )
# R = rectangle.sample_boundary(nr_points=n_points)
# R2 = rectangle2.sample_boundary(nr_points=n_points)

# cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))

# circ = circle & cut_rect

# Cb = circ.sample_boundary(nr_points=n_points)
# Ci = circ.sample_interior(nr_points=n_points)

# var_to_polyvtk(R, "Rectangle/test_rect_in")
# var_to_polyvtk(Cb, "Rectangle/test_boundary")
# var_to_polyvtk(R2, "Rectangle/test_rect_out")
# var_to_polyvtk(Ci, "Rectangle/test_interior")

# Make geometry
height = 1.0
width = 0.5

# Define inlet pipe
rec1 = Rectangle((0.0, 0.0), (0.5, 1.0))

# Define our outlet pipe, swap height and width
rec2 = Rectangle((0.5, 1.0), (1.5, 1.5))

# Define and cut our circle
center = (0.5, 1.0)     # Center of the circle
radius = 0.5            # Radius of the circle

circ1 = Circle(center = (0.5, 1.0), radius = 0.5)
cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))    # Define our cut rectangle to be the point where our other pipe ends, and the upper right hand corner we wish to end at

rec_inner = Rectangle((0.5, 0.875), (0.625, 1.0))
cut_circ = Circle(center = (0.625, 0.875), radius = 0.125)

bend = rec_inner - cut_circ 
circle = circ1

# Add all the geometries
Pipe = rec1 + rec2 + circle# + bend

x, y = Symbol("x"), Symbol("y")
Pb = circ1.sample_boundary(nr_points=n_points)
Pi = circ1.sample_interior(nr_points=n_points)

var_to_polyvtk(Pb, "Rectangle/test_boundary")
# var_to_polyvtk(Pi, "Rectangle/test_interior")