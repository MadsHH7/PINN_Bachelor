from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.utils.io.vtk import var_to_polyvtk

n_points = 5000

# Define the rectangle parameters
bottom_left_corner = (-1.0, -.5)  # Bottom left corner of the rectangle
top_right_corner = (1.0, .5)      # Top right corner of the rectangle

# Create the rectangle
rectangle = Rectangle(
    bottom_left_corner,
    top_right_corner
)

# R = rectangle.sample_boundary(nr_points=n_points)
R = rectangle.sample_interior(nr_points=n_points)

var_to_polyvtk(R, "Mads_leg/Rectangle/test_interior")