from modulus.sym.geometry.primitives_2d import Rectangle, Circle, Line
from modulus.sym.geometry.primitives_3d import Cylinder
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.utils.io.vtk import var_to_polyvtk

n_points = 5000

# Define the rectangle parameters

# Make geometry
height = 1.0
width = 0.5
center = (0,0,0)

# # Define inlet pipe
# rec1 = Rectangle((0.0, 0.0), (0.5, 1.0))

# # Define our outlet pipe, swap height and width
# rec2 = Rectangle((0.5, 1.0), (1.5, 1.5))

# # Define and cut our circle
# center = (0.5, 1.0)     # Center of the circle
# radius = 0.5            # Radius of the circle
# circ1 = Circle(center = (0.5, 1.0), radius = 0.5)
# cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))    # Define our cut rectangle to be the point where our other pipe ends, and the upper right hand corner we wish to end at

# rec_inner = Rectangle((0.5, 0.875), (0.625, 1.0))
# cut_circ = Circle(center = (0.625, 0.875), radius = 0.125)

# bend = rec_inner - cut_circ
# circle = circ1 & cut_rect

# inlet_line = Line( (0.0, 1.0),(0.5, 1))
# # outlet_line = Line()

# circle = circle - inlet_line
# # Add all the geometries
# Pipe = rec1 + rec2 + circle + bend
cylinder = Cylinder(center=center, radius= width,height=height)

Bound = cylinder.sample_boundary(nr_points=n_points)
Interior = cylinder.sample_interior(nr_points=n_points)

var_to_polyvtk(Interior, "GeomTest/test_interior")
var_to_polyvtk(Bound, "GeomTest/test_bound")