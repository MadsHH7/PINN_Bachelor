from modulus.sym.geometry.primitives_3d import Cone, Plane
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.utils.io.vtk import var_to_polyvtk
from pipe_bend_parameterized_geometry import PipeBend
from modulus.sym.geometry import Parameterization
from sympy import pi
from numpy import sum
# center = (0, 0, 0)
# radius = 2
# height = 10

# cone = Cone(center, radius, height)

# cut_plane = Plane((0, -1, 5), (0, 1, 5))
bend_angle = (pi/2,pi/2)
radius_pipe = (1,1)
radius_bend = (1,1)
inlet_pipe_length = (5,5)
outlet_pipe_length = (5,5)
n_points = 10000

pipe = PipeBend(bend_angle_range= bend_angle,
                radius_pipe_range=radius_pipe,
                radius_bend_range=radius_bend,
                inlet_pipe_length_range=inlet_pipe_length,
                outlet_pipe_length_range=outlet_pipe_length)

geo = pipe.outlet

# Bound = geo.sample_boundary(nr_points=n_points)
Bound = geo.sample_boundary(nr_points=n_points)
Interior = geo.sample_interior(nr_points=n_points)

total_area = sum(Bound['area'])
total_volume = sum(Bound['volume'])
print(total_area)
print(total_volume)

var_to_polyvtk(Interior, "GeomTest/test_interior")
var_to_polyvtk(Bound, "GeomTest/test_bound")