from modulus.sym.geometry.primitives_3d import Cylinder
from modulus.sym.utils.io.vtk import var_to_polyvtk

n_points = 10000

center = (0, 0, 0)
radius = 1
height = 10

Pipe = Cylinder(center, radius, height)

Boundary = Pipe.sample_boundary(nr_points=n_points)
Interior = Pipe.sample_interior(nr_points=n_points)

var_to_polyvtk(Boundary, "Pipe/boundary")
var_to_polyvtk(Interior, "Pipe/interior")