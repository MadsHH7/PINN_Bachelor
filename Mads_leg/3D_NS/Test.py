from modulus.sym.utils.io.vtk import var_to_polyvtk
from pipe_bend_parameterized_geometry import *
from modulus.sym.geometry.primitives_2d import Polygon
from sympy import Symbol, symbols
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.geometry import Geometry

from sympy import pi

### GEOMETRY ###
point0 = (213.12714072, 403.12714072)
point1 = (613.12714072, 403.12714072)
point2 = (613.12714072, 243.13183868)
point3 = (447.83179646, -34.70465573)
point4 = (169.99530204, -200.00)
point5 = (0.00, -200.00)
point6 = (0.00, 200.00)
point7 = (60.00469796, 200.00)
point8 = (156.00999255, 257.11714818)
point9 = (213.12714072, 353.12244276)

geo = Polygon(
    points=[
        point0,
        point1,
        point2,
        point3,
        point4,
        point5,
        point6,
        point7,
        point8,
        point9,
    ]
)

scaling = 1 / 1000

geo = geo.scale(scaling)

edge_points = geo.sample_boundary(nr_points=1000)

var_to_polyvtk(edge_points, "outputs/TEST")