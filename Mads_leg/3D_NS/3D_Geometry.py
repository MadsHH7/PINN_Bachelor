from modulus.sym.utils.io.vtk import var_to_polyvtk
from pipe_bend_parameterized_geometry import *
from modulus.sym.geometry.primitives_3d import Torus, Cylinder
from sympy import Symbol, symbols
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.geometry import Geometry

from sympy import pi

# Construct pipe geometry
# bend_angle_range=(1.323541349, 1.323541349)
# radius_pipe_range=(0.1, 0.1) 
# radius_bend_range=(0.1, 0.1)
# inlet_pipe_length_range=(0.2, 0.2) 
# outlet_pipe_length_range=(1.0, 1.0) 

# Pipe = PipeBend(bend_angle_range, 
#                 radius_pipe_range, 
#                 radius_bend_range,
#                 inlet_pipe_length_range, 
#                 outlet_pipe_length_range,
# )

# theta = bend_angle_range[1]
# radius = radius_bend_range[1]
# outlet_pipe_length = outlet_pipe_length_range[1]

class TorusPiece(Geometry):
    """
    3D Torus

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of torus
    radius : int or float
        distance from center to center of tube (major radius)
    radius_tube : int or float
        radius of tube (minor radius)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(
        self, center, radius, radius_tube, bend_angle, parameterization=Parameterization()
    ):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r_1, r_2, r_3 = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center[0] * N.i + center[1] * N.j + center[2] * N.k
        OP_xy = (x - center[0]) * N.i + (y - center[1]) * N.j + (0) * N.k
        OR = radius * OP_xy / sqrt(OP_xy.dot(OP_xy))
        OP = P - O
        RP = OP - OR
        dist = sqrt(RP.dot(RP))

        # surface of the torus
        curve_parameterization = Parameterization(
            {r_1: (0, 1), r_2: (3*pi / 2, 3*pi / 2 - bend_angle), r_3: (0, 1)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        theta = 2 * pi * r_1
        phi = r_2
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + (radius + radius_tube * cos(theta)) * cos(phi),
                "y": center[1] + (radius + radius_tube * cos(theta)) * sin(phi),
                "z": center[2] + radius_tube * sin(theta),
                "normal_x": 1 * cos(theta) * cos(phi),
                "normal_y": 1 * cos(theta) * sin(phi),
                "normal_z": 1 * sin(theta),
            },
            parameterization=curve_parameterization,
            area=2 * pi * phi * radius * radius_tube,
            criteria=radius_tube * Abs(radius + radius_tube * cos(theta))
            >= r_3 * radius_tube * (radius + radius_tube),
        )
        curves = [curve_1]

        # calculate SDF
        sdf = radius_tube - dist

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (
                    center[0] - radius - radius_tube,
                    center[0] + radius + radius_tube,
                ),
                Parameter("y"): (
                    center[1] - radius - radius_tube,
                    center[1] + radius + radius_tube,
                ),
                Parameter("z"): (center[2] - radius_tube, center[2] + radius_tube),
            },
            parameterization=parameterization,
        )

        # initialize Torus
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


bend_angle = 1.323541349
radius_bend = 0.1
center_torus = (0, 0.2, 0)

Bend = TorusPiece(center=center_torus, radius=0.2, radius_tube=0.1, bend_angle=bend_angle)

Length = 1

Outlet = Cylinder(center = (0, 0, 0), radius = 0.1, height = Length)
Outlet = Outlet.rotate(angle=float(pi/2), axis="y")
Outlet = Outlet.translate((Length / 2, 0, 0))

Length_inlet = 0.2
# outlet_center = (
#     radius_bend * float(cos(-bend_angle) - center_torus[0]),
#     radius_bend * float(sin(-bend_angle) - center_torus[1]),
#     0 - center_torus[2],
# )
# cylinder_center = (
#     (Length_inlet / 2 / radius_bend) * (-outlet_center[1]) + outlet_center[0],
#     (Length_inlet / 2 / radius_bend) * (outlet_center[0]) + outlet_center[1],
#     0,
# )

Inlet_angle = (3*pi / 2 - bend_angle)

# outlet_center = (
#     0.2 * (float(cos(Inlet_angle)) + 0),
#     0.2 * (float(sin(Inlet_angle)) + 0.2),
#     0 - center_torus[2],
# )
# cylinder_center = (
#     (Length_inlet / 2 / 0.2) * (-outlet_center[1]) + outlet_center[0],
#     (Length_inlet / 2 / 0.2) * (outlet_center[0]) + outlet_center[1],
#     0,
# )

d = (
    float(cos(Inlet_angle)),
    float(sin(Inlet_angle)),
    0
)
C = (
    0.2 * float(cos(Inlet_angle)),
    0.2 * float(sin(Inlet_angle)),
    0
)

center_inlet = (
    0.0 + (0.2) * float(cos(Inlet_angle)),
    0.2 + (0.2) * float(sin(Inlet_angle)),
    0.0 + 0
)

Inlet = Cylinder(center = (0, 0, 0), radius = 0.1, height = Length_inlet)
Inlet = Inlet.rotate(angle=float(pi/2), axis="y")
Inlet = Inlet.rotate(angle=-bend_angle, axis="z")
# Inlet = Inlet.translate((0.2 * float(cos(Inlet_angle)), 
#                          0.2 * float(sin(Inlet_angle)) + 0.2 + Length_inlet/2,
#                          0))
Inlet = Inlet.translate(center_inlet)

Pipe = Outlet
Bend = Bend

# Sample boundary (edge) points on the cylinder
edge_points = Pipe.sample_boundary(nr_points=10000)
edge_points2 = Inlet.sample_boundary(nr_points=10000)
bend = Bend.sample_boundary(nr_points=10000)

var_to_polyvtk(edge_points, "outputs/Pipe/Pipe Bend")
var_to_polyvtk(edge_points2, "outputs/Pipe/Pipe Bend2")
var_to_polyvtk(bend, "outputs/Pipe/Pipe Bend3")