from modulus.sym.utils.io.vtk import var_to_polyvtk
from pipe_bend_parameterized_geometry import *

from sympy import pi

# Construct pipe geometry
bend_angle_range=(1.323541349, 1.323541349)
radius_pipe_range=(0.1, 0.1) 
radius_bend_range=(0.1, 0.1)
inlet_pipe_length_range=(0.2, 0.2) 
outlet_pipe_length_range=(1.0, 1.0) 

Pipe = PipeBend(bend_angle_range, 
                radius_pipe_range, 
                radius_bend_range,
                inlet_pipe_length_range, 
                outlet_pipe_length_range,
)

theta = bend_angle_range[1]
radius = radius_bend_range[1]
outlet_pipe_length = outlet_pipe_length_range[1]
# Pipe.geometry = Pipe.geometry.rotate(angle=pi, axis="x")


# Sample boundary (edge) points on the cylinder
edge_points = Pipe.geometry.sample_boundary(nr_points=10000)

var_to_polyvtk(edge_points, "outputs/Pipe/Pipe Bend")