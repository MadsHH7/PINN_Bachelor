from pipe_bend_parameterized_geometry import PipeBend
# from modulus.sym.utils.io.vtk import var_to_polyvtk

# import numpy as np

# # test other geometry 
# Pipe = PipeBend(bend_angle_range=(0, np.pi/2), 
#                 radius_pipe_range=(1, 1), 
#                 radius_bend_range=(1, 1),
#                 inlet_pipe_length_range=(0.5, 5), 
#                 outlet_pipe_length_range=(0.5, 5),
# )

# bound = Pipe.sample_boundary(1000)

# var_to_polyvtk(bound, "Pipe_test/Pipe Bend")

from sympy import Symbol
import numpy as np
from modulus.sym.geometry.parameterization import Parameterization

# Initialize PipeBend with desired ranges as shown before
bend_angle_range = (np.pi / 4, np.pi / 2)
radius_pipe_range = (0.1, 0.2)
radius_bend_range = (1.0, 1.5)
inlet_pipe_length_range = (0.5, 1.0)
outlet_pipe_length_range = (0.5, 1.0)

pipe_bend = PipeBend(
    bend_angle_range=bend_angle_range,
    radius_pipe_range=radius_pipe_range,
    radius_bend_range=radius_bend_range,
    inlet_pipe_length_range=inlet_pipe_length_range,
    outlet_pipe_length_range=outlet_pipe_length_range
)

# Sample points in the interior
num_interior_points = 1000

# Define parameter values explicitly
parameterization = Parameterization({
    Symbol("bend_angle"): np.pi / 3,
    Symbol("radius_bend"): 1.25,
    Symbol("radius_pipe"): 0.15,
    Symbol("inlet_pipe_length"): 0.75,
    Symbol("outlet_pipe_length"): 0.75
})

# Sample points
interior_points = pipe_bend.sample_boundary(
    num_interior_points,
    parameterization=parameterization,
    # compute_sdf=True  # Set to True if you need SDF values
)

# View sampled points (coordinates and any other information)
var_to_polyvtk(interior_points, "Pipe_test/Pipe Bend")