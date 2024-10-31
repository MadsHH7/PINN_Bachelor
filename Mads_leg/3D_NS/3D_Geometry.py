from modulus.sym.utils.io.vtk import var_to_polyvtk

# from modulus.sym.geometry.primitives_3d import Cylinder, Plane, Box, Channel

# from sympy import Symbol
# import numpy as np
# from modulus.sym.geometry.parameterization import Parameterization

# # Sample point
# cyl = Cylinder(center=(0,0,0), radius=1, height=5).rotate(angle=np.pi/2, axis="x")#.rotate(angle=np.pi / 6, axis="z")

# # Define the planes that represent the two ends of the cylinder
# # end1 = Plane(point_1=(0,0,0), point_2=(0,1,0)).rotate(angle=np.pi/6, axis="x").rotate(angle=9*np.pi/12, axis="z")
# end1 = Channel(point_1=(1, 1, 1), point_2=(-1, -1, -1))#.rotate(angle=4*np.pi/6, axis="z")
# # end1 = end1.rotate(angle= np.pi / 2, axis="z")
# # end1 = end1.translate((0, -2.5, 0))

# Pipe = end1 
# # Pipe = cyl - end1
# # Pipe = cyl + Pipe

# Pipe = Pipe.sample_boundary(nr_points=5000)

# # View sampled points (coordinates and any other information)
# var_to_polyvtk(Pipe, "Pipe/Pipe Bend")

from sympy import Symbol
from modulus.sym.geometry.primitives_3d import Cylinder
from modulus.sym.geometry import Parameterization

# Define symbols
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# Create cylinder geometry (radius=1, height=2)
cylinder = Cylinder(center=[0, 0, 0], radius=1.0, height=2.0)

# Define parameterization for boundary (edge) sampling
param = Parameterization({x: 1.0})  # Adjust as needed for the edge

# Sample boundary (edge) points on the cylinder
edge_points = cylinder.sample_boundary(nr_points=100, parameterization=param)

var_to_polyvtk(edge_points, "Pipe/Pipe Bend")