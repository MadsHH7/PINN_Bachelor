from sympy import Eq, And, Symbol, sqrt, cos, sin, pi
import os

import numpy as np

import modulus.sym
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

from modulus.sym.hydra import instantiate_arch, ModulusConfig

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)

from modulus.sym.key import Key

from PINN_Helper import get_data

from pipe_bend_parameterized_geometry import *
import numpy as np
from modulus.sym.utils.io.vtk import var_to_polyvtk
import torch
# center = (0, 0, 0)
# radius = 2
# height = 10

# cone = Cone(center, radius, height)

# cut_plane = Plane((0, -1, 5), (0, 1, 5))
n_points = 1000

bend_angle_range = (1.323541349,1.323541349)
radius_pipe_range = (0.1,0.1)
radius_bend_range = (0.2,0.2)
inlet_pipe_length_range = (0.2,0.2)
outlet_pipe_length_range = (1,1)
bend_angle = bend_angle_range[0]
radius = radius_pipe_range[0]
inlet_length = inlet_pipe_length_range[0]
outlet_pipe_length = outlet_pipe_length_range[0]

Pipe = PipeBend(bend_angle_range= bend_angle_range,
                    radius_pipe_range=radius_pipe_range,
                    radius_bend_range=radius_bend_range,
                    inlet_pipe_length_range=inlet_pipe_length_range,
                    outlet_pipe_length_range=outlet_pipe_length_range)

geo = Pipe.outlet

# Bound = geo.sample_boundary(nr_points=n_points)
Bound = geo.sample_boundary(nr_points=n_points)
Interior = geo.sample_interior(nr_points=n_points)

theta = torch.tensor(float(pi/2))
rot_matrix = torch.tensor([[torch.cos(theta),-torch.sin(theta),0],
                           [torch.sin(theta),torch.cos(theta),0],
                           [0, 0, 1]])

data_path = f"/zhome/e3/5/167986/Desktop/PINN_Bachelor/Data"
key = "pt1"
    
input, output, nr_points = get_data(
        df_path= os.path.join(data_path, f"U0{key}_Laminar.csv"),
        desired_input_keys=["x", "y", "z"],
        original_input_keys=["X (m)", "Y (m)", "Z (m)"],
        desired_output_keys=["u", "v", "w", "p"],
        original_output_keys=["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"],
        rotation_matrix=rot_matrix,
    )

flow = PointwiseConstraint.from_numpy(
        nodes = nodes,
        invar = input,
        outvar = output,
        batch_size = nr_points,
    )
# total_area = sum(Bound['area'])
# total_volume = sum(Bound['volume'])
# print(total_area)
# print(total_volume)

# var_to_polyvtk(input, "GeomTest/test_interior")
# var_to_polyvtk(flow, "GeomTest/test_bound")