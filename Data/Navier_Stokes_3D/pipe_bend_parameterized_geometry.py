# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from modulus.sym.geometry.curve import SympyCurve
from modulus.sym.geometry.geometry import Geometry, csg_curve_naming
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.parameterization import Bounds, Parameter, Parameterization
from modulus.sym.geometry.primitives_3d import Cylinder, Plane
from sympy import Abs, Min, Symbol, ceiling, cos, evalf, pi, sign, sin, sqrt
from sympy.vector import CoordSys3D


class TorusPiece(Geometry):

    def __init__(
        self,
        bend_angle_range,
        radius_pipe_range,
        radius_bend_range,
    ):

        bend_angle = Symbol("bend_angle")
        radius_bend = Symbol("radius_bend")
        radius_pipe = Symbol("radius_pipe")

        if np.isscalar(bend_angle_range):
            bend_angle_range = (bend_angle_range,)

        parameters = {
            bend_angle: bend_angle_range,
            radius_bend: radius_bend_range,
            radius_pipe: radius_pipe_range,
        }

        pr = Parameterization(parameters)
        self.pr = parameters

        center_torus = (0, 0, 0)
        self.center_torus = center_torus

        # Make sympy symbols to use for spatial coordinates.
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r_1, r_2, r_3 = (
            Symbol(csg_curve_naming(0)),
            Symbol(csg_curve_naming(1)),
            Symbol(csg_curve_naming(2)),
        )

        # Calculate distance to surface of torus. To be used in SDF.
        N = CoordSys3D("N")
        P = x * N.i + y * N.j + z * N.k
        O = center_torus[0] * N.i + center_torus[1] * N.j + center_torus[2] * N.k
        OP_xy = (x - center_torus[0]) * N.i + (y - center_torus[1]) * N.j + (0) * N.k
        OR = radius_bend * OP_xy / sqrt(OP_xy.dot(OP_xy))
        OP = P - O
        RP = OP - OR
        dist = sqrt(RP.dot(RP))

        # Calculate signed distance to the line passing through end of torus piece projected onto XY-plane.
        # Define vector pointing from center_torus of torus to the center_torus of the end of the torus piece.
        OP_0 = (
            (radius_bend * cos(bend_angle) - center_torus[0]) * N.i
            + (radius_bend * sin(bend_angle) - center_torus[1]) * N.j
            + (0) * N.k
        )
        # Define normal vector of the line. Used to calculate signed distance.
        normal = cos(bend_angle - pi / 2) * N.i + sin(bend_angle - pi / 2) * N.j + (0) * N.k
        # Find the distance from P to the line passing through OP_0 and torus center_torus.
        d_vec = OP - (OP.dot(OP_0)) * OP_0
        d = sqrt(d_vec.dot(d_vec))
        # Determine the sign of the distance.
        sgn = sign(d_vec.dot(normal))
        # Get signed distance.
        signed_dist_to_line = sgn * d

        # surface of the torus
        curve_parameterization = Parameterization({r_1: (0, 1), r_2: (0, bend_angle_range[-1]), r_3: (0, 1)})
        curve_parameterization = Parameterization.combine(curve_parameterization, pr)

        theta = 2 * pi * r_1
        phi = r_2
        curve_1 = SympyCurve(
            functions={
                "x": center_torus[0] + (radius_bend + radius_pipe * cos(theta)) * cos(phi),
                "y": center_torus[1] + (radius_bend + radius_pipe * cos(theta)) * sin(phi),
                "z": center_torus[2] + radius_pipe * sin(theta),
                "normal_x": 1 * cos(theta) * cos(phi),
                "normal_y": 1 * cos(theta) * sin(phi),
                "normal_z": 1 * sin(theta),
            },
            parameterization=curve_parameterization,
            area=2 * pi * phi * radius_bend * radius_pipe,
            criteria=radius_pipe * Abs(radius_bend + radius_pipe * cos(theta))
            >= r_3 * radius_pipe * (radius_bend + radius_pipe),
        )
        curves = [curve_1]

        # The sdf is the minimum of the distances to the lines passing through each torus piece end in 2d, and the distance to the torus wall.
        # Note: If outside the torus, dist is negative. If outside the torus piece, but inside the torus:
        # Either signed_dist_to_end or y - center_torus[1] is negative. y - center_torus[1] is the signed distance to the line passing through one torus piece end by definition.
        sdf = Min(signed_dist_to_line, radius_pipe - dist, y - center_torus[1])

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (
                    center_torus[0] - radius_bend - radius_pipe,
                    center_torus[0] + radius_bend + radius_pipe,
                ),
                Parameter("y"): (
                    center_torus[1] - radius_bend - radius_pipe,
                    center_torus[1] + radius_bend + radius_pipe,
                ),
                Parameter("z"): (center_torus[2] - radius_pipe, center_torus[2] + radius_pipe),
            },
            parameterization=pr,
        )

        # initialize Torus
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=pr,
        )


class PipeBend(Geometry):

    def __init__(
        self,
        bend_angle_range,
        radius_pipe_range,
        radius_bend_range,
        inlet_pipe_length_range,
        outlet_pipe_length_range,
        nr_inlet_planes=2,
        nr_bend_planes=4,
        nr_outlet_planes=9,
    ):

        # Define the SymPy symbol to be used in the parameterizations.
        bend_angle = Symbol("bend_angle")
        radius_bend = Symbol("radius_bend")
        radius_pipe = Symbol("radius_pipe")
        inlet_pipe_length = Symbol("inlet_pipe_length")
        outlet_pipe_length = Symbol("outlet_pipe_length")

        # Define the dictionary of parameters for the parameterization.
        parameters = {
            bend_angle: bend_angle_range,
            radius_bend: radius_bend_range,
            radius_pipe: radius_pipe_range,
            inlet_pipe_length: inlet_pipe_length_range,
            outlet_pipe_length: outlet_pipe_length_range,
        }

        pr = Parameterization(parameters)
        self.pr = parameters

        # Set bend part of geometry using a torus piece.
        bend = TorusPiece(
            bend_angle_range,
            radius_pipe_range,
            radius_bend_range,
        )

        self.bend = bend

        center_torus = bend.center_torus
        self.center_torus = center_torus
        self.radius_pipe = radius_pipe
        self.radius_bend = radius_bend
        self.inlet_pipe_length = inlet_pipe_length
        self.outlet_pipe_length = outlet_pipe_length

        # Set inlet pipe.
        inlet_pipe = Cylinder(center=(0, 0, 0), radius=radius_pipe, height=inlet_pipe_length, parameterization=pr)
        inlet_pipe = inlet_pipe.rotate(angle=float(np.pi / 2), axis="x")
        inlet_pipe = inlet_pipe.translate((radius_bend, -inlet_pipe_length / 2, 0))
        self.inlet_pipe = inlet_pipe

        # Calculate center of outlet pipe.
        outlet_center = (
            radius_bend * (cos(bend_angle) - center_torus[0]),
            radius_bend * (sin(bend_angle) - center_torus[1]),
            0 - center_torus[2],
        )
        cylinder_center = (
            (outlet_pipe_length / 2 / radius_bend) * (-outlet_center[1]) + outlet_center[0],
            (outlet_pipe_length / 2 / radius_bend) * (outlet_center[0]) + outlet_center[1],
            0,
        )

        # Set outlet pipe.
        outlet_pipe = Cylinder(center=(0, 0, 0), radius=radius_pipe, height=outlet_pipe_length, parameterization=pr)
        outlet_pipe = outlet_pipe.rotate(angle=float(np.pi / 2), axis="x")
        outlet_pipe = outlet_pipe.rotate(angle=bend_angle, axis="z")
        outlet_pipe = outlet_pipe.translate(cylinder_center)
        self.outlet_pipe = outlet_pipe

        # Set the geometry on the class, so it can be accessed in PINN script.
        self.geometry = bend + inlet_pipe + outlet_pipe

        # Define inlet, outlet and integral planes.
        inlet = Plane(
            (center_torus[0], radius_pipe, radius_pipe),
            (center_torus[0], -radius_pipe, -radius_pipe),
            parameterization=pr,
        )
        inlet = inlet.rotate(angle=np.pi / 2)
        inlet = inlet.translate((radius_bend, -inlet_pipe_length, 0))
        center_in_x = radius_bend
        center_in_y = -inlet_pipe_length
        self.inlet = inlet
        self.inlet_center = (center_in_x, center_in_y)

        # Integral planes in inlet pipe. Plane at the end of inlet pipe included here. 
        inlet_pipe_planes = []
        inlet_pipe_planes_centers = []
        for i in range(nr_inlet_planes):
            plane = inlet.translate((0, (i + 1) / nr_inlet_planes * inlet_pipe_length, 0))
            center_x = radius_bend
            center_y = center_in_y + (i + 1) / nr_inlet_planes * inlet_pipe_length
            inlet_pipe_planes.append(plane)
            inlet_pipe_planes_centers.append((center_x, center_y))
        self.inlet_pipe_planes = inlet_pipe_planes
        self.inlet_pipe_planes_centers = inlet_pipe_planes_centers
        
        # Integral planes in the bend. Plane at the start of outlet pipe included here. 
        bend_planes = []
        bend_planes_centers = []
        # Integral planes in bend. 
        for i in range(nr_bend_planes):
            plane = inlet_pipe_planes[-1].rotate(angle=((i + 1) / nr_bend_planes * bend_angle))
            center_x = radius_bend * cos((i + 1) / nr_bend_planes * bend_angle)
            center_y = radius_bend * sin((i + 1) / nr_bend_planes * bend_angle)
            bend_planes.append(plane)
            bend_planes_centers.append((center_x, center_y))
        self.bend_planes = bend_planes
        self.bend_planes_centers = bend_planes_centers

        # Outlet pipe points in this direction with this length.
        direction = (outlet_pipe_length * cos(bend_angle + pi / 2), outlet_pipe_length * sin(bend_angle + pi / 2))

        # Integral planes in outlet pipe.
        outlet_pipe_planes = []
        outlet_pipe_planes_centers=[]
        for i in range(nr_outlet_planes):
            plane = bend_planes[-1].translate(
                ((i + 1) / (nr_outlet_planes + 1) * direction[0], (i + 1) / (nr_outlet_planes + 1) * direction[1], 0)
            )
            center_x = radius_bend * cos(bend_angle) + (i + 1) / (nr_outlet_planes + 1) * direction[0]
            center_y = radius_bend * sin(bend_angle) + (i + 1) / (nr_outlet_planes + 1) * direction[1]
            outlet_pipe_planes.append(plane)
            outlet_pipe_planes_centers.append((center_x, center_y))
        self.outlet_pipe_planes = outlet_pipe_planes
        self.outlet_pipe_planes_center = outlet_pipe_planes_centers
        
        # Outlet plane.
        outlet = bend_planes[-1].translate((direction[0], direction[1], 0))
        center_out_x = radius_bend * cos(bend_angle) + direction[0]
        center_out_y = radius_bend * sin(bend_angle) + direction[1]
        self.outlet = outlet
        self.outlet_center = (center_out_x, center_out_y)

        self.integral_planes = inlet_pipe_planes + bend_planes + outlet_pipe_planes
        self.planes_centers = inlet_pipe_planes_centers + bend_planes_centers + outlet_pipe_planes_centers