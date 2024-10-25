# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

print("program was opened")

import os
import warnings
from eq_Laplace2D import Laplace2D

import numpy as np
from sympy import Symbol, Function, Number, Eq, Abs, cos, pi, And

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Circle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

pi = float(pi)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    ns = Laplace2D(C=1.5,rho = 1)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"),Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]



    # add constraints to solver
    # make geometry
   # Make geometry
    height = 1.0
    width = 0.5
    x, y = Symbol("x"), Symbol("y")
    
    # Define inlet pipe
    origen_inlet = (0.0, 0.0)
    rec1 = Rectangle(origen_inlet, (origen_inlet[0] + width, origen_inlet[1] + height))
    
    # Define our outlet pipe, swap height and width
    origen_outlet = (0.5, 1.0)
    rec2 = Rectangle(origen_outlet, (origen_outlet[0] + height, origen_outlet[1] + width))
    
    # Define and cut our circle
    center = (0.5, 1.0)     # Center of the circle
    radius = 0.5            # Radius of the circle
    circ1 = Circle(center = center, radius = radius)
    cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))    # Define our cut rectangle to be the point where our other pipe ends, and the upper right hand corner we wish to end at

    circle = circ1 & cut_rect

    # Make the bend on the inside corner
    rec_inner = Rectangle((0.5, 0.875), (0.625, 1.0))
    cut_circ = Circle(center = (0.625, 0.875), radius = 0.125)

    bend = rec_inner - cut_circ

    # Add all the geometries
    Pipe = rec1 + circle + rec2 + bend
    # Make domain

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0, "v": 1},
        batch_size=cfg.batch_size.Inlet,
        lambda_weighting={"u": 1.0, "v": 1.0 - cos(2*x*pi)**2},  # weight edges to be zero
        criteria=Eq(y, 0),
    )
    ldc_domain.add_constraint(inlet, "inlet")

    # Outlet
    Outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec2,
        outvar={"p": 0},
        batch_size=cfg.batch_size.Inlet,
        criteria= Eq(x, 1.5),
    )
    ldc_domain.add_constraint(Outlet, "outlet")

    # no penetration
    Inlet_pipe_left = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec1,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(x, 0.0),
    )

    ldc_domain.add_constraint(Inlet_pipe_left, "Inlet_left")

    inlet_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization=And(Eq(x, 0.5), y <= 0.875),
    )
    ldc_domain.add_constraint(inlet_right, "inlet_right")

    Outer_bend = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        # outvar= {"normal_circle_outer": 0},
        outvar = {"normal_circle": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= And((x<=0.5), (y >= 1.0))
    )
    ldc_domain.add_constraint(Outer_bend, "outer_bend")

    Inner_bend = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = Pipe,
        outvar= {"normal_circle_inner": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria= And((0.5 <= x), x <= 0.625, (0.875 <= y), y <= 1)
    )
    ldc_domain.add_constraint(Inner_bend, "inner_bend")

    outlet_bottom = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec2,
        outvar={"v": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization=And(Eq(y, 1.0), x >= 0.625),
    )
    ldc_domain.add_constraint(outlet_bottom, "outlet_bottom")

    outlet_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec2,
        outvar={"v": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization={Eq(y,1.5)},
    )
    ldc_domain.add_constraint(outlet_top, "outlet_top")





    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"continuity": 0, "irrotational": 0, "Bernoulli": 0,},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "irrotational": Symbol("sdf"),
            "Bernoulli": Symbol("sdf"),
        },
    )
    ldc_domain.add_constraint(interior, "interior")

  
    print("All constrains where applied")

    print("Import calls ran")
    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()

    print("Program finished")


if __name__ == "__main__":
    run()