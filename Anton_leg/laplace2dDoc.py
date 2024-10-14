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

from sympy import Symbol, Eq, Abs, And

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
    height = 1.0
    width = 0.5 
    circ_center = (0.625, 0.875)
    r = 0.125
    x, y = Symbol("x"), Symbol("y")

    # Define inlet pipe
    rec1 = Rectangle((0.0, 0.0), (0.5, 1.0))

    # Define our outlet pipe, swap height and width
    rec2 = Rectangle((0.5, 1.0), (1.5, 1.5))

    # Define and cut our circle
    center = (0.5, 1.0)     # Center of the circle
    radius = 0.5            # Radius of the circle
    circ1 = Circle(center = (0.5, 1.0), radius = 0.5)
    cut_rect = Rectangle((0.0, 1.0), (0.5, 1.5))    # Define our cut rectangle to be the point where our other pipe ends, and the upper right hand corner we wish to end at

    rec_inner = Rectangle((0.5, 0.875), (0.625, 1.0))
    cut_circ = Circle(center =circ_center, radius = r)

    bend = rec_inner - cut_circ
    circle = circ1 & cut_rect

    # Add all the geometries
    Pipe = rec1 + rec2 + circle + bend

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0, "v": 1},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"u": 1, "v": 1.0 - 4 * Abs(x)},  # weight edges to be zero
        criteria=Eq(y, 0),
    )
    ldc_domain.add_constraint(inlet, "inlet")

    # no penetration
    inlet_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization={x:0, y:(0,1)},
    )

    ldc_domain.add_constraint(inlet_left, "inlet_left")

    inlet_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec1,
        outvar={"u": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization={x:0.5,y:(0,1)},
    )
    ldc_domain.add_constraint(inlet_right, "inlet_right")


    outlet_bottom = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec2,
        outvar={"u": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization={x:(0.5,1.5), y: 1},
    )
    ldc_domain.add_constraint(inlet_right, "outlet_bottom")

    outlet_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec2,
        outvar={"u": 0.0,},
        batch_size=cfg.batch_size.NoPen,
        parameterization={x:(0.5,1.5), y: 1.5},
    )
    ldc_domain.add_constraint(inlet_right, "outlet_top")


    circle_constriants = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Pipe,
        outvar={"u": 0,"v": 1},
        # outvar={"u": (x-circ_center[0])/r,"v":(y-circ_center[1])/r,},
        batch_size=cfg.batch_size.NoPen,
        criteria=Eq(x, 0.25)
    )
    ldc_domain.add_constraint(inlet_right, "circle_constriants")


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