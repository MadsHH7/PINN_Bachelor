# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Cylinder
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

print("imports ran")

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

# make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"),Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"),Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # add constraints to solver
    # make geometry
    height = 0.5
    width = 0.1
    center = (0,0,0)
    x, y,z = Symbol("x"), Symbol("y"), Symbol("z")

    cyl = Cylinder(center=center,radius=width,height=height)

# make ldc domain
    ldc_domain = Domain()

    # top wall
    Inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Cylinder,
        outvar={"u": 0, "v": 0,"w":1},
        batch_size=cfg.batch_size.Inlet,
        # lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(z, 0),
    )
    ldc_domain.add_constraint(Inlet, "Inlet")

    # Outlet = PointwiseBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=Cylinder,
    #     outvar={"u": 0, "v": 0, "w":0},
    #     batch_size=cfg.batch_size.NoSlip,
    #     criteria=z < height,
    # )
    # ldc_domain.add_constraint(no_slip, "no_slip")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Cylinder,
        outvar={"u": 0, "v": 0, "w":0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=z < height,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=Cylinder,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0,"momentum_z":0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
    )

    print("All constrains where applied")

    print("Import calls ran")
    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()

    print("Program finished")


if __name__ == "__main__":
    run()