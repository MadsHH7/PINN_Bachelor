import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from typing import Dict, List, Union, Tuple, Callable
import sympy as sp
import logging
import torch

from .constraint import Constraint
from .utils import _compute_outvar, _compute_lambda_weighting
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.loss import Loss, PointwiseLossNorm, IntegralLossNorm
from modulus.sym.distributed import DistributedManager
from modulus.sym.utils.sympy import np_lambdify

from modulus.sym.geometry import Geometry
from modulus.sym.geometry.helper import _sympy_criteria_to_criteria
from modulus.sym.geometry.parameterization import Parameterization, Bounds

from modulus.sym.dataset import (
    DictPointwiseDataset,
    ListIntegralDataset,
    ContinuousPointwiseIterableDataset,
    ContinuousIntegralIterableDataset,
    DictImportanceSampledPointwiseIterableDataset,
    DictVariationalDataset,
)
from sympy import Symbol, Eq, Abs

x, y = Symbol("x"), Symbol("y")
r = 1

test = {Eq(r,x**2+y**2)}
Union[sp.Basic, Callable, None]