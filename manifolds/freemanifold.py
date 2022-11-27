
import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh



class Freemanifold(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(Freemanifold, self).__init__()
        self.name = 'Freemanifold'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}


    def proj_tan(self, u, x, c):

        return u