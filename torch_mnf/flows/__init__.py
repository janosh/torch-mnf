"""This package implements various flows.
Each flow is invertible and outputs its log abs det J^-1 "regularization"
via the log_det method.

The Jacobian's determinant measures the local change of volume (due to
stretching or squashing of space) under the flows transformation function
y = f(x). We take the inverse Jacobian because we want to know how volume
changes under x = f^-1(y), going from the space Y of real-world data we can
observe to the space X of the base distribution. We work in log space for
numerical stability, and take the absolute value because we don't care about
changes in orientation, i.e. whether space is mirrored/reflected. We only
care if it is stretched, since all we need is conservation of probability
mass to retain a valid transformed PDF under f, no matter where in space
that mass ends up.
"""

from .affine_constant_flow import ActNormFlow, AffineConstantFlow
from .affine_half_flow import AffineHalfFlow
from .core import NormalizingFlow, NormalizingFlowModel
from .glow import Glow
from .maf import IAF, MAF
from .rnvp import RNVP
from .spline_flow import NSF_AR, NSF_CL
