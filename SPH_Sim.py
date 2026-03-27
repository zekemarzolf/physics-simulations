import sys
import cupy as cp
import numpy as np
import numba as nb
import MortonEncoder as mort
from numba import njit
from numba import cuda
from vispy import app, scene
from vispy.scene import visuals
from Constants import Gravity as g
from vispy.visuals.transforms import STTransform

G      = 6.674e-11
dt     = 1
N      = 500
EPS    = 0.75
SCALE  = 2
DENS   = 1
THETA  = 0.5
GRID   = 1024
BASE_H = 1

# equations and stuff is on cornell sph pdf
# use barnes-hut or RT cores to handle nearest neighbor searches
# just use vispy scatter for now
# use barnes-hut logic from n-body to compute gravity