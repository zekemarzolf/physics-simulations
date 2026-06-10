import numpy as np
import cupy as cp
import numba as nb
from numba import njit
from numba import cuda

@njit
def _spread_bits(v):
    v = v & nb.uint64(0x3FF)
    v = (v | (v << nb.uint64(16))) & nb.uint64(0xFF0000FF)
    v = (v | (v << nb.uint64(8)))  & nb.uint64(0x0F00F00F)
    v = (v | (v << nb.uint64(4)))  & nb.uint64(0xC30C30C3)
    v = (v | (v << nb.uint64(2)))  & nb.uint64(0x49249249)
    return v

@njit
def _interleave3(xi, yi, zi):
    x = _spread_bits(nb.uint64(xi))
    y = _spread_bits(nb.uint64(yi))
    z = _spread_bits(nb.uint64(zi))
    return x | (y << nb.uint64(1)) | (z << nb.uint64(2))

@njit
def morton_codes(norm_positions,GRID):
    # compute morton codes
    num = norm_positions.shape[0]
    g = nb.uint64(GRID)
    m_positions = np.zeros(num, dtype=np.uint64)
    for i in range(num):
        xi = min(int(norm_positions[i, 0] * GRID), GRID - 1)
        yi = min(int(norm_positions[i, 1] * GRID), GRID - 1)
        zi = min(int(norm_positions[i, 2] * GRID), GRID - 1)
        m_positions[i] = _interleave3(xi, yi, zi)   
    return m_positions

def normalize_positions(positions):
    mins   = positions.min(axis=0)
    extent = (positions.max(axis=0) - mins).max()
    if extent < 1e-30:
        extent = 1.0
    norm = (positions - mins) / extent
    # clamp so no value hits exactly 1.0 (would overflow grid index)
    norm = np.clip(norm, 0.0, 1.0 - 1e-9)
    return norm, mins, extent