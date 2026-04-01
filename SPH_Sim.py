import sys
import cupy as cp
import numpy as np
import numba as nb
import MortonEncoder as mort
import math
from numba import njit
from numba import cuda
from vispy import app, scene
from vispy.scene import visuals
from Constants import Gravity as g
from vispy.visuals.transforms import STTransform

G       = 6.674e-11
dt      = 1
N       = 500
EPS     = 0.75
scale   = 1
v_scale = 0
rho0    = 1
mass    = 1
k       = 1
mu      = 1
h       = 1
THETA   = 0.5
GRID    = 1024

m = np.full((N), mass)
r = np.random.normal(size=(N, 3), scale=scale)
rho = np.full((N), rho0)
v = np.random.normal(size=(N, 3), scale=v_scale)

@cuda.jit(device=True)
def range_mass_com(lo, hi, prefix_mass, prefix_mpos):
    mass = prefix_mass[hi] - prefix_mass[lo]
    if mass < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    com_x = (prefix_mpos[hi, 0] - prefix_mpos[lo, 0]) / mass
    com_y = (prefix_mpos[hi, 1] - prefix_mpos[lo, 1]) / mass
    com_z = (prefix_mpos[hi, 2] - prefix_mpos[lo, 2]) / mass
    return mass, com_x, com_y, com_z

@cuda.jit(device=True)
def _searchsorted(codes, lo, hi, val):
    left, right = 0, hi - lo
    while left < right:
        mid = (left + right) >> 1
        if codes[lo + mid] < val:
            left = mid + 1
        else:
            right = mid
    return left

@cuda.jit(device=True)
def collect_interactions(i, sorted_codes, sorted_positions, prefix_mass, prefix_mpos, theta, h_norm,
                         # output arrays
                         grav_coms, grav_masses, neighbor_list):
    num = sorted_positions.shape[0]
    grav_count = 0
    neighbor_count = 0
    
    stack = cuda.local.array((1024, 3), dtype=nb.int64)
    stack[0,0] = 0
    stack[0,1] = num
    stack[0,2] = 0
    top = 1
    
    while top > 0:
        top  -= 1
        lo    = stack[top, 0]
        hi    = stack[top, 1]
        depth = stack[top, 2]
        
        # same particle
        if lo >= hi:
            continue
        
        node_mass, com_x, com_y, com_z = range_mass_com(lo, hi, prefix_mass, prefix_mpos)
        size = 1/(2**depth)
        dx = com_x - sorted_positions[i, 0]
        dy = com_y - sorted_positions[i, 1]
        dz = com_z - sorted_positions[i, 2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # only 1 particle
        if hi - lo == 1:
            if lo != i:
                # check for collision
                if dist < h_norm:
                    if neighbor_count < neighbor_list.shape[0]:
                        neighbor_list[neighbor_count] = lo
                        neighbor_count += 1
                grav_masses[grav_count] = prefix_mass[hi] - prefix_mass[lo]
                grav_coms[grav_count, 0] = sorted_positions[lo, 0]
                grav_coms[grav_count, 1] = sorted_positions[lo, 1]
                grav_coms[grav_count, 2] = sorted_positions[lo, 2]
                grav_count += 1
            continue
        
        if dist > 1e-10 and size/dist < theta:
            grav_masses[grav_count] = node_mass
            grav_coms[grav_count, 0] = com_x
            grav_coms[grav_count, 1] = com_y
            grav_coms[grav_count, 2] = com_z
            grav_count += 1
            continue
            
        else:
            shift = nb.uint64(30 - 3 * (depth+1))
            parent_shift = nb.uint64(30 - 3 * depth)
            base = (sorted_codes[lo] >> parent_shift) << parent_shift
            for oct in range(8):
                child_start = base | (nb.uint64(oct) << shift)
                child_end = base | (nb.uint64(oct + 1) << shift)
                
                child_lo = lo + _searchsorted(sorted_codes, lo, hi, child_start)
                child_hi = lo + _searchsorted(sorted_codes, lo, hi, child_end)

                if child_lo < child_hi:
                    stack[top, 0] = child_lo
                    stack[top, 1] = child_hi
                    stack[top, 2] = depth + 1
                    top += 1

    return grav_count, neighbor_count

@cuda.jit
def collect_all_interactions(sorted_codes, sorted_positions, prefix_mass, prefix_mpos, extent, theta, h_norm,
                             grav_coms, grav_masses, grav_counts, neighbor_list, neighbor_counts):
    i = cuda.grid(1)
    if i >= sorted_positions.shape[0]:
        return
    
    grav_count, neighbor_count = collect_interactions(
        i, sorted_codes, sorted_positions, prefix_mass, prefix_mpos, theta, h_norm,
        grav_coms[i], grav_masses[i], neighbor_list[i])
    grav_counts[i] = grav_count
    neighbor_counts[i] = neighbor_count

@cuda.jit
def f_interact(r, v, rho, m, q, mu, k, rho0, h):
    i = cuda.grid(1)
    if i >= r.shape[0]:
        return
    
    return

def f_grav():

    return

@cuda.jit
def compute_densities(r, m, h, q, neighbor_list, neighbor_counts, densities):
    i = cuda.grid(1)
    if i >= r.shape[0]:
        return
    
    rho = 0.0
    C = (315*m)/(64*3.14159*h**3)
    for n in range(neighbor_counts[i]):
        j = neighbor_list[i, n]
        rho += (1-q[i, j]**2)**3
    
    densities[i] = rho*C

def verlet_step(r, m, h, rho, v, dt):
    
    
    # Compute q from neighbors lists
    # diff = r[:, None, :] - r[None, :, :]
    # dist = cp.sqrt(cp.sum(diff**2, axis=2))
    # q = dist / h
    compute_densities(r, m, h, q, neighbor_list, neighbor_counts, rho)
    return

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=r, face_color='w', size=1, edge_width=0, scaling='scene')
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(r, axis=0)

def update(ev):
    r, rho, v = verlet_step(r, m, h, rho, v, dt)
    scatter.set_data(r, face_color='w', size=1, edge_width=0)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

# equations and stuff is on cornell sph pdf
# use barnes-hut or RT cores to handle nearest neighbor searches
# just use vispy scatter for now
# use barnes-hut logic from n-body to compute gravity