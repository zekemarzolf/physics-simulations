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
N       = 50
EPS     = 0.75
scale   = 1
v_scale = 0
rho0    = 1
mass    = 1
K       = 1
MU      = 1
H       = 1
THETA   = 0.5
GRID    = 1024

masses = np.full((N), mass)
positions = np.random.normal(size=(N, 3), scale=scale)
rho = np.full((N), rho0)
velocities = np.random.normal(size=(N, 3), scale=v_scale)

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
def collect_interactions(i, sorted_codes, sorted_positions, prefix_mass, prefix_mpos, theta, h_norm, grav_coms, grav_masses, neighbor_list):
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
def collect_all_interactions(sorted_codes, sorted_positions, prefix_mass, prefix_mpos, extent, theta, h_norm, grav_coms, grav_masses, grav_counts, neighbor_list, neighbor_counts):
    i = cuda.grid(1)
    if i >= sorted_positions.shape[0]:
        return
    
    grav_count, neighbor_count = collect_interactions(i, sorted_codes, sorted_positions, prefix_mass, prefix_mpos, theta, h_norm, grav_coms[i], grav_masses[i], neighbor_list[i])
    grav_counts[i] = grav_count
    neighbor_counts[i] = neighbor_count

@cuda.jit
def f_interact(positions, velocities, rho, pressures, masses, norm_dist, mu, h, neighbor_list, neighbor_counts, interaction_forces):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    force_x = 0.0
    force_y = 0.0
    force_z = 0.0
    C = -45 / (3.14159 * h**6)
    
    for n in range(neighbor_counts[i]):
        j = neighbor_list[i, n]
        
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        dz = positions[i, 2] - positions[j, 2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        dir_x = dx / dist
        dir_y = dy / dist
        dir_z = dz / dist

        smoothing = 1.0 - norm_dist[i, j]
        Cj = C * masses[j] * smoothing / rho[j]

        pressure_term = pressures[i] / (rho[i]**2) + pressures[j] / (rho[j]**2)

        vx_term = mu * smoothing * (velocities[j, 0] - velocities[i, 0])
        vy_term = mu * smoothing * (velocities[j, 1] - velocities[i, 1])
        vz_term = mu * smoothing * (velocities[j, 2] - velocities[i, 2])

        force_x += Cj * (smoothing * pressure_term * dir_x + vx_term)
        force_y += Cj * (smoothing * pressure_term * dir_y + vy_term)
        force_z += Cj * (smoothing * pressure_term * dir_z + vz_term)

    interaction_forces[i, 0] = force_x
    interaction_forces[i, 1] = force_y
    interaction_forces[i, 2] = force_z

@cuda.jit
def f_grav(positions, masses, eps, extent, grav_coms, grav_masses, grav_counts, grav_forces):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    force_x = 0.0
    force_y = 0.0
    force_z = 0.0
    C = -45 / (3.14159 * h**6)
    
    for n in range(grav_counts[i]):
        j = grav_coms[i, n]
        
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        dz = positions[i, 2] - positions[j, 2]
        r2  = dx*dx + dy*dy + dz*dz + (eps/extent)**2
        r3  = r2**1.5
        fac = G * masses[i] * grav_masses[j] / r3

        force_x += fac*dx
        force_y += fac*dy
        force_z += fac*dz

    grav_forces[i, 0] = force_x
    grav_forces[i, 1] = force_y
    grav_forces[i, 2] = force_z

@cuda.jit
def compute_densities(positions, masses, h, norm_dist, neighbor_list, neighbor_counts, densities):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    rho = 0.0
    C = 315/(64*3.14159*h**3)
    for n in range(neighbor_counts[i]):
        j = neighbor_list[i, n]
        rho += masses[j]*(1-norm_dist[i, j]**2)**3
    
    densities[i] = rho*C

@cuda.jit
def compute_pressures(rho, rho0, k, pressures):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    pressures[i] = k * (rho[i] - rho0)

def verlet_step(positions, masses, rho, velocities, h, k, mu eps, theta dt):
    
    
    # Compute norm_dist from neighbors lists
    # diff = positions[:, None, :] - positions[None, :, :]
    # dist = cp.sqrt(cp.sum(diff**2, axis=2))
    # norm_dist = dist / h
    compute_densities(positions, masses, h, norm_dist, neighbor_list, neighbor_counts, rho)
    return

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=positions, face_color='w', size=0.5, edge_width=0, scaling='scene')
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(positions, axis=0)

def update(ev):
    positions, rho, velocities = verlet_step(positions, masses, rho, velocities, H, K, MU, EPS, THETA dt)
    scatter.set_data(positions, face_color='w', size=0.5, edge_width=0)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

