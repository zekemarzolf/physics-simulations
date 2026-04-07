import os
import cupy as cp
import numpy as np
import numba as nb
import random as rand
import MortonEncoder as mort
import math
from numba import njit
from numba import cuda
from vispy import app, scene
from vispy.scene import visuals
from Constants import Gravity as g
from vispy.visuals.transforms import STTransform

#G       = 6.674e-11
G       = 1
dt      = 0.001
N       = 1000
EPS     = 20
scale   = 1
v_scale = 100
rho0    = 1
K       = 1e6
MU      = 1
GAMMA   = 4
H       = 15
THETA   = 0.8
GRID    = 1024

def sphere_lattice(N_target, inner_radius, outer_radius):
    # figure out spacing to get approximately N particles
    volume = (4/3) * np.pi * (outer_radius**3 - inner_radius**3)
    spacing = (volume / N_target) ** (1/3)
    
    # generate cubic lattice over bounding box
    coords = np.arange(-outer_radius, outer_radius + spacing, spacing)
    gx, gy, gz = np.meshgrid(coords, coords, coords)
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    
    # clip to shell
    r = np.linalg.norm(points, axis=1)
    mask = (r >= inner_radius) & (r <= outer_radius)
    points = points[mask]
    
    return points

inner_radius = 25.0
outer_radius = 50.0
volume = (4/3)*np.pi*(outer_radius**3 - inner_radius**3)

positions = sphere_lattice(N, inner_radius=10.0, outer_radius=50.0)
N = len(positions)
mass  = rho0 * volume / N
masses = np.full((N), mass)
rho = np.full((N), rho0)
velocities = np.random.normal(size=(N, 3), scale=v_scale)
halfstep_up_velocities = np.zeros_like(velocities)

volume = (4/3) * np.pi * outer_radius**3 - (4/3) * np.pi * inner_radius**3
avg_spacing = (volume / N) ** (1/3)
#H = 100 * avg_spacing

def prefix_aggregates(sorted_masses, sorted_positions):
    num = len(sorted_positions)
    prefix_mass = cp.zeros(num + 1, dtype=cp.float64)
    prefix_mpos = cp.zeros((num + 1, 3), dtype=cp.float64)
    prefix_mass[1:] = cp.cumsum(sorted_masses)
    prefix_mpos[1:] = cp.cumsum(sorted_masses[:, None] * sorted_positions, axis=0)
    return prefix_mass, prefix_mpos

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
def collect_interactions(i, sorted_codes, sorted_positions, sorted_real_positions, prefix_mass, prefix_mpos, theta, h_norm, grav_coms, grav_masses, neighbor_list):
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
                grav_coms[grav_count, 0] = sorted_real_positions[lo, 0]
                grav_coms[grav_count, 1] = sorted_real_positions[lo, 1]
                grav_coms[grav_count, 2] = sorted_real_positions[lo, 2]
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
def collect_all_interactions(sorted_codes, sorted_positions, sorted_real_positions, prefix_mass, prefix_mpos, extent, theta, h_norm, grav_coms, grav_masses, grav_counts, neighbor_list, neighbor_counts):
    i = cuda.grid(1)
    if i >= sorted_positions.shape[0]:
        return
    
    grav_count, neighbor_count = collect_interactions(i, sorted_codes, sorted_positions, sorted_real_positions, prefix_mass, prefix_mpos, theta, h_norm, grav_coms[i], grav_masses[i], neighbor_list[i])
    grav_counts[i] = grav_count
    neighbor_counts[i] = neighbor_count

@cuda.jit
def f_interact(positions, velocities, rho, pressures, masses, mu, h, neighbor_list, neighbor_counts, interaction_forces):
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
        q = dist/h
        dir_x = dx / dist
        dir_y = dy / dist
        dir_z = dz / dist

        smoothing = 1.0 - q
        Cj = C * masses[j] * smoothing / rho[j]

        pressure_term = pressures[i] / (rho[i]**2) + pressures[j] / (rho[j]**2)

        vx_term = mu * smoothing * (velocities[j, 0] - velocities[i, 0])
        vy_term = mu * smoothing * (velocities[j, 1] - velocities[i, 1])
        vz_term = mu * smoothing * (velocities[j, 2] - velocities[i, 2])

        force_x += Cj * (smoothing * pressure_term * dir_x - vx_term)
        force_y += Cj * (smoothing * pressure_term * dir_y - vy_term)
        force_z += Cj * (smoothing * pressure_term * dir_z - vz_term)

    interaction_forces[i, 0] = force_x * masses[i]
    interaction_forces[i, 1] = force_y * masses[i]
    interaction_forces[i, 2] = force_z * masses[i]

@cuda.jit
def f_grav(positions, masses, eps, G, grav_coms, grav_masses, grav_counts, grav_forces):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    force_x = 0.0
    force_y = 0.0
    force_z = 0.0
    
    for n in range(grav_counts[i]):
        dx = positions[i, 0] - grav_coms[i, n, 0]
        dy = positions[i, 1] - grav_coms[i, n, 1]
        dz = positions[i, 2] - grav_coms[i, n, 2]
        r2  = dx*dx + dy*dy + dz*dz + eps**2
        r3  = r2**1.5
        fac = G * masses[i] * grav_masses[i, n] / r3

        force_x -= fac*dx
        force_y -= fac*dy
        force_z -= fac*dz

    grav_forces[i, 0] = force_x
    grav_forces[i, 1] = force_y
    grav_forces[i, 2] = force_z

@cuda.jit
def compute_densities(positions, masses, h, neighbor_list, neighbor_counts, densities):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    
    C = 315/(64*3.14159*h**3)
    rho = masses[i]
    for n in range(neighbor_counts[i]):
        j = neighbor_list[i, n]
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        dz = positions[i, 2] - positions[j, 2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        q = dist / h
        rho += masses[j]*(1-q**2)**3
    
    densities[i] = rho*C

@cuda.jit
def compute_pressures(rho, rho0, k, gamma, pressures):
    i = cuda.grid(1)
    if i >= rho.shape[0]:
        return
    # Linear Pressure
    #pressures[i] = k * abs(rho[i] - rho0)
    # Polytropic Pressure
    pressures[i] = k * rho[i] ** gamma

def leapfrog_step(positions, masses, velocities, halfstep_down_velocities, h, k, gamma, mu, eps, theta, dt):
    threads = 256
    blocks = (N + threads - 1) // threads
    
    gpu_positions = cp.asarray(positions)
    gpu_masses = cp.asarray(masses)
    gpu_velocities = cp.asarray(velocities)
    gpu_halfstep_down_velocities = cp.asarray(halfstep_down_velocities)
    
    norm_positions, mins, extent = mort.normalize_positions(positions)
    codes = cp.asarray(mort.morton_codes(norm_positions, GRID))
    sort_idx = cp.argsort(codes)
    unsort_idx = cp.argsort(sort_idx)
    
    gpu_norm_positions = cp.asarray(norm_positions)
    sorted_codes = codes[sort_idx]
    sorted_positions = gpu_norm_positions[sort_idx]
    sorted_real_positions = gpu_positions[sort_idx]
    sorted_masses = gpu_masses[sort_idx]
    sorted_velocities = gpu_velocities[sort_idx]
    prefix_mass, prefix_mpos = prefix_aggregates(sorted_masses, sorted_positions)
    prefix_mass_real, prefix_mpos_real = prefix_aggregates(sorted_masses, sorted_real_positions)
    h_norm = h/extent
    
    grav_coms = cp.empty((N, 1024, 3), dtype=cp.float64)
    grav_masses = cp.zeros((N, 1024), dtype=cp.float64)
    grav_counts = cp.zeros((N), dtype=cp.int32)
    neighbor_list = cp.empty((N, 64), dtype=cp.int32)
    neighbor_counts = cp.zeros((N), dtype=cp.int32)
    
    collect_all_interactions[blocks, threads](sorted_codes, sorted_positions, sorted_real_positions, prefix_mass, prefix_mpos, extent, theta, h_norm, grav_coms, grav_masses, grav_counts, neighbor_list, neighbor_counts)
    
    print(f"neighbor_counts min: {cp.asnumpy(neighbor_counts).min()}, max: {cp.asnumpy(neighbor_counts).max()}")
    print(f"extent: {extent}, h_norm: {h_norm}")
    
    rho = cp.zeros((N))
    compute_densities[blocks, threads](sorted_real_positions, sorted_masses, h, neighbor_list, neighbor_counts, rho)
    pressures = cp.zeros((N))
    compute_pressures[blocks, threads](rho, rho0, k, gamma, pressures)
    
    interaction_forces = cp.zeros((N, 3))
    grav_forces = cp.zeros((N, 3))
    f_interact[blocks, threads](sorted_real_positions, sorted_velocities, rho, pressures, sorted_masses, mu, h, neighbor_list, neighbor_counts, interaction_forces)
    f_grav[blocks, threads](sorted_real_positions, sorted_masses, eps, G, grav_coms, grav_masses, grav_counts, grav_forces)
    accelerations = (interaction_forces+grav_forces)/sorted_masses[:, None]
    unsorted_accelerations = accelerations[unsort_idx]
    
    gpu_halfstep_up_velocities = gpu_halfstep_down_velocities + unsorted_accelerations*dt
    gpu_new_velocities = gpu_halfstep_up_velocities + unsorted_accelerations*dt
    gpu_new_positions = gpu_positions + gpu_halfstep_up_velocities*dt
    
    #com_velocity = cp.sum(gpu_masses[:,None] * gpu_new_velocities, axis=0) / cp.sum(gpu_masses)
    #gpu_new_velocities -= com_velocity
    #gpu_halfstep_up_velocities -= com_velocity
    
    new_positions = cp.asnumpy(gpu_new_positions)
    new_masses = cp.asnumpy(gpu_masses)
    new_velocities = cp.asnumpy(gpu_new_velocities)
    halfstep_up_velocities = cp.asnumpy(gpu_halfstep_up_velocities)
    rho_cpu = cp.asnumpy(rho[unsort_idx])

    return new_positions, new_masses, new_velocities, halfstep_up_velocities, rho_cpu

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
initial_colors = np.ones((N, 4)) * [0.3, 0.1, 0.8, 0.5]
scatter = visuals.Markers(pos=positions, face_color=initial_colors, size=0.5, edge_width=0, scaling='scene')
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(positions, axis=0)

def update(ev):
    global positions, masses, velocities, halfstep_up_velocities
    positions, masses, velocities, halfstep_up_velocities, rho_cpu = leapfrog_step(positions, masses, velocities, halfstep_up_velocities, H, K, MU, GAMMA, EPS, THETA, dt)
    
    # scale size and color by density
    rho_norm = np.clip(rho_cpu / (rho0 * 2), 0, 1)
    print(f"rho min: {rho_cpu.min():.2f}, max: {rho_cpu.max():.2f}, std: {rho_cpu.std():.4f}")
    print(f"rho_norm min: {rho_norm.min():.4f}, max: {rho_norm.max():.4f}")
    sizes = rho_norm * 2 # denser = bigger
    colors = np.zeros((N, 4))
    colors[:, 0] = rho_norm # red (increases with density)
    colors[:, 1] = 0.3 * (1 - rho_norm)  # green (decreases with density)
    colors[:, 2] = 0.8 * (1 - rho_norm)  # blue (decreases with density)
    colors[:, 3] = 0.7  # constant alpha    
    
    scatter.set_data(positions, face_color=colors, size=sizes, edge_width=0)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

