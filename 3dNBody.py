import sys
import numpy as np
import numba as nb
import MortonEncoder as mort
from numba import njit
from vispy import app, scene
from vispy.scene import visuals
from Constants import Gravity as g
from vispy.visuals.transforms import STTransform

G     = 6.674e-11
dt    = 1
N     = 500
eps   = .5
scale = 2
mass = 1e5
BASE_MASS = mass
rad = .02
BASE_RAD  = rad
THETA = 0.5
GRID  = 1024

def prefix_aggregates(sorted_masses, sorted_positions):
    num = len(sorted_positions)
    prefix_mass = np.zeros(num + 1, dtype=np.float64)
    prefix_mpos = np.zeros((num + 1, 3), dtype=np.float64)
    prefix_mass[1:] = np.cumsum(sorted_masses)
    prefix_mpos[1:] = np.cumsum(sorted_masses[:, None] * sorted_positions, axis=0)
    return prefix_mass, prefix_mpos

@njit 
def range_mass_com(lo, hi, prefix_mass, prefix_mpos):
    mass = prefix_mass[hi] - prefix_mass[lo]
    if mass < 1e-30:
        return 0.0, np.zeros(3)
    com = (prefix_mpos[hi] - prefix_mpos[lo])/mass
    return mass, com

@njit
def collect_interactions(q_idx, sorted_codes, sorted_positions, prefix_mass, prefix_mpos, sorted_radii, extent, theta):
    num = sorted_positions.shape[0]
    count = 0
    out_masses = np.zeros(8*num)
    out_coms = np.zeros((8*num, 3))
    out_collisions = np.empty(64, dtype=nb.int64)
    collision_count = 0
    
    stack = np.empty((1024, 3), dtype=nb.int64)
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
        
        node_mass, node_com = range_mass_com(lo, hi, prefix_mass, prefix_mpos)
        size = 1/(2**depth)
        dx = node_com[0] - sorted_positions[q_idx, 0]
        dy = node_com[1] - sorted_positions[q_idx, 1]
        dz = node_com[2] - sorted_positions[q_idx, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # only 1 particle
        if hi - lo == 1:
            if lo != q_idx:
                # check for collision
                if dist < (sorted_radii[q_idx] + sorted_radii[lo])/extent:
                    out_collisions[collision_count] = lo
                    collision_count += 1
                out_masses[count] = prefix_mass[hi] - prefix_mass[lo]
                out_coms[count, 0] = sorted_positions[lo, 0]
                out_coms[count, 1] = sorted_positions[lo, 1]
                out_coms[count, 2] = sorted_positions[lo, 2]
                count += 1
            continue
        
        if dist > 1e-10 and size/dist < theta:
            out_masses[count] = node_mass
            out_coms[count, 0] = node_com[0]
            out_coms[count, 1] = node_com[1]
            out_coms[count, 2] = node_com[2]
            count += 1
            continue
            
        else:
            shift = np.uint64(30 - 3 * (depth+1))
            parent_shift = np.uint64(30 - 3 * depth)
            base = (sorted_codes[lo] >> parent_shift) << parent_shift
            for oct in range(8):
                child_start = base | (np.uint64(oct) << shift)
                child_end = base | (np.uint64(oct + 1) << shift)
                
                child_lo = lo + _searchsorted(sorted_codes, lo, hi, child_start)
                child_hi = lo + _searchsorted(sorted_codes, lo, hi, child_end)

                if child_lo < child_hi:
                    stack[top, 0] = child_lo
                    stack[top, 1] = child_hi
                    stack[top, 2] = depth + 1
                    top += 1
            
    return out_masses, out_coms, count, out_collisions, collision_count

@njit
def _searchsorted(codes, lo, hi, val):
    left, right = 0, hi - lo
    while left < right:
        mid = (left + right) >> 1
        if codes[lo + mid] < val:
            left = mid + 1
        else:
            right = mid
    return left

@njit
def _compute_acc(sorted_pos, sorted_codes, prefix_mass, prefix_mpos, sorted_radii, theta, eps, extent):
    num = sorted_pos.shape[0]
    acc = np.zeros((num, 3))
    collisions = np.full((num * 64, 2), -1, dtype=nb.int64)
    n_coll = 0
    
    for i in range(num):
        out_masses, out_coms, count, out_collisions, collision_count = collect_interactions(i, sorted_codes, sorted_pos, prefix_mass, prefix_mpos, sorted_radii, extent, theta)
        ax = ay = az = 0.0
        for j in range(count):
            dx  = out_coms[j, 0] - sorted_pos[i, 0]
            dy  = out_coms[j, 1] - sorted_pos[i, 1]
            dz  = out_coms[j, 2] - sorted_pos[i, 2]
            r2  = dx*dx + dy*dy + dz*dz + (eps/extent)**2
            r3  = r2**1.5
            fac = G * out_masses[j] / r3
            ax += fac * dx
            ay += fac * dy
            az += fac * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
        for k in range(collision_count):
            j = out_collisions[k]
            if i < j:
                collisions[n_coll, 0] = i
                collisions[n_coll, 1] = j
                n_coll += 1
    return acc/(extent**2), collisions[:n_coll]
    
def bh_acc_calc(pos, mass, rad, theta, eps):
    norm_pos, mins, extent = mort.normalize_positions(pos)
    codes = mort.morton_codes(norm_pos, GRID)
    sort_idx = np.argsort(codes)
    sorted_codes = codes[sort_idx]
    sorted_pos   = norm_pos[sort_idx]
    sorted_mass  = mass[sort_idx]
    sorted_radii = rad[sort_idx]
    prefix_mass, prefix_mpos = prefix_aggregates(sorted_mass, sorted_pos)
    acc_sorted, collisions = _compute_acc(
    sorted_pos, sorted_codes, prefix_mass, prefix_mpos, sorted_radii, theta, eps, extent)
    unsort_idx = np.argsort(sort_idx)
    if len(collisions) > 0:
        valid = collisions[:,0] >= 0 
        collisions[valid, 0] = sort_idx[collisions[valid, 0]]
        collisions[valid, 1] = sort_idx[collisions[valid, 1]]
        return acc_sorted[unsort_idx], collisions[valid]
    return acc_sorted[unsort_idx], collisions
    
def resolve_collisions(p, v, m, r, collisions):
    if len(collisions) == 0:
        return p, v, m, r
    num = len(m)
    parent = np.arange(num)
    rank = np.zeros(num, dtype=np.int64)
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
    
    for k in range(len(collisions)):
        union(collisions[k, 0], collisions[k, 1])
    
    new_p = p.copy()
    new_v = v.copy()
    new_m = m.copy()
    new_r = r.copy()
    alive = np.ones(num, dtype=bool)
    
    for i in range(num):
        if not alive[i]:
            continue
        root = find(i)
        if root == i:
            continue
        
        mi   = new_m[root]
        mj   = new_m[i]
        mtot = mi + mj

        new_v[root] = (mi * new_v[root] + mj * new_v[i]) / mtot
        new_p[root] = (mi * new_p[root] + mj * new_p[i]) / mtot
        new_m[root] = mtot

        new_r[root] = BASE_RAD * (new_m[root] / BASE_MASS) ** (1/3)

        alive[i] = False
    
    return new_p[alive], new_v[alive], new_m[alive], new_r[alive]

def vec_acc_calc(p, mass):
    displacements = p[None,:,:] - p[:,None,:]
    sqDist = np.sum(displacements**2, axis=2) + eps**2
    r3 = sqDist**1.5
    np.fill_diagonal(r3, np.inf)
    massProd = G * mass[None,:]
    accelerations = np.sum((massProd/r3)[...,None]*displacements, axis=1)
    return accelerations

def verlet_step(p, v, r, mass, dt):
    a, _ = bh_acc_calc(p, mass, r, THETA, eps)
    new_p = p + v*dt + 0.5*a*dt**2
    new_a, collisions = bh_acc_calc(new_p, mass, r, THETA, eps)
    new_v = v + 0.5*(a + new_a)*dt
    new_p, new_v, new_m, new_r = resolve_collisions(new_p, new_v, mass, r, collisions)
    return new_p, new_v, new_m, new_r

global positions, velocities, masses
positions = np.random.normal(size=(N, 3), scale=scale)
#velocities = np.zeros((N, 3))
velocities = np.random.normal(size=(N, 3), scale=0.01)
masses = np.full((N), mass)
radii = np.full((N), rad)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=positions, face_color='w', size=10, edge_width=0)
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(positions, axis=0)

def update(ev):
    global positions, velocities, masses, radii
    positions, velocities, masses, radii = verlet_step(positions, velocities, radii, masses, dt)
    scatter.set_data(positions)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

### TO DO:
# Make Collision Resolution Function