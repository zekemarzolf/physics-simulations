import sys
import numpy as np
from numba import njit
from vispy import app, scene
from vispy.scene import visuals
from Constants import Gravity as g
from vispy.visuals.transforms import STTransform

@njit
def build_octree(positions, masses, node_centers, node_sizes, node_masses, node_coms, node_children, node_particle, n_nodes):

    # compute morton codes
    m_positions = np.zeros(N, dType=np.uint64)
    for idx, pos in positions:
        code=0
        for j in range(16):
            code |= ((pos[0] & (1 << j)) << (2*j)) | \
                    ((pos[1] & (1 << j)) << (2*j+1)) | \
                    ((pos[2] & (1 << j)) << (2*j+2))
        m_positions[idx] = code
    
    node_centers[0,:] = 0.5
    node_sizes[0]   = 1.0
    node_masses[0]  = 0.0
    node_coms[0,:] = 0.0
    node_children[0] = -1   # no children yet
    node_particle[0] = -1   # not a leaf yet
    n_nodes[0] = 1
    
    

def acc_calc(p, mass):
    displacements = p[None,:,:] - p[:,None,:]
    sqDist = np.sum(displacements**2, axis=2) + eps**2
    r3 = sqDist**1.5
    np.fill_diagonal(r3, np.inf)
    massProd = g.G * mass[None,:]
    accelerations = np.sum((massProd/r3)[...,None]*displacements, axis=1)
    return accelerations

def verlet_step(p, v, mass, dt):
    a = acc_calc(p, mass)
    new_p = p + v*dt + 0.5*a*dt**2
    new_a = acc_calc(new_p, mass)
    new_v = v + 0.5*(a + new_a)*dt
    return new_p, new_v

dt = 1
N = 200
eps = .1
global positions, velocities, masses
positions = np.random.normal(size=(N, 3), scale=1)
velocities = np.zeros((N, 3))
#velocities = np.random.normal(size=(N, 3), scale=0.01)
masses = np.full((N), 100000)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=positions, face_color='w', size=10, edge_width=0)
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(positions, axis=0)

def update(ev):
    global positions, velocities, masses
    positions, velocities = verlet_step(positions, velocities, masses, dt)
    scatter.set_data(positions)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

### TO DO:
# Barnes-Hut Implementation
## Morton Codes to Sort
## Octree Hierarchy
## Acceleration Computation via Octree
# Merging