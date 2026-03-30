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

r = np.random.normal(size=(N, 3), scale=scale)
rho = np.full((N), rho0)
v = np.random.normal(size=(N, 3), scale=v_scale)

def collect_interactions():
    
    
    return interactions, interaction_count, grav_masses, grav_coms, grav_count

def fij_interact():
    

def fij_grav():


def density():
    

def verlet_step(r, rho, v, dt):
    
    

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=r, face_color='w', size=1, edge_width=0, scaling='scene')
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.center = np.mean(r, axis=0)

def update(ev):
    r, rho, v = verlet_step(r, rho, v, dt)
    scatter.set_data(r, face_color='w', size=1, edge_width=0)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()

# equations and stuff is on cornell sph pdf
# use barnes-hut or RT cores to handle nearest neighbor searches
# just use vispy scatter for now
# use barnes-hut logic from n-body to compute gravity