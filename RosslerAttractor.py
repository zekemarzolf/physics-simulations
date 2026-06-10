import numpy as np
import cupy as cp
from vispy import app, scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

N = 200000
dt = 0.01
a = 0.2
b = 0.2
c = 14

positions = cp.random.normal(size=(N, 3), scale=0.1) + cp.array([[-5.0, 0.0, 0.0]])

def timestep(positions, dt):
    x_velocities = (positions[:, 1] + positions[:, 2]) * -1
    y_velocities = positions[:, 0] + a * positions[:, 1]
    z_velocities = b + positions[:, 2] * (positions[:, 0] - c)
    
    new_positions = cp.copy(positions)
    new_positions[:, 0] += x_velocities*dt
    new_positions[:, 1] += y_velocities*dt
    new_positions[:, 2] += z_velocities*dt
    
    return new_positions

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers(pos=cp.asnumpy(positions), face_color='white', size=0.1, edge_width=0, scaling='scene')
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(1, 1, 1))
view.add(scatter)
view.camera = 'turntable'
view.camera.distance = 50
view.camera.center = (0, 0, 10)

def update(ev):
    global positions
    positions = timestep(positions, dt)
    scatter.set_data(cp.asnumpy(positions), face_color='white', size=0.1, edge_width=0)
    view.camera.center = (0, 0, 10)

timer = app.Timer(interval=.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()