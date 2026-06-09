import sys
import numpy as np
from vispy import plot as vp
from vispy import scene, app, color
from scipy.linalg import solve_banded

dt = 1
N = 1000
conductivity = 1
rho = 1
specific_heat = 1
alpha = conductivity / (rho * specific_heat)
CFL = (alpha * dt) / 2
global temperatures
rng = np.random.default_rng()
#temperatures = rng.random(size=1000, dtype=np.float32) * 100
temperatures = np.full(N, 20, dtype=np.float32)
temperatures[450:550] = 500.0
temperatures[0:60] = 200.0
temperatures[200:210] = 1000.0
temperatures[800:830] = 80.0

ab = np.array([np.append(0, -CFL * np.ones(N-1)), 
                (1 + 2 * CFL) * np.ones(N),
                np.append(-CFL * np.ones(N-1), 0)])

def spectral_laplacian(temperatures):
    F_temps = np.fft.fft(temperatures)
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0)
    F_temps_new = np.exp(-(k**2) * alpha * dt) * F_temps
    return np.fft.ifft(F_temps_new).real

def crank_nicolson(temperatures):
    b = CFL * temperatures[:-2] + (1 - 2 * CFL) * temperatures[1:-1] + CFL * temperatures[2:]
    b = np.insert(b, 0, 0)
    b = np.append(b, 0)
    u = temperatures.copy()
    u = solve_banded((1,1), ab, b)
    return u

colors = ['blue', 'red', 'pink', 'white']
controls = [0.0, 0.5, 0.75, 1.0]
cmap = color.Colormap(colors, controls=controls)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 200), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image = scene.visuals.Image(temperatures[np.newaxis, :].astype(np.float32), cmap=cmap, clim=(0.0, 1000.0), parent=view.scene)
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, len(temperatures)), y=(0, 1))

def update(ev):
    global temperatures
    #temperatures = crank_nicolson(temperatures)
    temperatures = spectral_laplacian(temperatures)
    image.set_data(temperatures[np.newaxis, :].astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()