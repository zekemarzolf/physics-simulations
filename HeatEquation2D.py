import sys
import numpy as np
from vispy import plot as vp
from vispy import scene, app, color

dt = 0.1
N = 1000
conductivity = 1
rho = 1
specific_heat = 1
global temperatures
rng = np.random.default_rng()
#temperatures = rng.random(size=(N,N), dtype=np.float32) * 100
temperatures = np.full((N,N), 20, dtype=np.float32)
temperatures[0, :] = 10000.0

def spectral_laplacian(temperatures):
    F_temps = np.fft.fft2(temperatures)
    Ny, Nx = temperatures.shape
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    K2 = (2 * np.pi * KX)**2 + (2 * np.pi * KY)**2
    F_laplacian = -K2 * F_temps
    real_laplacian = np.fft.ifft2(F_laplacian).real
    return real_laplacian

def finite_gradient(temperatures):
    gradient = np.empty((N), dtype=np.float32)
    for i in range(temperatures.shape[0]):
        if i == 0:
            dT = temperatures[1] - temperatures[0]
        elif i == N-1:
            dT = temperatures[i] - temperatures[i-1]
        else:
            dT = temperatures[i+1] - temperatures[i-1]
        gradient[i] = dT/2
    return gradient

def finite_laplacian(temperatures):
    gradient = finite_gradient(temperatures)
    laplacian = finite_gradient(gradient)
    return laplacian


colors = ['blue', 'red', 'white']
controls = [0.0, 0.1, 1.0]
cmap = color.Colormap(colors, controls=controls)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image = scene.visuals.Image(temperatures.astype(np.float32), cmap=cmap, clim=(0.0, 1000.0), parent=view.scene)
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, temperatures.shape[0]), y=(0, temperatures.shape[1]))
colorbar = scene.widgets.ColorBarWidget(cmap=cmap, orientation='bottom', clim = (0.0, 1000.0), border_width=1.0)

def update(ev):
    global temperatures
    alpha = conductivity / (rho * specific_heat)
    dT = alpha * spectral_laplacian(temperatures)
    temperatures = temperatures + dT * dt
    image.set_data(temperatures.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()