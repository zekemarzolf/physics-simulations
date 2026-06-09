import numpy as np
from vispy import plot as vp
from vispy import scene, app, color
from scipy import ndimage

dt = 10
N = 1000
conductivity = 1
rho = 1
specific_heat = 1
alpha = conductivity / (rho * specific_heat)
global temperatures
rng = np.random.default_rng()
#temperatures = rng.random(size=(N,N), dtype=np.float32) * 100
temperatures = np.full((N,N), 20, dtype=np.float32)
#temperatures[0, :] = 10000.0
temperatures[450:550, 450:550] = 1000.0

def random_surface():
    surface = np.full((N,N), 1, dtype=np.float32)
    for y in range(N):
        for x in range(N):
            z = np.modf(np.sin(12.9898*(x/100) + 78.233*(y/100)) * 43758.5453)[0]
            surface[y,x] = abs(z) * 1000
    return surface

#temperatures = random_surface()

def spectral_laplacian(temperatures):
    F_temps = np.fft.fft2(temperatures)
    Ny, Nx = temperatures.shape
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    K2 = (2 * np.pi * KX)**2 + (2 * np.pi * KY)**2
    F_new_temps = np.exp(alpha * -K2 * dt) * F_temps
    new_temps = np.fft.ifft2(F_new_temps).real
    return new_temps

colors = ['blue', 'red', 'pink']
controls = [0.0, 0.5, 1.0]
#colors = ['black', 'white']
#controls = [0.0, 1.0]
cmap = color.Colormap(colors, controls=controls)

canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=0)
image = scene.visuals.Image(temperatures.astype(np.float32), cmap='viridis', clim=(0.0, 1000.0), parent=view.scene)
view.camera = scene.cameras.PanZoomCamera(aspect=None)
view.camera.set_range(x=(0, temperatures.shape[0]), y=(0, temperatures.shape[1]))

def update(ev):
    global temperatures
    temperatures = spectral_laplacian(temperatures)
    image.set_data(temperatures.astype(np.float32))
    canvas.update()

timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()