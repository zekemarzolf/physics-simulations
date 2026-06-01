import numpy as np

conductivity = 1
rho = 1
specific_heat = 1
temperatures = np.full(1000, 20, dtype=np.float64)

def laplacian(temperatures):
    F_temps = np.fft.fft(temperatures)
    N = int(temperatures.shape)
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0)
    F_laplacian = -1 * (k**2) * F_temps
    real_laplacian = np.fft.ifft(F_laplacian).real
    return real_laplacian

def timestep(temperatures):
    alpha = conductivity / (rho * specific_heat)
    dT = alpha * laplacian(temperatures)