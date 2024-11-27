import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

epsilon = 1.0
delta = 0.1
nx, ny = 151, 101
xmax, ymax = delta * (nx - 1), delta * (ny - 1)
V1_val, V2_val = 10.0, 0.0
TOL = 1e-8

sigma_x = 0.1 * xmax
sigma_y = 0.1 * ymax


def rho(x, y):
    rho1 = (+1) * np.exp(-((x - 0.35 * xmax) ** 2 / sigma_x ** 2) - ((y - 0.5 * ymax) ** 2 / sigma_y ** 2))
    rho2 = (-1) * np.exp(-((x - 0.65 * xmax) ** 2 / sigma_x ** 2) - ((y - 0.5 * ymax) ** 2 / sigma_y ** 2))
    return rho1 + rho2


x = np.linspace(0, xmax, nx)
y = np.linspace(0, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
rho_grid = rho(X, Y)


@jit(nopython=True)
def calculate_S(V, rho, delta):
    nx, ny = V.shape
    S = 0.0
    for i in range(nx - 1):
        for j in range(ny - 1):
            Ex = (V[i + 1, j] - V[i, j]) / delta
            Ey = (V[i, j + 1] - V[i, j]) / delta
            S += delta ** 2 * (0.5 * (Ex ** 2 + Ey ** 2) - rho[i, j] * V[i, j])
    return S


@jit(nopython=True)
def global_relaxation(V, rho, omega, delta, epsilon, TOL, max_iter=100000):
    nx, ny = V.shape
    S_list = []
    S_prev = calculate_S(V, rho, delta)

    for iteration in range(max_iter):
        V_new = V.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                V_new[i, j] = 0.25 * (
                        V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] + (delta ** 2 / epsilon) * rho[i, j]
                )
        V_new[0, :] = V_new[1, :]
        V_new[-1, :] = V_new[-2, :]

        V = (1 - omega) * V + omega * V_new

        S_current = calculate_S(V, rho, delta)
        S_list.append(S_current)

        if abs(S_current - S_prev) / abs(S_prev) < TOL:
            break
        S_prev = S_current

    return V, S_list


@jit(nopython=True)
def local_relaxation(V, rho, omega, delta, epsilon, TOL, max_iter=100000):
    nx, ny = V.shape
    S_list = []
    S_prev = calculate_S(V, rho, delta)

    for iteration in range(max_iter):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                V_new = 0.25 * (
                        V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] + (delta ** 2 / epsilon) * rho[i, j]
                )
                V[i, j] = (1 - omega) * V[i, j] + omega * V_new

        V[0, :] = V[1, :]
        V[-1, :] = V[-2, :]

        S_current = calculate_S(V, rho, delta)
        S_list.append(S_current)

        if abs(S_current - S_prev) / abs(S_prev + 1e-10) < TOL:
            break
        S_prev = S_current

    return V, S_list


V_initial = np.zeros((nx, ny))
V_initial[:, 0] = V1_val
V_initial[:, -1] = V2_val

V_global_0_6, S_global_0_6 = global_relaxation(
    V_initial.copy(), rho_grid, omega=0.6, delta=delta, epsilon=epsilon, TOL=TOL
)
V_global_1_0, S_global_1_0 = global_relaxation(
    V_initial.copy(), rho_grid, omega=1.0, delta=delta, epsilon=epsilon, TOL=TOL
)

V_local_1_0, S_local_1_0 = local_relaxation(
    V_initial.copy(), rho_grid, omega=1.0, delta=delta, epsilon=epsilon, TOL=TOL
)
V_local_1_4, S_local_1_4 = local_relaxation(
    V_initial.copy(), rho_grid, omega=1.4, delta=delta, epsilon=epsilon, TOL=TOL
)
V_local_1_8, S_local_1_8 = local_relaxation(
    V_initial.copy(), rho_grid, omega=1.8, delta=delta, epsilon=epsilon, TOL=TOL
)
V_local_1_9, S_local_1_9 = local_relaxation(
    V_initial.copy(), rho_grid, omega=1.9, delta=delta, epsilon=epsilon, TOL=TOL
)

iterations_global_0_6 = len(S_global_0_6)
iterations_global_1_0 = len(S_global_1_0)
iterations_local_1_0 = len(S_local_1_0)
iterations_local_1_4 = len(S_local_1_4)
iterations_local_1_8 = len(S_local_1_8)
iterations_local_1_9 = len(S_local_1_9)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, iterations_global_0_6 + 1), S_global_0_6, label=f'ω = 0.6, {iterations_global_0_6} iterations')
plt.plot(range(1, iterations_global_1_0 + 1), S_global_1_0, label=f'ω = 1.0, {iterations_global_1_0} iterations')
plt.xlabel('Iterations')
plt.ylabel('S')
plt.xscale('log')
plt.xlim(1, 100000)
ax1 = plt.gca()
major_ticks = [1, 10, 100, 1000, 10000, 100000]
ax1.xaxis.set_major_locator(FixedLocator(major_ticks))
ax1.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in major_ticks]))
plt.ylim(bottom=0)
plt.legend()
plt.title('Convergence of Function S for Global Relaxation')

plt.subplot(1, 2, 2)
plt.plot(range(1, iterations_local_1_0 + 1), S_local_1_0, label=f'ω = 1.0, {iterations_local_1_0} iterations')
plt.plot(range(1, iterations_local_1_4 + 1), S_local_1_4, label=f'ω = 1.4, {iterations_local_1_4} iterations')
plt.plot(range(1, iterations_local_1_8 + 1), S_local_1_8, label=f'ω = 1.8, {iterations_local_1_8} iterations')
plt.plot(range(1, iterations_local_1_9 + 1), S_local_1_9, label=f'ω = 1.9, {iterations_local_1_9} iterations')
plt.xlabel('Iterations')
plt.ylabel('S')
plt.xscale('log')
plt.xlim(1, 100000)
ax2 = plt.gca()
ax2.xaxis.set_major_locator(FixedLocator(major_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in major_ticks]))
plt.ylim(bottom=0)
plt.legend()
plt.title('Convergence of Function S for Local Relaxation')
plt.tight_layout()
plt.show()


def plot_potential_maps(omega_values, V_results, title_prefix, filename_prefix):
    for omega, V in zip(omega_values, V_results):
        plt.figure(figsize=(8, 6))
        plt.imshow(V.T, extent=(0, xmax, 0, ymax), origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Potential V(x, y)')
        plt.title(f'{title_prefix} (ω = {omega})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{filename_prefix}_omega_{omega}.png")
        plt.show()


def plot_error_maps(omega_values, V_results, rho_grid, title_prefix, filename_prefix):
    nx, ny = V_results[0].shape
    delta_squared = delta ** 2

    x_interior = x[1:-1]
    y_interior = y[1:-1]

    for omega, V in zip(omega_values, V_results):
        laplace_V = (np.roll(V, -1, axis=0) + np.roll(V, 1, axis=0) +
                     np.roll(V, -1, axis=1) + np.roll(V, 1, axis=1) - 4 * V) / delta_squared

        laplace_V_interior = laplace_V[1:-1, 1:-1]
        rho_interior = rho_grid[1:-1, 1:-1]

        error = laplace_V_interior + rho_interior / epsilon

        plt.figure(figsize=(8, 6))
        extent = [x_interior[0], x_interior[-1], y_interior[0], y_interior[-1]]
        plt.imshow(error.T, extent=extent, origin='lower', cmap='seismic', aspect='auto')
        plt.colorbar(label='Error δ')
        plt.title(f'{title_prefix} (ω = {omega})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{filename_prefix}_omega_{omega}.png")
        plt.show()


plot_potential_maps(
    [0.6, 1.0],
    [V_global_0_6, V_global_1_0],
    "Potential Map - Global Relaxation",
    "global_potential"
)

plot_error_maps(
    [0.6, 1.0],
    [V_global_0_6, V_global_1_0],
    rho_grid,
    "Error Map δ - Global Relaxation",
    "global_error"
)
