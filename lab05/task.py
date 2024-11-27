import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Constants
DELTA = 0.2
GRID_SIZE_X, GRID_SIZE_Y = 128, 128
X_MAX, Y_MAX = DELTA * GRID_SIZE_X, DELTA * GRID_SIZE_Y
TOLERANCE = 1e-8
K_VALUES = [16, 8, 4, 2, 1]
MAX_ITERATIONS = 10000  # Maximum iterations for the relaxation loop

# Grid setup
x_values = np.linspace(0, X_MAX, GRID_SIZE_X + 1)
y_values = np.linspace(0, Y_MAX, GRID_SIZE_Y + 1)


# Boundary conditions
def boundary_condition_left(y_l):
    return np.sin(np.pi * y_l / Y_MAX)


def boundary_condition_right(y_r):
    return np.sin(np.pi * y_r / Y_MAX)


def boundary_condition_top(x_t):
    return -np.sin(2 * np.pi * x_t / X_MAX)


def boundary_condition_bottom(x_b):
    return np.sin(2 * np.pi * x_b / X_MAX)


def initialize_potential(grid_size_x, grid_size_y):
    potential_init = np.zeros((grid_size_y + 1, grid_size_x + 1))
    potential_init[:, 0] = boundary_condition_left(y_values)
    potential_init[:, -1] = boundary_condition_right(y_values)
    potential_init[-1, :] = boundary_condition_top(x_values)
    potential_init[0, :] = boundary_condition_bottom(x_values)
    return potential_init


def set_boundary_conditions(potential_set, x_set, y_set):
    potential_set[:, 0] = boundary_condition_left(y_set)
    potential_set[:, -1] = boundary_condition_right(y_set)
    potential_set[-1, :] = boundary_condition_top(x_set)
    potential_set[0, :] = boundary_condition_bottom(x_set)


@njit
def calculate_s(potential_s, k_s, delta):
    functional_S = 0.0
    ny, nx = potential_s.shape[0] - 1, potential_s.shape[1] - 1
    for i in range(0, ny, k_s):
        for j in range(0, nx, k_s):
            term1 = (potential_s[i + k_s, j] - potential_s[i, j] +
                     potential_s[i + k_s, j + k_s] - potential_s[i, j + k_s])
            term2 = (potential_s[i, j + k_s] - potential_s[i, j] +
                     potential_s[i + k_s, j + k_s] - potential_s[i + k_s, j])
            functional_S += (term1 / (2 * k_s * delta)) ** 2
            functional_S += (term2 / (2 * k_s * delta)) ** 2
    functional_S *= ((k_s * delta) ** 2) / 2.0
    return functional_S


@njit
def relaxation(potential_relax, k_relax, delta, tolerance, max_iterations):
    s_list = []
    ny, nx = potential_relax.shape[0] - 1, potential_relax.shape[1] - 1

    for iteration in range(max_iterations):
        # Relaxation step
        for i in range(k_relax, ny, k_relax):
            for j in range(k_relax, nx, k_relax):
                potential_relax[i, j] = 0.25 * (
                        potential_relax[i + k_relax, j] + potential_relax[i - k_relax, j] +
                        potential_relax[i, j + k_relax] + potential_relax[i, j - k_relax])

        # Calculate functional S
        functional_S = calculate_s(potential_relax, k_relax, delta)
        s_list.append(functional_S)

        # Stopping condition
        if len(s_list) > 1 and abs((s_list[-1] - s_list[-2]) / s_list[-2]) < tolerance:
            break

    return potential_relax, s_list


@njit
def refine_grid(potential_grid, k_grid):
    ny, nx = potential_grid.shape[0] - 1, potential_grid.shape[1] - 1
    for i in range(0, ny, k_grid):
        for j in range(0, nx, k_grid):
            if i + k_grid <= ny and j + k_grid <= nx:
                mid_i, mid_j = i + k_grid // 2, j + k_grid // 2
                potential_grid[mid_i, mid_j] = 0.25 * (
                        potential_grid[i, j] + potential_grid[i + k_grid, j] +
                        potential_grid[i, j + k_grid] + potential_grid[i + k_grid, j + k_grid])
                potential_grid[mid_i, j] = 0.5 * (potential_grid[i, j] + potential_grid[i + k_grid, j])
                potential_grid[mid_i, j + k_grid] \
                    = 0.5 * (potential_grid[i, j + k_grid] + potential_grid[i + k_grid, j + k_grid])
                potential_grid[i, mid_j] = 0.5 * (potential_grid[i, j] + potential_grid[i, j + k_grid])
                potential_grid[i + k_grid, mid_j] \
                    = 0.5 * (potential_grid[i + k_grid, j] + potential_grid[i + k_grid, j + k_grid])
    return potential_grid


def plot_potential_maps(results_pot_map, x_pot_val, y_pot_val, k_values):
    for index, result in enumerate(results_pot_map):
        potential_map = result["potential"]
        k_i = k_values[index]
        plt.figure()
        plt.title(f"k = {k_i}")
        plt.pcolor(x_pot_val[::k_i], y_pot_val[::k_i], potential_map[::k_i, ::k_i], cmap="jet", shading="auto")
        plt.colorbar(label="V")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def plot_functional_integral(results_integral, k_values):
    plt.figure(figsize=(10, 6))
    total_iterations = 0

    for i, result in enumerate(results_integral):
        k_id = k_values[i]
        S_values = result["S"]
        start, end = total_iterations, total_iterations + len(S_values) - 1
        total_iterations += len(S_values)

        global_iterations = range(start + 1, end + 2)
        plt.plot(global_iterations, S_values, label=f'k = {k_id} [{start + 1}-{end + 1}]')

    plt.xlabel("Global Iterations")
    plt.ylabel("S(k)")
    plt.title("Change of functional integral S(k) over global iterations")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    potential = initialize_potential(GRID_SIZE_X, GRID_SIZE_Y)
    results = []

    for k in K_VALUES:
        print(f"Starting relaxation for k = {k}")
        potential, S_list = relaxation(potential, k, DELTA, TOLERANCE, MAX_ITERATIONS)
        results.append({"potential": potential.copy(), "S": S_list})
        if k > 1:
            potential = refine_grid(potential, k)
            x = np.linspace(0, X_MAX, potential.shape[1])
            y = np.linspace(0, Y_MAX, potential.shape[0])
            set_boundary_conditions(potential, x, y)

    plot_functional_integral(results, K_VALUES)
    plot_potential_maps(results, x_values, y_values, K_VALUES)
