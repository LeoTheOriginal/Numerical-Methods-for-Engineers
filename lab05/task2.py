import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Constants
delta = 0.2
nx, ny = 128, 128
xmax, ymax = nx * delta, ny * delta
TOL = 1e-8


# Boundary condition functions
def VB1(x): return +1 * np.sin(np.pi * x / xmax)      # Bottom edge y=0
def VB2(y): return -1 * np.sin(2 * np.pi * y / ymax)  # Right edge x=xmax
def VB3(x): return +1 * np.sin(np.pi * x / xmax)      # Top edge y=ymax
def VB4(y): return +1 * np.sin(2 * np.pi * y / ymax)  # Left edge x=0

# Set boundary conditions
def set_boundary_conditions(V, x, y):
    V[0, :] = VB1(x)        # Bottom edge y=0
    V[-1, :] = VB3(x)       # Top edge y=ymax
    V[:, 0] = VB4(y)        # Left edge x=0
    V[:, -1] = VB2(y)       # Right edge x=xmax

# Functional integral S(k)
@njit
def calculate_S(V, k, delta):
    S = 0.0
    ny, nx = V.shape[0] - 1, V.shape[1] - 1
    for i in range(0, ny, k):
        for j in range(0, nx, k):
            if i + k <= ny and j + k <= nx:
                term1 = ((V[i + k, j] - V[i, j]) + (V[i + k, j + k] - V[i, j + k])) / (2 * k * delta)
                term2 = ((V[i, j + k] - V[i, j]) + (V[i + k, j + k] - V[i + k, j])) / (2 * k * delta)
                S += (k * delta)**2 * 0.5 * (term1**2 + term2**2)
    return S

# Relaxation function
@njit
def relax(V, k, delta, TOL, max_iter=1000):
    ny, nx = V.shape[0] - 1, V.shape[1] - 1
    iterations = 0
    max_iterations = max_iter
    S_history = np.zeros(max_iterations)
    S_prev = calculate_S(V, k, delta)
    for iteration in range(max_iterations):
        iterations += 1
        # Relaxation step
        for i in range(k, ny, k):
            for j in range(k, nx, k):
                if i + k <= ny and j + k <= nx:
                    V[i,j] = 0.25 * (V[i + k, j] + V[i - k, j] + V[i, j + k] + V[i, j - k])
        # Calculate S
        S_current = calculate_S(V, k, delta)
        S_history[iteration] = S_current
        # Stopping condition
        if iteration > 0:
            if abs(S_current - S_prev) / S_prev < TOL:
                S_history = S_history[:iteration + 1]
                break
        S_prev = S_current
    return V, iterations, S_history

# Grid refinement function
@njit
def refine_grid(V, k):
    ny, nx = V.shape[1] - 1, V.shape[0] - 1
    for i in range(0, ny, k):
        for j in range(0, nx, k):
            if i + k <= ny and j + k <= nx:
                mid_i = i + k // 2
                mid_j = j + k // 2
                # Interpolate interior points
                V[mid_i, mid_j] = 0.25 * (V[i, j] + V[i + k, j] + V[i, j + k] + V[i + k, j + k])
                # Interpolate edges
                V[mid_i, j] = 0.5 * (V[i, j] + V[i + k, j])
                V[mid_i, j + k] = 0.5 * (V[i, j + k] + V[i + k, j + k])
                V[i, mid_j] = 0.5 * (V[i, j] + V[i, j + k])
                V[i + k, mid_j] = 0.5 * (V[i + k, j] + V[i + k, j + k])
    return V

# Main script
x = np.linspace(0, xmax, nx + 1)
y = np.linspace(0, ymax, ny + 1)
V = np.zeros((ny + 1, nx + 1))  # V[y, x]
set_boundary_conditions(V, x, y)

S_data = {}
V_data = {}
iterations_data = {}
k_values = [16, 8, 4, 2, 1]


for k in k_values:
    print(f"Starting relaxation for k = {k}")
    V, iterations, S_history = relax(V, k, delta, TOL)
    V_data[k] = V.copy()
    S_data[k] = S_history
    iterations_data[k] = iterations
    if k > 1:
        V = refine_grid(V, k)
        x = np.linspace(0, xmax, V.shape[1])
        y = np.linspace(0, ymax, V.shape[0])
        set_boundary_conditions(V, x, y)

print(f"Relaxations completed with a total number of iterations: {sum(iterations_data[k] for k in k_values)}")

# Plotting the functional integral S(k)
plt.figure(figsize=(10, 6))
iteration_offset = 0
for k in k_values:
    iterations = len(S_data[k])
    plt.plot(
        np.arange(iteration_offset, iteration_offset + iterations),
        S_data[k],
        label=f'k = {k} [{iteration_offset + 1}-{iteration_offset + iterations}]'
    )
    iteration_offset += iterations
plt.xlabel('Global Iterations')
plt.ylabel('S(k)')
plt.title('Change of functional integral S(k) over global iterations')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the potential maps
for k in k_values:
    V = V_data[k]
    plt.figure()
    plt.imshow(V, extent=(0, xmax, 0, ymax), origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Potential V(x, y)')
    plt.title(f'Potential map for k = {k}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
