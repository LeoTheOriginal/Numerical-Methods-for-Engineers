import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit


@njit
def picard_method(initial_infected, beta=0.001, gamma=0.1, n=500, time_step=0.1, max_time=100, tolerance=1e-6,
                  max_iterations=20):
    """Solution using the trapezoidal method with Picard iteration."""
    steps, time, infected, iterations, alpha = initialize_simulation(
        initial_infected, beta, gamma, n, time_step, max_time
    )

    for n in range(steps):
        infected_init = infected[n]
        infected_new = infected_init

        for i in range(max_iterations):
            next_infected = infected[n] + (time_step / 2) * (
                    (alpha * infected[n] - beta * infected[n] ** 2) +
                    (alpha * infected_new - beta * infected_new ** 2)
            )

            if math.fabs(next_infected - infected_new) < tolerance:
                iterations[n] = i
                break

            infected_new = next_infected

        infected[n + 1] = infected_new

    return time, infected, iterations


@njit
def newton_method(initial_infected, beta=0.001, gamma=0.1, n=500, time_step=0.1, max_time=100, tolerance=1e-6,
                  max_iterations=20):
    """Solution using the trapezoidal method with Newton iteration."""
    steps, time, infected, iterations, alpha = initialize_simulation(
        initial_infected, beta, gamma, n, time_step, max_time
    )

    for n in range(steps):
        infected_new = infected[n]

        for i in range(max_iterations):
            f = infected_new - infected[n] - (time_step / 2) * (
                    (alpha * infected[n] - beta * infected[n] ** 2) +
                    (alpha * infected_new - beta * infected_new ** 2)
            )
            d_f = 1 - (time_step / 2) * (alpha - 2 * beta * infected_new)

            delta = f / d_f
            next_infected = infected_new - delta

            if math.fabs(delta) < tolerance:
                iterations[n] = i
                break

            infected_new = next_infected

        infected[n + 1] = infected_new

    return time, infected, iterations


@njit
def rk2_method(initial_infected, beta=0.001, gamma=0.1, n=500, time_step=0.1, max_time=100, tolerance=1e-6,
               max_iterations=20):
    """Solution using the implicit RK2 method."""
    c1 = 1 / 2 - np.sqrt(3) / 6
    c2 = 1 / 2 + np.sqrt(3) / 6
    a11 = 1 / 4
    a12 = 1 / 4 - np.sqrt(3) / 6
    a21 = 1 / 4 + np.sqrt(3) / 6
    a22 = 1 / 4
    b1 = 1 / 2
    b2 = 1 / 2

    steps, time, infected, iterations, alpha = initialize_simulation(
        initial_infected, beta, gamma, n, time_step, max_time
    )

    for n in range(steps):
        u1 = infected[n]
        u2 = infected[n]

        for i in range(max_iterations):
            f1 = u1 - infected[n] - time_step * (
                    a11 * (alpha * u1 - beta * u1 ** 2) +
                    a12 * (alpha * u2 - beta * u2 ** 2)
            )
            f2 = u2 - infected[n] - time_step * (
                    a21 * (alpha * u1 - beta * u1 ** 2) +
                    a22 * (alpha * u2 - beta * u2 ** 2)
            )

            m11 = 1 - time_step * a11 * (alpha - 2 * beta * u1)
            m12 = -time_step * a12 * (alpha - 2 * beta * u2)
            m21 = -time_step * a21 * (alpha - 2 * beta * u1)
            m22 = 1 - time_step * a22 * (alpha - 2 * beta * u2)

            det = m11 * m22 - m12 * m21
            d_u1 = (f2 * m12 - f1 * m22) / det
            d_u2 = (f1 * m21 - f2 * m11) / det

            u1_next = u1 + d_u1
            u2_next = u2 + d_u2

            if max(math.fabs(d_u1), math.fabs(d_u2)) < tolerance:
                iterations[n] = i
                break

            u1 = u1_next
            u2 = u2_next

        infected[n + 1] = infected[n] + time_step * (
                b1 * (alpha * u1 - beta * u1 ** 2) +
                b1 * (alpha * u2 - beta * u2 ** 2)
        )

    return time, infected, iterations


@njit
def initialize_simulation(initial_infected, beta, gamma, n, time_step, max_time):
    """Initialize common parameters for the simulation."""
    steps = int(max_time / time_step)
    time = np.linspace(0, max_time, steps + 1)
    infected = np.zeros(steps + 1)
    iterations = np.zeros(steps)
    infected[0] = initial_infected
    alpha = beta * n - gamma
    return steps, time, infected, iterations, alpha


def plot_population(ax, time, infected, n, method_name):
    """Plot population for the infected and susceptible groups."""
    ax.plot(time, infected, 'b-', label='Infected (u)')
    ax.plot(time, n - infected, 'r-', label='Susceptible (z)')
    ax.set_title(f'{method_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend()
    ax.grid(True)


def plot_iterations(ax, time, iterations, method_name):
    """Plot iterations over time for a given method."""
    ax.plot(time[:-1], iterations, 'g-', label='Iterations')
    ax.set_title(f'Iterations over Time - {method_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Iterations')
    ax.legend()
    ax.grid(True)


initial_infected = 1
N = 500

time_p, infected_p, iterations_p = picard_method(initial_infected)
time_n, infected_n, iterations_n = newton_method(initial_infected)
time_r, infected_r, iterations_r = rk2_method(initial_infected)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

plot_population(ax1, time_p, infected_p, N, 'Trapezoidal Method with Picard Iteration')
plot_population(ax2, time_n, infected_n, N, 'Trapezoidal Method with Newton Iteration')
plot_population(ax3, time_r, infected_r, N, 'Implicit RK2 Method')

plt.tight_layout()
plt.savefig('solution_population.png', dpi=300, bbox_inches='tight')

fig_iter, (ax_iter1, ax_iter2, ax_iter3) = plt.subplots(3, 1, figsize=(10, 12))

plot_iterations(ax_iter1, time_p, iterations_p, 'Picard Method')
plot_iterations(ax_iter2, time_n, iterations_n, 'Newton Method')
plot_iterations(ax_iter3, time_r, iterations_r, 'RK2 Method')

fig_iter.tight_layout()
fig_iter.savefig('solution_iterations.png', dpi=300, bbox_inches='tight')

plt.close()
