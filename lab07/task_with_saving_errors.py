import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Parametry symulacji
delta = 0.01
rho = 1.0
mi = 1.0
nx = 200
ny = 90
i1 = 50
j1 = 55
j2 = j1 + 2
IT_MAX = 20000

# Przedziały
y_ny = delta * ny
y_j1 = delta * j1

# Siatka
x = np.linspace(0, nx * delta, nx + 1)
y = np.linspace(0, ny * delta, ny + 1)


@njit
def compute_Qwy(Q, y_ny, y_j1, mi):
    return Q * (y_ny ** 3 - y_j1 ** 3 - 3 * y_j1 * y_ny ** 2 + 3 * (y_j1 ** 2) * y_ny) / (y_ny ** 3)


@njit
def set_WB_psi(psi, Q, Qwy, delta, mi, i1, j1, nx, ny):
    # Brzeg A
    for j in range(j1, ny + 1):
        y_val = j * delta
        psi[0, j] = Q / (2 * mi) * (y_val ** 3 / 3 - (y_val ** 2 / 2) * (y_j1 + y_ny) + y_val * y_j1 * y_ny)

    # Brzeg C
    for j in range(0, ny + 1):
        y_val = j * delta
        psi[nx, j] = (Qwy / (2 * mi)) * (y_val ** 3 / 3 - (y_val ** 2 * (y_ny / 2))) + (
                    (Q * y_j1 ** 2 * (-y_j1 + 3 * y_ny)) / (12 * mi))

    # Brzeg B
    for i in range(1, nx):
        psi[i, ny] = psi[0, ny]

    # Brzeg D
    for i in range(i1, nx):
        psi[i, 0] = psi[0, j1]

    # Brzeg E
    for j in range(1, j1 + 1):
        psi[i1, j] = psi[0, j1]

    # Brzeg F
    for i in range(1, i1 + 1):
        psi[i, j1] = psi[0, j1]


@njit
def set_WB_zeta(zeta, psi, Q, Qwy, delta, mi, i1, j1, nx, ny):
    # Brzeg A
    for j in range(j1, ny + 1):
        y_val = j * delta
        zeta[0, j] = (Q / (2 * mi)) * (2 * y_val - y_j1 - y_ny)

    # Brzeg C
    for j in range(0, ny + 1):
        y_val = j * delta
        zeta[nx, j] = (Qwy / (2 * mi)) * (2 * y_val - y_ny)

    # Brzeg B
    for i in range(1, nx):
        zeta[i, ny] = 2 / (delta ** 2) * (psi[i, ny - 1] - psi[i, ny])

    # Brzeg D
    for i in range(i1 + 1, nx):
        zeta[i, 0] = 2 / (delta ** 2) * (psi[i, 1] - psi[i, 0])

    # Brzeg E
    for j in range(1, j1):
        zeta[i1, j] = 2 / (delta ** 2) * (psi[i1 + 1, j] - psi[i1, j])

    # Brzeg F
    for i in range(1, i1 + 1):
        zeta[i, j1] = 2 / (delta ** 2) * (psi[i, j1 + 1] - psi[i, j1])

    # Wierzchołek E/F
    zeta[i1, j1] = 0.5 * (zeta[i1 - 1, j1] + zeta[i1, j1 - 1])


@njit
def is_boundary(i, j, i1, j1, nx, ny):
    return (
            (i == 0 and j1 <= j <= ny) or  # A
            (j == ny) or  # B
            (i == nx) or  # C
            (i1 <= i <= nx and j == 0) or  # D
            (0 <= i <= i1 and j == j1) or  # F
            (i == i1 and 0 <= j <= j1)  # E
    )


@njit
def relaxation(Q, delta, rho, mi, i1, j1, j2, nx, ny, IT_MAX, y_ny, y_j1):
    # Inicjalizacja pól
    u = np.zeros((nx + 1, ny + 1))
    v = np.zeros((nx + 1, ny + 1))
    psi = np.zeros((nx + 1, ny + 1))
    zeta = np.zeros((nx + 1, ny + 1))

    # Ścianka
    for i in range(i1):
        for j in range(j1):
            psi[i, j] = np.nan
            zeta[i, j] = np.nan

    # Oblicz Qwy
    Qwy = compute_Qwy(Q, y_ny, y_j1, mi)

    # Ustaw warunki brzegowe dla psi
    set_WB_psi(psi, Q, Qwy, delta, mi, i1, j1, nx, ny)

    gamma_values = []
    for it in range(1, IT_MAX + 1):
        for i in range(1, nx):
            for j in range(1, ny):
                if not is_boundary(i, j, i1, j1, nx, ny):
                    psi[i, j] = (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1] - (delta ** 2) * zeta[
                        i, j]) / 4.0
                    zeta[i, j] = 0.25 * (zeta[i + 1, j] + zeta[i - 1, j] + zeta[i, j + 1] + zeta[i, j - 1])

                    if it > 2000:
                        zeta[i, j] -= (rho / (16 * mi)) * (
                                (psi[i, j + 1] - psi[i, j - 1]) * (zeta[i + 1, j] - zeta[i - 1, j]) -
                                (psi[i + 1, j] - psi[i - 1, j]) * (zeta[i, j + 1] - zeta[i, j - 1])
                        )

        # Aktualizuj warunki brzegowe dla zeta
        set_WB_zeta(zeta, psi, Q, Qwy, delta, mi, i1, j1, nx, ny)

        # Oblicz gamma (błąd)
        gamma = 0.0
        for i in range(1, nx):
            gamma += (
                    psi[i + 1, j2] + psi[i - 1, j2] +
                    psi[i, j2 + 1] + psi[i, j2 - 1] -
                    4 * psi[i, j2] - (delta ** 2) * zeta[i, j2]
            )
        gamma_values.append(gamma)

    # Oblicz prędkości u i v
    for i in range(1, nx):
        for j in range(1, ny):
            if i > i1 or j > j1:
                u[i, j] = (psi[i, j + 1] - psi[i, j - 1]) / (2.0 * delta)
                v[i, j] = -(psi[i + 1, j] - psi[i - 1, j]) / (2.0 * delta)

    return [u, v, psi, zeta], gamma_values


def create_contour_plot(x, y, data, title, cmap, levels, label):
    plt.figure(figsize=(12, 6))
    cp = plt.contour(
        x,
        y,
        np.transpose(data),
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
        cmap=cmap,
        levels=levels
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(cp)
    plt.title(title)
    plt.show()


def create_pcolor_plot(x, y, data, title, cmap, label):
    plt.figure(figsize=(12, 6))
    pc = plt.pcolor(
        x,
        y,
        np.transpose(data),
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
        shading="auto",
        cmap=cmap
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(pc)
    plt.title(title)
    plt.show()


def save_errors(gamma_values, Q):
    with open(f"error_Q={Q}.txt", "w") as f:
        for it, val in enumerate(gamma_values, 1):
            f.write(f"Iteracja={it}, error: {val}\n")


def plot_results(results, errors, Q):
    u, v, psi, zeta = results
    gamma_values = errors

    # Tytuły wykresów
    titles = {
        'psi': f"Psi dla Q = {Q}",
        'zeta': f"Zeta dla Q = {Q}",
        'u': f"Prędkość u dla Q = {Q}",
        'v': f"Prędkość v dla Q = {Q}"
    }

    # Rysowanie psi
    print(f"Wyświetlanie wykresu psi dla Q = {Q}")
    create_contour_plot(x, y, psi, titles['psi'], "gnuplot", 50, "Psi")

    # Rysowanie zeta
    print(f"Wyświetlanie wykresu zeta dla Q = {Q}")
    create_contour_plot(x, y, zeta, titles['zeta'], "gnuplot", 100, "Zeta")

    # Rysowanie u
    print(f"Wyświetlanie wykresu u dla Q = {Q}")
    create_pcolor_plot(x, y, u, titles['u'], "jet", "u")

    # Rysowanie v
    print(f"Wyświetlanie wykresu v dla Q = {Q}")
    create_pcolor_plot(x, y, v, titles['v'], "jet", "v")

    # Zapisywanie wartości błędów do pliku
    print(f"Zapisywanie wartości błędów dla Q = {Q}")
    save_errors(gamma_values, Q)


def main():
    # Definicje wartości Q
    Q_values = [-1000, -4000, 4000]

    # Uruchomienie symulacji dla każdej wartości Q
    for Q in Q_values:
        print(f"Rozpoczynanie relaksacji dla Q = {Q}")
        results, errors = relaxation(Q, delta, rho, mi, i1, j1, j2, nx, ny, IT_MAX, y_ny, y_j1)
        plot_results(results, errors, Q)


if __name__ == "__main__":
    main()
