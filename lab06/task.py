import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numba import njit

DELTA = 0.1
SIGMA_SCALE = 10
V1_DEFAULT = 10.0
V2_DEFAULT = -10.0
V3_DEFAULT = 10.0
V4_DEFAULT = -10.0
VMIN_DEFAULT = -10
VMAX_DEFAULT = 10


@njit
def rho1(i, j, xmax, ymax, sigma):
    return np.exp(-(((i * DELTA - 0.25 * xmax) / sigma) ** 2) - (((j * DELTA - 0.5 * ymax) / sigma) ** 2))


@njit
def rho2(i, j, xmax, ymax, sigma):
    return -np.exp(-(((i * DELTA - 0.75 * xmax) / sigma) ** 2) - (((j * DELTA - 0.5 * ymax) / sigma) ** 2))


@njit
def rho(i, j, xmax, ymax, sigma):
    return rho1(i, j, xmax, ymax, sigma) + rho2(i, j, xmax, ymax, sigma)


def build_sparse_matrix_and_rhs(nx, ny, epsilon1, epsilon2, v1, v2, v3, v4, use_function_rho, xmax, ymax, sigma):
    N = (nx + 1) * (ny + 1)
    max_nonzeros = 5 * N
    a = np.zeros(max_nonzeros)
    ja = np.zeros(max_nonzeros, dtype=int)
    ia = np.zeros(N + 1, dtype=int)
    b = np.zeros(N)

    k = 0
    for i in range(N):
        ix = i % (nx + 1)
        jy = i // (nx + 1)
        edge = False    #wskaznik położenia: 0 - srodek obszaru; 1- brzeg
        vb = 0.0    #potencjal na brzegu

        if ix == 0:  #lewy brzeg
            edge = True
            vb = v1
        elif ix == nx:  #prawy brzeg
            edge = True
            vb = v3
        elif jy == 0:  #dolny brzeg
            edge = True
            vb = v4
        elif jy == ny:  #górny brzeg
            edge = True
            vb = v2

        ia[i] = k   #początek nowego wiersza

        if edge:
            b[i] = vb   #wymoszamy wartosc potencjalu na brzegu
            a[k] = 1.0
            ja[k] = i
            k += 1
        else:   #dla punktów wewnętrznych
            if use_function_rho:
                b[i] = -rho(ix, jy, xmax, ymax, sigma)  #wypełniamy od razu wektor wyrazów wolnyc
                                                        # h jeśli w środku jest gęstość

            sum_epsilon = 0.0
            epsilon_c = epsilon1 if ix <= nx // 2 else epsilon2

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx_pos = ix + dx
                ny_pos = jy + dy

                if 0 <= nx_pos <= nx and 0 <= ny_pos <= ny:
                    m = i + dx + dy * (nx + 1)
                    epsilon_n = epsilon1 if nx_pos <= nx // 2 else epsilon2
                    epsilon_avg = (epsilon_c + epsilon_n) / 2.0
                    coeff = epsilon_avg / (DELTA ** 2)

                    a[k] = coeff
                    ja[k] = m
                    k += 1
                    sum_epsilon += coeff

            a[k] = -sum_epsilon  #współczynniki dla aktualnego punktu
            ja[k] = i
            k += 1

    ia[N] = k   #koniec wiersza
    return csr_matrix((a[:k], ja[:k], ia), shape=(N, N)), b


def poisson(nx, ny, epsilon1, epsilon2, v1, v2, v3, v4, use_function_rho, title, vmin, vmax):
    xmax = DELTA * nx
    ymax = DELTA * ny
    sigma = xmax / SIGMA_SCALE

    A, b = build_sparse_matrix_and_rhs(nx, ny, epsilon1, epsilon2, v1, v2, v3, v4, use_function_rho, xmax, ymax, sigma)

    if nx == 4 :
        save_to_file(A, b, nx, ny, filename=f"output_nx{nx}_ny{ny}.dat")

    V = spsolve(A, b)

    V = V.reshape((ny + 1, nx + 1))

    plt.figure(figsize=(8, 6))
    img = plt.imshow(V, extent=(0, xmax, 0, ymax), origin='lower', cmap='bwr', vmin=vmin, vmax=vmax, aspect='equal')
    plt.colorbar(img, label='V')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def save_to_file(A, b, nx, ny, filename="output.dat"):
    with open(filename, "w") as f:

        f.write("Macierz b\n\n")
        f.write("# l\ti\tj\tb[l]\n")
        for idx, val in enumerate(b):
            i = idx % (nx + 1)
            j = idx // (nx + 1)
            f.write(f"{idx}\t{i}\t{j}\t{val:.1f}\n")
            if (idx + 1) % 5 == 0:
                f.write(f"\n")

        f.write("\nMacierz A\n\n")
        f.write("# k\ta[k]\n")
        A_coo = A.tocoo()
        for k, (val, i, j) in enumerate(zip(A_coo.data, A_coo.row, A_coo.col)):
            f.write(f"{k}\t{val:.1f}\n")


def main():
    cases = [
        {"nx_ny": 50, "epsilon": (1.0, 1.0), "use_rho": False, "v1": V1_DEFAULT, "v2": V2_DEFAULT, "v3": V3_DEFAULT,
         "v4": V4_DEFAULT, "title": r"$nx=ny=50, \epsilon_1=1, \epsilon_2=1$", "vmin": VMIN_DEFAULT, "vmax": VMAX_DEFAULT},
        {"nx_ny": 100, "epsilon": (1.0, 1.0), "use_rho": False, "v1": V1_DEFAULT, "v2": V2_DEFAULT, "v3": V3_DEFAULT,
         "v4": V4_DEFAULT, "title": r"$nx=ny=100$", "vmin": VMIN_DEFAULT, "vmax": VMAX_DEFAULT},
        {"nx_ny": 200, "epsilon": (1.0, 1.0), "use_rho": False, "v1": V1_DEFAULT, "v2": V2_DEFAULT, "v3": V3_DEFAULT,
         "v4": V4_DEFAULT, "title": r"$nx=ny=200$", "vmin": VMIN_DEFAULT, "vmax": VMAX_DEFAULT},
        {"nx_ny": 100, "epsilon": (1.0, 1.0), "use_rho": True, "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 0.0,
         "title": r"$nx=ny=100, \epsilon_1=1, \epsilon_2=1$", "vmin": -0.8, "vmax": 0.8},
        {"nx_ny": 100, "epsilon": (1.0, 2.0), "use_rho": True, "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 0.0,
         "title": r"$\epsilon_1=1, \epsilon_2=2$", "vmin": -0.8, "vmax": 0.8},
        {"nx_ny": 100, "epsilon": (1.0, 10.0), "use_rho": True, "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 0.0,
         "title": r"$\epsilon_1=1, \epsilon_2=10$", "vmin": -0.8, "vmax": 0.8}
    ]

    for case in cases:
        poisson(case["nx_ny"], case["nx_ny"], case["epsilon"][0], case["epsilon"][1],
                case["v1"], case["v2"], case["v3"], case["v4"],
                use_function_rho=case["use_rho"], title=case["title"], vmin=case["vmin"], vmax=case["vmax"])


if __name__ == "__main__":
    main()

    poisson(4,4,1,1,10,-10,10,-10,False, "sprawdzenie nx=ny=4", -10, 10)