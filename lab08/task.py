import os
import numpy as np
from numba import njit

# Parametry zgodne z zadaniem
NX = 400
NY = 90
I1 = 200
I2 = 210
J1 = 50
DELTA = 0.01
SIGMA = 10.0 * DELTA
XA = 0.45
YA = 0.45
IT_MAX = 10_000


def x(i):
    return DELTA * i


def y(j):
    return DELTA * j


def initial_u(xi, yj):
    return np.exp(-((xi - XA) ** 2 + (yj - YA) ** 2) / (2.0 * SIGMA ** 2)) / (2.0 * np.pi * SIGMA ** 2)


@njit
def compute_velocity_field(psi_v_f, vx_v_f, vy_v_f):
    for i in range(1, NX):
        for j in range(1, NY):
            vx_v_f[i, j] = (psi_v_f[i, j + 1] - psi_v_f[i, j - 1]) / (2.0 * DELTA)
            vy_v_f[i, j] = -(psi_v_f[i + 1, j] - psi_v_f[i - 1, j]) / (2.0 * DELTA)

    for i in range(I1, I2 + 1):
        for j in range(J1 + 1):
            vx_v_f[i, j] = 0.0
            vy_v_f[i, j] = 0.0

    for i in range(1, NX):
        vx_v_f[i, 0] = 0.0
        vy_v_f[i, NY] = 0.0

    for j in range(NY + 1):
        vx_v_f[0, j] = vx_v_f[1, j]
        vx_v_f[NX, j] = vx_v_f[NX - 1, j]


@njit
def find_vmax(vx_f_m, vy_f_m):
    v_f_max = 0.0
    for i in range(NX + 1):
        for j in range(NY + 1):
            v_mag = np.sqrt(vx_f_m[i, j] * vx_f_m[i, j] + vy_f_m[i, j] * vy_f_m[i, j])
            if v_mag > v_f_max:
                v_f_max = v_mag
    return v_f_max


@njit
def crank_nicolson_step(u0_n_s, u1_n_s, vx_n_s, vy_n_s, d, dt_n_s):
    for k_n_s in range(20):
        for i in range(NX + 1):
            for j in range(1, NY):
                # Zastawka
                if I1 <= i <= I2 and j <= J1:
                    continue

                i_left = i - 1
                i_right = i + 1
                if i == 0:
                    i_left = NX
                if i == NX:
                    i_right = 0

                u0_ij = u0_n_s[i, j]

                # Adwekcja w x
                dudx0 = (u0_n_s[i_right, j] - u0_n_s[i_left, j]) / (2.0 * DELTA)
                dudx1 = (u1_n_s[i_right, j] - u1_n_s[i_left, j]) / (2.0 * DELTA)
                # Adwekcja w y
                dudy0 = (u0_n_s[i, j + 1] - u0_n_s[i, j - 1]) / (2.0 * DELTA)
                dudy1 = (u1_n_s[i, j + 1] - u1_n_s[i, j - 1]) / (2.0 * DELTA)

                # Dyfuzja
                lapl0 = (u0_n_s[i_right, j] + u0_n_s[i_left, j] + u0_n_s[i, j + 1]
                         + u0_n_s[i, j - 1] - 4.0 * u0_ij) / (DELTA * DELTA)
                lapl1 = (u1_n_s[i_right, j] + u1_n_s[i_left, j] + u1_n_s[i, j + 1]
                         + u1_n_s[i, j - 1]) / (DELTA * DELTA)

                numer = (u0_ij - dt_n_s * (vx_n_s[i, j] / 2.0) * (dudx0 + dudx1) -
                         dt_n_s * (vy_n_s[i, j] / 2.0) * (dudy0 + dudy1) + (dt_n_s * d / 2.0) * (lapl0 + lapl1))

                denom = 1.0 + (2.0 * d * dt_n_s) / (DELTA * DELTA)

                u1_n_s[i, j] = numer / denom

    return u1_n_s


os.makedirs("plots", exist_ok=True)

grid_x = np.array([x(i) for i in range(NX + 1)])
grid_y = np.array([y(j) for j in range(NY + 1)])
np.savetxt("plots/grid_x.txt", grid_x)
np.savetxt("plots/grid_y.txt", grid_y)

psi = np.zeros((NX + 1, NY + 1), dtype=np.float64)
with open("psi.dat", "r") as f:
    for line in f:
        i_, j_, val = line.split()
        i_ = int(i_)
        j_ = int(j_)
        val = float(val)
        psi[i_, j_] = val

vx = np.zeros((NX + 1, NY + 1), dtype=np.float64)
vy = np.zeros((NX + 1, NY + 1), dtype=np.float64)

compute_velocity_field(psi, vx, vy)

np.savetxt("plots/vx.txt", vx)
np.savetxt("plots/vy.txt", vy)

v_max = find_vmax(vx, vy)
dt = DELTA / (4.0 * v_max)
print(f"Vmax = {v_max}, dt = {dt}")

grid_t = np.array([dt * it for it in range(IT_MAX)])
np.savetxt("plots/grid_t.txt", grid_t)

u0 = np.zeros((NX + 1, NY + 1), dtype=np.float64)
u1 = np.zeros((NX + 1, NY + 1), dtype=np.float64)

for D in [0.0, 0.1]:
    print(f"\nCalculating for D = {D}")
    c_file = open(f"plots/c_{D}.txt", "w")
    x_sr_file = open(f"plots/x_sr_{D}.txt", "w")

    for i in range(NX + 1):
        for j in range(NY + 1):
            u0[i, j] = initial_u(grid_x[i], grid_y[j])

    for it in range(1, IT_MAX + 1):
        u1[:] = u0[:]
        u1 = crank_nicolson_step(u0, u1, vx, vy, D, dt)
        u0[:] = u1[:]

        c_val = np.sum(u0) * DELTA * DELTA
        x_sr_val = 0.0
        for i in range(NX + 1):
            for j in range(NY + 1):
                x_sr_val += grid_x[i] * u0[i, j] * DELTA * DELTA

        c_file.write(f"{c_val}\n")
        x_sr_file.write(f"{x_sr_val}\n")

        t_curr = it * dt
        tmax = IT_MAX * dt
        for k in range(1, 6):
            if abs(t_curr - k * (tmax / 5.0)) < 1e-14:
                # zapisujemy do pliku
                np.savetxt(f"plots/u_{D}_{k}.txt", u0)
    c_file.close()
    x_sr_file.close()

print("Done.")
