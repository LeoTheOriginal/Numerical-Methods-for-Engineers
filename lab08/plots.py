import numpy as np
import matplotlib.pyplot as plt
import os


grid_x = np.loadtxt("plots/grid_x.txt")
grid_y = np.loadtxt("plots/grid_y.txt")
grid_t = np.loadtxt("plots/grid_t.txt")

c_0 = np.loadtxt("plots/c_0.0.txt")
c_01 = np.loadtxt("plots/c_0.1.txt")
x_sr_0 = np.loadtxt("plots/x_sr_0.0.txt")
x_sr_01 = np.loadtxt("plots/x_sr_0.1.txt")

fig, ax = plt.subplots(2,1,figsize=(8,6))
ax[0].plot(grid_t, c_0, label="D=0.0")
ax[0].plot(grid_t, c_01, label="D=0.1")
ax[0].set_xlabel("t")
ax[0].set_ylabel("c(t)")
ax[0].legend()
ax[0].set_title("Całka gęstości c(t)")

ax[1].plot(grid_t, x_sr_0, label="D=0.0")
ax[1].plot(grid_t, x_sr_01, label="D=0.1")
ax[1].set_xlabel("t")
ax[1].set_ylabel("x_sr(t)")
ax[1].legend()
ax[1].set_title("Średnie położenie pakietu x_sr(t)")

plt.tight_layout()
plt.show()

vx = np.loadtxt("plots/vx.txt")
vy = np.loadtxt("plots/vy.txt")

plt.figure(figsize=(8,3))
plt.imshow(vx.T, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]), aspect='auto', cmap='jet')
plt.colorbar(label='vx(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Składowa prędkości vx(x,y)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,3))
plt.imshow(vy.T, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]), aspect='auto', cmap='jet')
plt.colorbar(label='vy(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Składowa prędkości vy(x,y)')
plt.tight_layout()
plt.show()

D_values = [0.0, 0.1]

for D in D_values:
    for k in range(1,6):
        filename = f"plots/u_{D}_{k}.txt"
        if os.path.exists(filename):
            u_map = np.loadtxt(filename)
            plt.figure(figsize=(8,3))
            plt.imshow(u_map.T, origin='lower',
                       extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
                       aspect='auto', cmap='jet')
            plt.colorbar(label='u(x,y)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"Rozkład gęstości u(x,y) dla D={D} w kroku {k}")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Brak pliku: {filename}, pomijam ten wykres.")
