import numpy as np
import matplotlib.pyplot as plt
import os

# Zakładamy, że pliki są w katalogu "plots"
# Wczytujemy siatki
grid_x = np.loadtxt("plots/grid_x.txt")
grid_y = np.loadtxt("plots/grid_y.txt")
grid_t = np.loadtxt("plots/grid_t.txt")

# Wczytujemy dane c(t) i x_sr(t) dla D=0 i D=0.1
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

# Wczytujemy vx, vy
vx = np.loadtxt("plots/vx.txt")
vy = np.loadtxt("plots/vy.txt")

# 3. Mapa vx(x,y)
plt.figure(figsize=(8,3))
plt.imshow(vx.T, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]), aspect='auto', cmap='jet')
plt.colorbar(label='vx(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Składowa prędkości vx(x,y)')
plt.tight_layout()
plt.show()

# 4. Mapa vy(x,y)
plt.figure(figsize=(8,3))
plt.imshow(vy.T, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]), aspect='auto', cmap='jet')
plt.colorbar(label='vy(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Składowa prędkości vy(x,y)')
plt.tight_layout()
plt.show()

# Wyświetlanie map u(x,y) dla D=0.0 i D=0.1 w 5 chwilach czasowych
# Zakładamy, że nazwy plików to u_{D}_{k}.txt, gdzie D=0.0 lub 0.1 i k=1..5
# oraz że zostały wygenerowane w poprzednim etapie.
D_values = [0.0, 0.1]
# 5 czasów: T_k = k * t_max/5 (k=1,...,5)
# pliki np. "u_0.0_1.txt", "u_0.0_2.txt" ... "u_0.1_5.txt"
# Jeśli nazwy plików są inne, należy je dostosować.

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
