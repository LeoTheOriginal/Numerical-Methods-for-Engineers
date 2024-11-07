import math
import numpy as np
import matplotlib.pyplot as plt

########## PROBLEM AUTONOMICZNY	##########


l = -1
t_min = 0
t_max = 5
delta_t = (0.01, 0.1, 1.0)

def function(t):
    return np.exp(l * t)

############# Metody jawne #############
def euler(y, delta, n):
    y.append(y[n] + delta * l * y[n])

def RK2(y, delta, n):
    k1= l * y[n]
    k2 = l * (y[n] + delta * k1)
    y.append(y[n] + delta * (k1 + k2) / 2)

def RK4(y, delta, n):
    k1 = l * y[n]
    k2 = l * (y[n] + delta / 2 * k1)
    k3 = l * (y[n] + delta / 2 * k2)
    k4 = l * (y[n] + delta * k3)
    y.append(y[n] + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


############# Funckja rozwiązująca #############
def solve(file_name, method_name, recipe):
    plt.figure(0)
    plt.figure(1)

    for delta in delta_t:
        x = [0]
        y = [1]
        y_function = [1]
        diff = [0]

        for n in range(int((t_max - t_min) / delta)):
            x.append(round(x[n] + delta, 2))
            y_function.append(function(x[n + 1]))
            recipe(y, delta, n)
            diff.append(y[n + 1] - y_function[n + 1])

        plt.figure(0)
        plt.plot(x, y, label=r'$\Delta t = ' + str(delta) + '$')
        plt.figure(1)
        plt.plot(x, diff, label=r'$\Delta t = ' + str(delta) + '$')

    min_delta = min(delta_t)
    x = [0]
    y = [1]
    for n in range(int((t_max - t_min) / min_delta)):
        x.append(x[n] + min_delta)
        y.append(function(x[n + 1]))
    plt.figure(0)
    plt.plot(x, y, label='Analityczne')

    plt.legend()
    plt.grid()
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    plt.title('Metoda jawna ' + method_name)
    plt.savefig(file_name + '.png')

    plt.figure(1)
    plt.legend()
    plt.grid()
    plt.xlabel('$t$')
    plt.ylabel(r'$y_{numeryczne}(t) - y_{analityczne}(t)$')
    plt.title('Różnica metody jawnej ' + method_name + ' z rozwiązaniem analitycznym')
    plt.savefig(file_name + ' Diff.png')

    plt.close('all')


############## Wywołanie funkcji ##############
solve('Euler', 'Eulera', euler)
solve('RK2', 'RK2 (trapezów)', RK2)
solve('RK4', 'RK4', RK4)


################# RRZ 2. RZĘDU	###################

dt = 10e-4
R = 100
L = 0.1
C = 0.001
om0 = 1 / math.sqrt(L * C)
T0 = 2 * math.pi / om0
tmin = 0
tmax = 4 * T0
omv_mods = (0.5, 0.8, 1.0, 1.2)

plt.figure(0)
plt.figure(1)

for omv_mod in omv_mods:
    x = [0]
    Q = [0]
    I = [0]
    V = lambda n: 10 * math.sin(omv_mod * om0 * (x[0] + n * dt))
    for n in range(int((tmax - tmin) / dt)):
        x.append(x[n] + dt)
        kQ1 = I[n]
        kI1 = V(n) / L - Q[n] / (L * C) - R * I[n] / L
        kQ2 = I[n] + dt / 2 * kI1
        kI2 = V(n + 0.5) / L - (Q[n] + dt / 2 * kQ1) / (L * C) - R * (I[n] + dt / 2 * kI1) / L
        kQ3 = I[n] + dt / 2 * kI2
        kI3 = V(n + 0.5) / L - (Q[n] + dt / 2 * kQ2) / (L * C) - R * (I[n] + dt / 2 * kI2) / L
        kQ4 = I[n] + dt * kI3
        kI4 = V(n + 1) / L - (Q[n] + dt * kQ3) / (L * C) - R * (I[n] + dt * kI3) / L
        Q.append(Q[n] + dt / 6 * (kQ1 + 2 * kQ2 + 2 * kQ3 + kQ4))
        I.append(I[n] + dt / 6 * (kI1 + 2 * kI2 + 2 * kI3 + kI4))
    plt.figure(0)
    plt.plot(x, Q, label=r'$\omega_V = ' + str(omv_mod) + r'\omega_0$')
    plt.figure(1)
    plt.plot(x, I, label=r'$\omega_V = ' + str(omv_mod) + r'\omega_0$')

plt.figure(0)
plt.legend()
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$Q(t)$')
plt.title('RRZ 2. rzędu (metoda jawna RK4) – ładunek')
plt.savefig('RRZ2 Q.png')

plt.figure(1)
plt.legend()
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$I(t)$')
plt.title('RRZ 2. rzędu (metoda jawna RK4) – natężenie')
plt.savefig('RRZ2 I.png')

plt.close('all')