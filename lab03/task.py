import matplotlib.pyplot as plt


x0 = 0.01
v0 = 0
dt0 = 1
S = 0.75
p = 2
t_max = 40
alpha = 5
delta = 1e-10
TOL = [1e-2, 1e-5]


def f(v):
    return v


def g(x, v):
    return alpha * (1 - x ** 2) * v - x


def draw_single_plot(x1, y1, x2, y2, x_name, y_name, method, filename):
    plt.figure()
    plt.plot(x1, y1, label='$TOL = 10^{-2}$')
    plt.plot(x2, y2, label='$TOL = 10^{-5}$')
    plt.legend()
    plt.grid()
    plt.xlabel('$' + x_name + '$')
    plt.ylabel('$' + y_name + '(' + x_name + ')$')
    plt.title(method + ' $' + y_name + '(' + x_name + ')$')
    plt.savefig(f"{filename}_{y_name}({x_name}).png")


def draw_plots(method, filename, t1, t2, x1, x2, v1, v2, dt1, dt2):
    draw_single_plot(t1, v1, t2, v2, 't', 'v', method, filename)
    draw_single_plot(t1, x1, t2, x2, 't', 'x', method, filename)
    draw_single_plot(t1, dt1, t2, dt2, 't', 'dt', method, filename)
    draw_single_plot(x1, v1, x2, v2, 'x', 'v', method, filename)


def solve(scheme, tol):
    t = [0]
    dt = [dt0, dt0]
    x = [x0]
    v = [v0]
    while True:
        x2_n1, v2_n1 = scheme(x[-1], v[-1], dt[-1])
        x2_n2, v2_n2 = scheme(x2_n1, v2_n1, dt[-1])

        x1_n2, v1_n2 = scheme(x[-1], v[-1], 2 * dt[-1])

        Ex = (x2_n2 - x1_n2) / (2 ** p - 1)
        Ev = (v2_n2 - v1_n2) / (2 ** p - 1)

        if max(abs(Ex), abs(Ev)) < tol:
            t.append(t[-1] + 2 * dt[-1])
            x.append(x2_n2)
            v.append(v2_n2)
            dt.append(dt[-1])

        dt[-1] *= ((S * tol) / (max(abs(Ex), abs(Ev)))) ** (1 / (p + 1))

        if t[-1] >= t_max:
            dt.pop()
            break
    return t, x, v, dt


def trapezoids_scheme(x, v, dt):
    x1 = x
    v1 = v
    while True:
        F = x1 - x - dt / 2 * (f(v) + f(v1))
        G = v1 - v - dt / 2 * (g(x, v) + g(x1, v1))

        a11 = 1
        a12 = -dt / 2
        a21 = -dt / 2 * (-2 * alpha * x1 * v1 - 1)
        a22 = 1 - dt / 2 * alpha * (1 - x1 ** 2)

        dx = (-F * a22 + G * a12) / (a11 * a22 - a12 * a21)
        dv = (-G * a11 + F * a21) / (a11 * a22 - a12 * a21)

        x1 += dx
        v1 += dv

        if abs(dx) < delta and abs(dv) < delta:
            break
    return x1, v1


t1, x1, v1, dt1 = solve(trapezoids_scheme, TOL[0])
t2, x2, v2, dt2 = solve(trapezoids_scheme, TOL[1])

draw_plots('Metoda Trapezów', 'Trapezy', t1, t2, x1, x2, v1, v2, dt1, dt2)


def rk2_scheme(x, v, dt):
    k1x = f(v)
    k1v = g(x, v)

    k2x = f(v + dt * k1v)
    k2v = g(x + dt * k1x, v + dt * k1v)

    x1 = x + dt / 2 * (k1x + k2x)
    v1 = v + dt / 2 * (k1v + k2v)

    return x1, v1


t1, x1, v1, dt1 = solve(rk2_scheme, TOL[0])
t2, x2, v2, dt2 = solve(rk2_scheme, TOL[1])

draw_plots('Metoda RK2', 'RK2', t1, t2, x1, x2, v1, v2, dt1, dt2)
