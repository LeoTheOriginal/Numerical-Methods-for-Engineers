import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def oblicz_qwy(przeplyw, calkowita_wysokosc, granica_y):
    licznik = (calkowita_wysokosc ** 3 - granica_y ** 3 - 3 * granica_y * calkowita_wysokosc ** 2 +
               3 * (granica_y ** 2) * calkowita_wysokosc)
    return przeplyw * licznik / (calkowita_wysokosc ** 3)


@njit
def zastosuj_granice_psi(psi_pole, przeplyw, qwy, krok, lepkosc,
                         granica_x_start, granica_y_start, siatka_x, siatka_y,
                         calkowita_wysokosc, granica_y):
    # Granica A
    for j in range(granica_y_start, siatka_y + 1):
        y = j * krok
        psi_pole[0, j] = (przeplyw / (2 * lepkosc)) * (
                y ** 3 / 3 - (y ** 2 / 2) * (granica_y + calkowita_wysokosc) + y * granica_y * calkowita_wysokosc
        )

    # Granica C
    for j in range(0, siatka_y + 1):
        y = j * krok
        psi_pole[siatka_x, j] = (qwy / (2 * lepkosc)) * (
                y ** 3 / 3 - (y ** 2 * (calkowita_wysokosc / 2))
        ) + ((przeplyw * granica_y ** 2 * (-granica_y + 3 * calkowita_wysokosc)) / (12 * lepkosc))

    # Granica B
    for i in range(1, siatka_x):
        psi_pole[i, siatka_y] = psi_pole[0, siatka_y]

    # Granica D
    for i in range(granica_x_start, siatka_x):
        psi_pole[i, 0] = psi_pole[0, granica_y_start]

    # Granica E
    for j in range(1, granica_y_start + 1):
        psi_pole[granica_x_start, j] = psi_pole[0, granica_y_start]

    # Granica F
    for i in range(1, granica_x_start + 1):
        psi_pole[i, granica_y_start] = psi_pole[0, granica_y_start]


@njit
def zastosuj_granice_zeta(zeta_pole, psi_pole, przeplyw, qwy, krok, lepkosc,
                          granica_x_start, granica_y_start, siatka_x, siatka_y,
                          calkowita_wysokosc, granica_y):
    # Granica A
    for j in range(granica_y_start, siatka_y + 1):
        y = j * krok
        zeta_pole[0, j] = (przeplyw / (2 * lepkosc)) * (2 * y - granica_y - calkowita_wysokosc)

    # Granica C
    for j in range(0, siatka_y + 1):
        y = j * krok
        zeta_pole[siatka_x, j] = (qwy / (2 * lepkosc)) * (2 * y - calkowita_wysokosc)

    # Granica B
    for i in range(1, siatka_x):
        zeta_pole[i, siatka_y] = 2 / (krok ** 2) * (psi_pole[i, siatka_y - 1] - psi_pole[i, siatka_y])

    # Granica D
    for i in range(granica_x_start + 1, siatka_x):
        zeta_pole[i, 0] = 2 / (krok ** 2) * (psi_pole[i, 1] - psi_pole[i, 0])

    # Granica E
    for j in range(1, granica_y_start):
        zeta_pole[granica_x_start, j] = 2 / (krok ** 2) * (
                psi_pole[granica_x_start + 1, j] - psi_pole[granica_x_start, j])

    # Granica F
    for i in range(1, granica_x_start + 1):
        zeta_pole[i, granica_y_start] = 2 / (krok ** 2) * (
                psi_pole[i, granica_y_start + 1] - psi_pole[i, granica_y_start])

    # Rog E/F
    zeta_pole[granica_x_start, granica_y_start] = 0.5 * (zeta_pole[granica_x_start - 1, granica_y_start] +
                                                         zeta_pole[granica_x_start, granica_y_start - 1])


@njit
def czy_granica_punkt(i, j, granica_x_start, granica_y_start, siatka_x, siatka_y):
    return (
            (i == 0 and granica_y_start <= j <= siatka_y) or            # A
            (j == siatka_y) or                                          # B
            (i == siatka_x) or                                          # C
            (granica_x_start <= i <= siatka_x and j == 0) or            # D
            (0 <= i <= granica_x_start and j == granica_y_start) or     # F
            (i == granica_x_start and 0 <= j <= granica_y_start)        # E
    )


@njit
def wykonaj_relaksacje(przeplyw, krok, gestosc, lepkosc,
                       granica_x_start, granica_y_start, granica_y_koniec,
                       siatka_x, siatka_y, maks_iteracje, calkowita_wysokosc, granica_y):
    predkosc_u = np.zeros((siatka_x + 1, siatka_y + 1))
    predkosc_v = np.zeros((siatka_x + 1, siatka_y + 1))
    psi_pole = np.zeros((siatka_x + 1, siatka_y + 1))
    zeta_pole = np.zeros((siatka_x + 1, siatka_y + 1))

    for i in range(granica_x_start):
        for j in range(granica_y_start):
            psi_pole[i, j] = np.nan
            zeta_pole[i, j] = np.nan

    Qwy = oblicz_qwy(przeplyw, calkowita_wysokosc, granica_y)

    zastosuj_granice_psi(psi_pole, przeplyw, Qwy, krok, lepkosc,
                         granica_x_start, granica_y_start, siatka_x, siatka_y,
                         calkowita_wysokosc, granica_y)

    historia_gamma = []
    for iteracja in range(1, maks_iteracje + 1):
        for i in range(1, siatka_x):
            for j in range(1, siatka_y):
                if not czy_granica_punkt(i, j, granica_x_start, granica_y_start, siatka_x, siatka_y):
                    psi_pole[i, j] = (
                                             psi_pole[i + 1, j] + psi_pole[i - 1, j] +
                                             psi_pole[i, j + 1] + psi_pole[i, j - 1] -
                                             (krok ** 2) * zeta_pole[i, j]
                                     ) / 4.0

                    zeta_pole[i, j] = 0.25 * (
                            zeta_pole[i + 1, j] + zeta_pole[i - 1, j] +
                            zeta_pole[i, j + 1] + zeta_pole[i, j - 1]
                    )

                    if iteracja > 2000:
                        zeta_pole[i, j] -= (gestosc / (16 * lepkosc)) * (
                                (psi_pole[i, j + 1] - psi_pole[i, j - 1]) *
                                (zeta_pole[i + 1, j] - zeta_pole[i - 1, j]) -
                                (psi_pole[i + 1, j] - psi_pole[i - 1, j]) *
                                (zeta_pole[i, j + 1] - zeta_pole[i, j - 1])
                        )

        zastosuj_granice_zeta(zeta_pole, psi_pole, przeplyw, Qwy, krok,
                              lepkosc, granica_x_start, granica_y_start, siatka_x, siatka_y,
                              calkowita_wysokosc, granica_y)

        gamma = 0.0
        for i in range(1, siatka_x):
            gamma += (
                    psi_pole[i + 1, granica_y_koniec] + psi_pole[i - 1, granica_y_koniec] +
                    psi_pole[i, granica_y_koniec + 1] + psi_pole[i, granica_y_koniec - 1] -
                    4 * psi_pole[i, granica_y_koniec] - (krok ** 2) * zeta_pole[i, granica_y_koniec]
            )
        historia_gamma.append(gamma)

    for i in range(1, siatka_x):
        for j in range(1, siatka_y):
            if i > granica_x_start or j > granica_y_start:
                predkosc_u[i, j] = (psi_pole[i, j + 1] - psi_pole[i, j - 1]) / (2.0 * krok)
                predkosc_v[i, j] = -(psi_pole[i + 1, j] - psi_pole[i - 1, j]) / (2.0 * krok)

    return [predkosc_u, predkosc_v, psi_pole, zeta_pole], historia_gamma


def rysuj_izolinie(wsp_x, wsp_y, dane, tytul, kolor_mapy, liczba_poziomow):
    plt.figure(figsize=(16, 6))
    izolinie = plt.contour(
        wsp_x, wsp_y, dane.T,
        levels=liczba_poziomow,
        cmap=kolor_mapy,
        vmin=np.nanmin(dane),
        vmax=np.nanmax(dane)
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(izolinie)
    plt.title(tytul)
    plt.show()


def rysuj_pseudokolor(wsp_x, wsp_y, dane, tytul, kolor_mapy):
    plt.figure(figsize=(16, 6))
    pseudokolor = plt.pcolor(
        wsp_x, wsp_y, dane.T,
        cmap=kolor_mapy,
        shading="auto",
        vmin=np.nanmin(dane),
        vmax=np.nanmax(dane)
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(pseudokolor)
    plt.title(tytul)
    plt.show()


class Symulacja:
    def __init__(self):
        # Parametry Symulacji
        self.ROZMIAR_KROKU = 0.01
        self.GESTOSC = 1.0
        self.LEPKOSC = 1.0
        self.SIATKA_X = 200
        self.SIATKA_Y = 90
        self.GRANICA_X_START = 50
        self.GRANICA_Y_START = 55
        self.GRANICA_Y_KONIEC = self.GRANICA_Y_START + 2
        self.MAKS_ITERACJE = 20000

        self.calkowita_wysokosc = self.ROZMIAR_KROKU * self.SIATKA_Y
        self.granica_y = self.ROZMIAR_KROKU * self.GRANICA_Y_START

        self.wsp_x = np.linspace(0, self.SIATKA_X * self.ROZMIAR_KROKU, self.SIATKA_X + 1)
        self.wsp_y = np.linspace(0, self.SIATKA_Y * self.ROZMIAR_KROKU, self.SIATKA_Y + 1)

    def rysuj_izolinie_metoda(self, dane, tytul, kolor_mapy, liczba_poziomow):
        rysuj_izolinie(self.wsp_x, self.wsp_y, dane, tytul, kolor_mapy, liczba_poziomow)

    def rysuj_pseudokolor_metoda(self, dane, tytul, kolor_mapy):
        rysuj_pseudokolor(self.wsp_x, self.wsp_y, dane, tytul, kolor_mapy)

    def wizualizuj_wyniki(self, dane_symulacji, przeplyw):
        predkosc_u, predkosc_v, psi_pole, zeta_pole = dane_symulacji

        tytuly = {
            'stream': f"Funkcja Strumienia Psi(x,y) dla Q = {przeplyw}",
            'vorticity': f"Funkcja wirowości Ksi(x,y) dla Q = {przeplyw}",
            'velocity_u': f"Pozioma składowa Prędkości u(x,y) dla Q = {przeplyw}",
            'velocity_v': f"Pionowa składowa Prędkości v(x,y) dla Q = {przeplyw}"
        }

        self.rysuj_izolinie_metoda(psi_pole, tytuly['stream'], "gnuplot", 50)
        self.rysuj_izolinie_metoda(zeta_pole, tytuly['vorticity'], "gnuplot", 50)
        self.rysuj_pseudokolor_metoda(predkosc_u, tytuly['velocity_u'], "jet")
        self.rysuj_pseudokolor_metoda(predkosc_v, tytuly['velocity_v'], "jet")

    def uruchom_relaksacje(self, q):
        print(f"Rozpoczynanie procesu relaksacji dla Q = {q}")
        dane_symulacji, _ = wykonaj_relaksacje(
            q, self.ROZMIAR_KROKU, self.GESTOSC, self.LEPKOSC,
            self.GRANICA_X_START, self.GRANICA_Y_START, self.GRANICA_Y_KONIEC,
            self.SIATKA_X, self.SIATKA_Y, self.MAKS_ITERACJE,
            self.calkowita_wysokosc, self.granica_y
        )
        self.wizualizuj_wyniki(dane_symulacji, q)

    def uruchom_symulacje(self):
        przeplywy = [-1000, -4000, 4000]
        for Q in przeplywy:
            self.uruchom_relaksacje(Q)


def main():
    symulacja = Symulacja()
    symulacja.uruchom_symulacje()


if __name__ == "__main__":
    main()
