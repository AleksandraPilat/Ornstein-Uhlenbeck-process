import numpy as np
import matplotlib.pyplot as plt


def simulate_ornstein_uhlenbeck(theta, mu, sigma, x0, dt, T):
    """
    Symulacja procesu Ornsteina-Uhlenbecka.

    :param theta: Współczynnik regulacji.
    :param mu: Średnia wartość procesu.
    :param sigma: Odchylenie standardowe szumu.
    :param x0: Początkowa wartość procesu.
    :param dt: Skoki.
    :param T: Całkowity czas trwania symulacji.
    :return: t: Tablica wartości czasu.
             x: Tablica wartości procesu.
    """
    num_steps = int(T / dt)
    t = np.linspace(0.0, T, num_steps)
    x = np.zeros(num_steps)
    x[0] = x0

    for i in range(1, num_steps):
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * dW

    return t, x


# Parametry
theta = 0.5
mu = 1.0
sigma = 0.1
x0 = 0.0
T = 10.0

# Wartości skoków
dt_values = [0.1, 0.05, 0.01, 0.005]

# Prawdziwe rozwiązanie procesu Ornsteina-Uhlenbecka
t_true, x_true = simulate_ornstein_uhlenbeck(theta, mu, sigma, x0, 0.001, T)

# Gęsta siatka czasowa
t_dense = np.linspace(0.0, T, 10000)
x_dense = np.interp(t_dense, t_true, x_true)

# Rzadka siatka czasowa
t_sparse = np.arange(0.0, T, 0.5)
x_sparse = np.interp(t_sparse, t_true, x_true)

# Zestaw kolorów
colors = ['blue', 'orange', 'green', 'red']

# Wykres
plt.figure(figsize=(10, 12))

# Główny wykres
plt.subplot(3, 1, 1)
plt.plot(t_true, x_true, label='Prawdziwe rozwiązanie', color='black')

# Podwykres z błędem empirycznym i analizą błędu empirycznego
plt.subplot(3, 1, 2)
plt.xlabel('Czas')
plt.ylabel('Błąd empiryczny')
plt.title('Błąd empiryczny przybliżeń')
plt.grid(True)

# Podwykres z gęstą i rzadką siatką czasową
plt.subplot(3, 1, 3)
plt.plot(t_dense, x_dense, label='Gęsta siatka czasowa', color='blue')
plt.plot(t_sparse, x_sparse, label='Rzadka siatka czasowa', color='red')

plt.xlabel('Czas')
plt.ylabel('Wartość x')
plt.title('Gęsta i rzadka siatka czasowa')
plt.legend()
plt.grid(True)

# Analiza błędu empirycznego i rysowanie wykresu błędu empirycznego
for i, dt in enumerate(dt_values):
    t, x = simulate_ornstein_uhlenbeck(theta, mu, sigma, x0, dt, T)

    if len(x) != len(x_true):
        x_true_interp = np.interp(t, t_true, x_true)
        error = np.abs(x_true_interp - x)
    else:
        error = np.abs(x_true - x)

    mean_error = np.mean(error)  # Błąd empiryczny jako średnia wartość bezwzględna różnic

    # Podwykres z błędem empirycznym i analizą błędu empirycznego
    plt.subplot(3, 1, 2)
    plt.plot(t, error, label=f'Przybliżenie (dt = {dt}), błąd = {mean_error:.4f}', color=colors[i])

    # Główny wykres
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label=f'Przybliżenie (dt = {dt}), błąd = {mean_error:.4f}', color=colors[i])

plt.subplot(3, 1, 1)
plt.xlabel('Czas')
plt.ylabel('Wartość x')
plt.title('Proces Ornsteina-Uhlenbecka')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
