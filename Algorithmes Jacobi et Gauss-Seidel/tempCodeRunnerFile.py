import numpy as np
import time
import matplotlib.pyplot as plt

# Fonction pour générer une matrice aléatoire bien conditionnée
def generate_matrix(n):
    A = np.random.rand(n, n) * 10
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1)  # Rendre la matrice diagonale dominante
    b = np.random.rand(n) * 10
    return A, b

# Implémentation de la méthode de Jacobi
def jacobi(A, b, tol=1e-10, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    for _ in range(max_iterations):
        for i in range(n):
            x_new[i] = (b[i] - np.sum(A[i, :] * x) + A[i, i] * x[i]) / A[i, i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new.copy()
    
    return x

# Implémentation de la méthode de Gauss-Seidel
def gauss_seidel(A, b, tol=1e-10, max_iterations=1000):
    n = len(A)
    x = np.zeros(n)
    
    for _ in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.sum(A[i, :i] * x[:i]) - np.sum(A[i, i+1:] * x_old[i+1:])) / A[i, i]
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            break
    
    return x

# Mesure des performances
def measure_time(method, A, b):
    start = time.perf_counter()
    method(A, b)
    end = time.perf_counter()
    return end - start

# Expérimentation
sizes = [10, 40, 50, 70, 100, 150, 200]
jacobi_times = []
gauss_seidel_times = []
num_trials = 5

for n in sizes:
    jacobi_avg_time = 0
    gauss_seidel_avg_time = 0
    for _ in range(num_trials):
        A, b = generate_matrix(n)
        jacobi_avg_time += measure_time(jacobi, A, b)
        gauss_seidel_avg_time += measure_time(gauss_seidel, A, b)
    
    jacobi_times.append(jacobi_avg_time / num_trials)
    gauss_seidel_times.append(gauss_seidel_avg_time / num_trials)

# Affichage des résultats
def plot_results(sizes, jacobi_times, gauss_seidel_times):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, jacobi_times, label='Jacobi', marker='o')
    plt.plot(sizes, gauss_seidel_times, label='Gauss-Seidel', marker='s')
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Comparaison des performances : Jacobi vs Gauss-Seidel')
    plt.legend()
    plt.grid()
    plt.show()

plot_results(sizes, jacobi_times, gauss_seidel_times)

# Affichage des résultats sous forme de tableau
import pandas as pd

df = pd.DataFrame({
    'Taille de la matrice': sizes,
    'Temps Jacobi (s)': jacobi_times,
    'Temps Gauss-Seidel (s)': gauss_seidel_times
})
print(df)

