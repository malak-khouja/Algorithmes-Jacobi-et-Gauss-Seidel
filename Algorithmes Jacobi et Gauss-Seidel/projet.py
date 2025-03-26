import numpy as np
import time
import matplotlib.pyplot as plt
import platform
import psutil
   
#creer matrice A inversible et convergente:
def creer_matrice_A(x,y):
    while True:
        A = np.random.randint(-10, 10, size=(x, y))
        np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1)
        if np.linalg.det(A) != 0:
            return A
        
#creer matrice diagonale D:
def creer_matrice_D(A,x,y):
    D=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if i==j:
                D[i][j]=A[i][j]
            else:
                D[i][j]=0
    return D
    
#fonctions de calcul:
def somme(M,N,x,y):
    S=np.zeros((x,y))    
    for i in range(x):
        for j in range(y):
            S[i][j]=M[i][j]+N[i][j]
    return S


def calc_case(i,j,M,N,y):
        s=0
        for p in range(y):
            s += M[i][p] * N[p][j]
        return s
       
def produit(M,N,x,y):
    P=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            P[i][j]=calc_case(i,j,M,N,y)
    return P

def inverse(M,x,y):
    I=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if i==j and M[i][j]!=0 :
                I[i][j]=1/M[i][j]
            else:
                I[i][j]=0
    return I


#creer matrice b:
def creer_matrice_b(x):
    b=np.random.randint(-10,10,size=(x,1))
    return b

# Implémentation de la méthode de Jacobi:
def jacobi(A,D,x,y):
    #creer matrice trig inf E
    E=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if i>=j:
                E[i][j]=-A[i][j]
            else:
                E[i][j]=0
    #creer matrice trig sup F
    F=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if i<=j:
                F[i][j]=-A[i][j]
            else:
                F[i][j]=0    
    J=produit(inverse(D,x,y),somme(E,F,x,y),x,y)
    cj=np.dot(inverse(D,x,y),b)
    return J,cj


def calcul_Xk(A,D,x0,cj,b,tol=1e-6,n=100):
    x=np.zeros(n)
    J=jacobi(A,D,x,y)
    k=0
    x[k]=x0
    while not np.linalg.norm(np.dot(A,x[k])-b)<tol :
        x[k]=somme(produit(J,x[k-1]),cj,y)
        k=k+1
    return x,k

# Mesure du temps d'exécution pour Jacobi:
def mesure_temps_jacobi(A,D,x,y):
    t=np.zeros(5)
    for i in range(5):
        start_time=time.perf_counter()
        J=jacobi(A,D,x,y)
        #print("J =",J)
        end_time=time.perf_counter()
        temps_execution = end_time - start_time
        #print(f"Temps d'exécution (Essai {i+1}) = {temps_execution:.4f} secondes")
        t[i]=temps_execution
    return np.mean(t)

# Implémentation de la méthode de Gauss-Seidel:
def gauss_seidel(A, b, x0, tol=1e-6, max_iterations=500):
    n = len(A)
    x = x0.copy()
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Mesure du temps d'exécution pour Gauss-Seidel:
def measure_time(A, b, x0):
    start_time = time.perf_counter()
    gauss_seidel(A, b, x0)
    end_time = time.perf_counter()
    return end_time - start_time

#valeurs propres
def val_prop(A):
    valeurs_propres = np.linalg.eigvals(A)
    return np.max(np.abs(valeurs_propres))

# Graphe:
def plot_results(sizes, jacobi_times, gs_times, message):
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, jacobi_times, label='Jacobi', marker='o')
    plt.plot(sizes, gs_times, label='Gauss-Seidel', marker='s')
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Comparaison des performances : Jacobi vs Gauss-Seidel')
    plt.legend()
    plt.grid()
    plt.text(sizes[-1] * 0.5, max(jacobi_times + gs_times) * 0.9, message, fontsize=12, color="red", fontweight="bold", ha='center')
    plt.figtext(0.5, -0.1, message, ha='center', fontsize=12, color='blue', fontweight='bold')
    plt.show()

#analyse resultat
sizes = [100, 400, 500, 700, 1000, 1500, 2000]
jacobi_times = []
gs_times = []
for size in sizes:
    x = y = size 
    A = creer_matrice_A(x,y)
    essai=0
    b= creer_matrice_b(x)
    x0 = np.zeros_like(b)  
    #print("A =",A)
    D = np.zeros((x, y))
    if A is not None: 
        D = creer_matrice_D(A,x,y) 
        #print("D =",D) 
        jacobi_times.append(mesure_temps_jacobi(A,D,x,y))
        gs_times.append(measure_time(A, b, x0))
J, cj = jacobi(A,D,x,y)
gs_result = gauss_seidel(A, b, x0, tol=1e-6, max_iterations=500)
if np.linalg.norm(np.dot(A, gs_result) - b.flatten()) < np.linalg.norm(np.dot(A, J) - b.flatten()):
    message = "Gauss-Seidel converge plus vite que Jacobi"
else:
    message = "Jacobi converge plus vite que Gauss-Seidel"
plot_results(sizes, jacobi_times, gs_times, message)


#pour avoir les informations de la machine
def get_system_info():
    info = {
        "OS": platform.system(),
        "Version OS": platform.version(),
        "Nom de la machine": platform.node(),
        "Processeur": platform.processor(),
        "Architecture": platform.architecture()[0],
        "Cœurs physiques": psutil.cpu_count(logical=False),
        "Cœurs logiques": psutil.cpu_count(logical=True),
        "Fréquence CPU (MHz)": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
        "Mémoire RAM totale (Go)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Langage": platform.python_implementation(),
        "Version Python": platform.python_version(),
        "Compilateur": platform.python_compiler()
    }
    return info