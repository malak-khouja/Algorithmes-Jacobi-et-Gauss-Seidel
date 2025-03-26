# 📌 **Jacobi vs Gauss-Seidel Performance Analysis**

## 📝 **Description**  
This Python project compares the performance of two numerical methods—**Jacobi and Gauss-Seidel**—for solving linear systems. It generates **random matrices**, applies **iterative methods**, measures execution time, and visualizes the results using **Matplotlib**.

---

## 🚀 **Features**  
✔️ **Generate** invertible and convergent matrices.  
✔️ **Implement** Jacobi and Gauss-Seidel iterative methods.  
✔️ **Measure** execution time for each method.  
✔️ **Plot** performance comparisons.  
✔️ **Retrieve** system information (CPU, RAM, OS).  

---

## 🛠 **Technologies Used**  
- **Python**  
- **NumPy** (Matrix operations)  
- **Matplotlib** (Data visualization)  
- **psutil** (System information)  
- **platform** (OS details)  
- **time** (Performance measurement)  

---

## 📥 **Installation**  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/malakkhouja/jacobi-vs-gauss-seidel.git
```
2️⃣ **Navigate to the project folder:**  
```bash
cd jacobi-vs-gauss-seidel
```
3️⃣ **Install dependencies:**  
```bash
pip install numpy matplotlib psutil
```
4️⃣ **Run the script:**  
```bash
python main.py
```

---

## 🎯 **Usage**  

1️⃣ The script generates a **random invertible matrix** and a **solution vector**.  
2️⃣ It **applies** both **Jacobi and Gauss-Seidel** methods.  
3️⃣ It **compares convergence speed** and **execution time**.  
4️⃣ A **graph** is displayed showing **performance differences**.  
5️⃣ System information is **retrieved** and printed.  

---

## 📂 **Project Structure**  
```
jacobi-vs-gauss-seidel/
│── main.py           # Main script
│── README.md         # Documentation
│── requirements.txt  # Dependencies
```

---

## 📌 **Key Functions**  

### 🔹 **Matrix Creation**  
```python
def creer_matrice_A(x, y):
    while True:
        A = np.random.randint(-10, 10, size=(x, y))
        np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1)
        if np.linalg.det(A) != 0:
            return A
```

### 🔹 **Jacobi Method**  
```python
def jacobi(A, D, x, y):
    E = -np.tril(A, -1)
    F = -np.triu(A, 1)
    J = np.dot(np.linalg.inv(D), (E + F))
    cj = np.dot(np.linalg.inv(D), b)
    return J, cj
```

### 🔹 **Gauss-Seidel Method**  
```python
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
```

### 🔹 **Performance Measurement**  
```python
def mesure_temps_jacobi(A, D, x, y):
    start_time = time.perf_counter()
    jacobi(A, D, x, y)
    end_time = time.perf_counter()
    return end_time - start_time
```

### 🔹 **System Information Retrieval**  
```python
def get_system_info():
    info = {
        "OS": platform.system(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Python Version": platform.python_version(),
    }
    return info
```

---

## 📊 **Performance Graph**  
The script **plots execution times** for **different matrix sizes**.  
Example output:  

📌 **"Gauss-Seidel converges faster than Jacobi"**  
📌 **"Jacobi converges faster than Gauss-Seidel"**  

```python
plt.plot(sizes, jacobi_times, label='Jacobi', marker='o')
plt.plot(sizes, gs_times, label='Gauss-Seidel', marker='s')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (s)')
plt.title('Jacobi vs Gauss-Seidel Performance')
plt.legend()
plt.show()
```

---

## 🤝 **Contributing**  
Contributions are welcome! To contribute:  

🍴 **Fork the repository**  
🔀 **Create a new branch**  
📝 **Commit your changes**  
🚀 **Submit a pull request**  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  

---

## 👤 **Author**  
🔗 [**Malak Khouja**](https://github.com/malakkhouja)  

