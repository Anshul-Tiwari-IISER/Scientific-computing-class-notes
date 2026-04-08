import numpy as np
import matplotlib.pyplot as plt

A = np.diag(4 * np.ones(16))
A += np.diag(-np.ones(15), 1)
A += np.diag(-np.ones(15), -1)
A += np.diag(-np.ones(12), -4)
A += np.diag(-np.ones(12), 4)

b = np.arange(1, 10)
b = np.append(b, np.arange(0, 7))

tol = 1e-8

print("\nA:\n", A)
print("\nb:", b)
print("\ntolerance:", tol)

def grad_f(x, A, b):
    return A @ x - b

x = np.zeros(b.shape[0])

r = b - A @ x
p = r

n = 0
error_history = []

while True:
    n += 1
    
    alpha = (r.T @ r) / (p.T @ A @ p)
    x = x + alpha * p
    r_n = r - alpha * A @ p
    
    current_error = np.linalg.norm(r_n)
    error_history.append(current_error)
    
    if current_error < tol:
        break
        
    beta = (r_n.T @ r_n) / (r.T @ r)
    p = r_n + beta * p
    r = r_n
    
    if n == b.shape[0]:
        print("x after theoretical maximum needed iterations", x)
        print("Numerical error at maximum needed iterations", current_error)

print("\nx once error is less than tolerance:\n", x)
print("\nError:", current_error)
print("\nNumber of iterations taken:", n, "\n")
print(x)

plt.figure(figsize=(8, 6))
plt.plot(range(1, n + 1), error_history, marker='o', linestyle='-', color='b')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm ||r||')
plt.title('Convergence History')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.show()
