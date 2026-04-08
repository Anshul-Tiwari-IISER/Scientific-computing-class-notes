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
x = np.zeros(b.shape[0])

n = 0
error_history = []

while True:
    grad = A @ x - b
    current_error = np.linalg.norm(grad)
    error_history.append(current_error)
    
    if current_error < tol:
        break
        
    p = np.linalg.solve(A, -grad)
    
    alpha = -(grad.T @ p) / (p.T @ A @ p)
    
    x = x + alpha * p
    n += 1

print("\nx once error is less than tolerance:\n", x)
print("\nError:", current_error)
print("\nNumber of iterations taken:", n, "\n")

plt.figure(figsize=(8, 6))
plt.plot(range(len(error_history)), error_history, marker='o', linestyle='-', color='r', markersize=8)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm ||∇f(x)||')
plt.title("Modified Newton's Method Convergence")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.show()
