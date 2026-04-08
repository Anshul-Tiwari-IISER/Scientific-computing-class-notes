import numpy as np
import matplotlib.pyplot as plt

m = int(input("Enter number of rows: "))


A = np.zeros([m,m])
b = np.zeros(m)

for i in range(m):
	for j in range(m):
		A[i, j] = float(input(f"A[{i}, {j}]: "))
		
for i in range(m):
	b[i] = float(input(f"b[{i}]: "))

#def f(A, b, x):
#	return 0.5 * x.T @ A @ x- b.T @ x
	
	

def grad_f(A, b, x):
	return A@x - b 


x = np.ones(m)

X = [x]

p = -grad_f(A, b, x)
n = 0

tol = float(input("Tolerance:"))

if not tol:
	tol = 1e-8


while np.linalg.norm(p) > tol and n < 500:
	p = -grad_f(A, b, x)
	if p@A@p == 0:
		break
	alpha = (p @ p) / (p@A@p)
	x = x + alpha * p
	
	X.append(x)
	n+= 1

print("Solution:", x)
print("Error:", A@x-b)

print("Number of iterations taken:", n)

X = np.array(X)

if m == 2:
	#plt.xlim(-3, 3)
	#plt.ylim(-3, 3)
	plt.plot(X[:, 0], X[:, 1], '-o')
	plt.scatter(x[0], x[1], color='red', s=30, zorder=2)
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.title("Steepest descent method")
	plt.axis("on")
	plt.savefig('steepest_descent.pdf')
	plt.show()
