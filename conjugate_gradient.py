import numpy as np
#import matplotlib.pyplot as plt

#A = np.array([[4, 1], [1, 3]])
#b = np.array([1, 2])


A = np.diag(4*np.ones(16))
A += np.diag(-np.ones(15), 1)
A += np.diag(-np.ones(15), -1)
A += np.diag(-np.ones(12), -4)
A += np.diag(-np.ones(12), 4)

b = np.arange(1, 10)
b = np.append(b, np.arange(0, 7))

#A = np.array([[4, 1], [1, 3]])
#b = np.array([1, 2]).T

tol = 1e-8


print("\nA:\n", A)
print("\nb:", b)
print("\ntolerance:", tol)

def grad_f(x, A, b):
	return A@x - b

x = np.zeros(b.shape[0])

r = b - A@x
p = r


n = 0

while 1:
	n += 1
	#print(n)	
	alpha = (r.T@r) / (p.T@A@p)
	x += alpha*p
	r_n = r - alpha*A@p
	#print(r_n.T@r_n)
	
	if np.linalg.norm(r_n) < tol:
		break
	
	
	beta = (r_n.T@r_n) / (r.T@r)
	p = r_n + beta*p
	r = r_n
	
	if n == b.shape[0]:
		print("x after theoretical maximum needed iterations", x)
		print("Numerical error at maximum needed iterations", np.linal.norm(r_n))
	
	#print(n)
	#print(r_n)
	
print("\nx once error is less than tolerance:\n", x)
print("\nError:", np.linalg.norm(r_n))
print("\nNumber of iterations taken:", n, "\n")
print(x) 
