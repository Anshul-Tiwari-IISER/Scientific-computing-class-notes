#f(x) = x1^2 + x2^4
#x_0 = [1,1]\

import numpy as np
import matplotlib.pyplot as plt

x = np.array([-10.0,20.0])

def f(x):
	return (x[0] + 4)**2 + x[1]**4
	
def grad_f(x):
	return np.array([2*(x[0] + 4), 4*(x[1]**3)])
	
def grad2_f(x):
	return 2 + 12*(x[1]**2)

def hess_f(x):
	return np.array([[2, 0],
			[0, 12*(x[1]**2)]])

def alpha_k(x_k, p_k):
	alpha = 0.0
	for i in range(10):
		alpha = alpha - (dphi(x_k, alpha, p_k) / d2phi(x_k, alpha, p_k))
		#print(d2phi(x_k, alpha, p_k))
	
	#print("alpha", alpha)
	return alpha
	
	
def phi(x_k, alpha, p_k):
	return f(x_k + alpha*p_k)

def dphi(x_k, alpha, p_k):
	return grad_f(x_k + alpha*p_k) @ p_k
	
def d2phi(x_k, alpha, p_k):
	return p_k.T * grad2_f(x_k) @ p_k

n = 0
tol = 1e-8
p = -grad_f(x)
X = []
while np.linalg.norm(p) > tol and n < 1000:
	p = -grad_f(x)
	#print(p)

	alpha = alpha_k(x, p)
	x = x + alpha * p
	
	X.append(x)
	#print(x)
	n+= 1
	
X = np.array(X)
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
plt.plot(X[:,0], X[:,1])
plt.scatter(X[-1, 0], X[-1, 1], color='r')
plt.show()
print(x)
print(n)
