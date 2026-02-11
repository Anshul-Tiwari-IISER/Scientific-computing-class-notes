import numpy as np

def my_svd(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    ATA = A.T @ A

    eigvals, eigvecs = np.linalg.eigh(ATA)

    eigvals = np.clip(eigvals, 0, None)
    singular_values = np.sqrt(eigvals)

    idx = np.argsort(singular_values)[::-1]
    S = singular_values[idx]
    V = eigvecs[:, idx]

    tol = max(m, n) * np.max(S) * 1e-15
    r = np.sum(S > tol)


    S = S[:r]
    V = V[:, :r]
    
    U = np.zeros((m, r))
    for i in range(r):
            U[:, i] = (A @ V[:, i]) / S[i]

    return U, S, V.T, r


m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

print("Enter the matrix row by row (space-separated):")
A = []
for i in range(m):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)

U, S, VT, r = my_svd(A)

print("\nMatrix A:")
print(A)

print("\nMatrix rank:")
print(r)

print("\nMatrix U:")
print(U)

print("\nSingular values (Σ):")
print(S)

print("\nMatrix V^T:")
print(VT)

A_reconstructed = U @ np.diag(S) @ VT

print("\nReconstructed A (U Σ V^T):")
print(A_reconstructed)

print("\nReconstruction correct?",
      np.allclose(A, A_reconstructed))

