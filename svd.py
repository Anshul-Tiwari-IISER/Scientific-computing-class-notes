import numpy as np

def my_svd(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    ATA = A.T @ A

    eigvals, eigvecs = np.linalg.eig(ATA)

    eigvals = np.clip(eigvals, 0, None)
    singular_values = np.sqrt(eigvals)

    idx = np.argsort(singular_values)[::-1]
    S = singular_values[idx]
    V = eigvecs[:, idx]

    r = len(S)
    U = np.zeros((m, r))
    for i in range(r):
        if S[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / S[i]

    return U, S, V.T


m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

print("Enter the matrix row by row (space-separated):")
A = []
for i in range(m):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)

U, S, VT = my_svd(A)

print("\nMatrix A:")
print(A)

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

