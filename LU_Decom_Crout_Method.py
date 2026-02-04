print("LU Decomposition using Crout's Method")
print("This program decomposes A into L and U such that A = L * U")

n = int(input("Enter order of square matrix (n): "))

print("Enter the elements of matrix A (each row in one line, space separated):")
A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

L = [[0.0 for _ in range(n)] for _ in range(n)]
U = [[0.0 for _ in range(n)] for _ in range(n)]

for i in range(n):
    U[i][i] = 1.0

singular = False

for j in range(n):
    for i in range(j, n):
        s = 0.0
        for k in range(j):
            s += L[i][k] * U[k][j]
        L[i][j] = A[i][j] - s

    if abs(L[j][j]) < 1e-12:
        singular = True
        break

    for i in range(j + 1, n):
        s = 0.0
        for k in range(j):
            s += L[j][k] * U[k][i]
        U[j][i] = (A[j][i] - s) / L[j][j]

if singular:
    print("LU Decomposition not possible (zero pivot encountered)")
else:
    print("Lower Triangular Matrix L:")
    for i in range(n):
        for j in range(n):
            print(L[i][j], end=" ")
        print()

    print("Upper Triangular Matrix U:")
    for i in range(n):
        for j in range(n):
            print(U[i][j], end=" ")
        print()
