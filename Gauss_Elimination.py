print("\nGauss Elimination Method with Partial Pivoting")
print("This program solves a system of linear equations Ax = b\n")

n = int(input("Enter number of variables (n): "))

print("Enter the coefficients of matrix A (each row in one line, space separated):")
A = []
for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

print("Enter the constants vector b (one value per line):")
b = []
for i in range(n):
    b.append(float(input()))

singular = False

for i in range(n):
    max_row = i
    for k in range(i + 1, n):
        if abs(A[k][i]) > abs(A[max_row][i]):
            max_row = k

    A[i], A[max_row] = A[max_row], A[i]
    b[i], b[max_row] = b[max_row], b[i]

    if abs(A[i][i]) < 1e-12:
        singular = True
        break

    for k in range(i + 1, n):
        factor = A[k][i] / A[i][i]
        for j in range(i, n):
            A[k][j] -= factor * A[i][j]
        b[k] -= factor * b[i]

if singular:
    print("No unique solution exists for the given system")
else:
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]

    print("Solution of the system (values of variables):")
    for i in range(n):
        print("x" + str(i + 1) + " =", x[i])
