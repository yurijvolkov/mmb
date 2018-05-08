import time
import math
import numpy as np

def straightway(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError('Shapes are incorrect {A.shape[1] != B.shape[0]}')
    
    C = np.ndarray( (A.shape[0], B.shape[1]) )
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C_ij = 0
            for k in range(A.shape[1]):
                C_ij += A[i, k] * B[k, j]
            C[i, j] = C_ij

    return C

def strassen(A, B):
    init_shape_A = A.shape
    init_shape_B = B.shape

    def _wrap_matrix_(M, n):
        if M.shape[0] != n:
            diff = n - M.shape[0]
            M = np.append(M, [[0] * M.shape[1]] * diff, axis=0)
        if M.shape[1] != n:
            diff = n - M.shape[1]
            M = np.append(M, [[0] * diff] * M.shape[0], axis=1)
        return M
    
    if ( not( A.shape[0] == A.shape[1] == B.shape[1] )
         or( np.log2(A.shape[0]) % 1 != 0) ):
        n = max(A.shape[0], A.shape[1], B.shape[1])
        n = 2 ** math.ceil(np.log2(n))
        A = _wrap_matrix_(A, n)
        B = _wrap_matrix_(B, n)

    if A.shape[0] == 1:
        return np.array( [[ A[0, 0] * B[0, 0] ]] )
    else:
        n = A.shape[0]
        a11 = A[:n//2, :n//2]
        a12 = A[:n//2, n//2:]
        a21 = A[n//2:, :n//2]
        a22 = A[n//2:, n//2:]
        b11 = B[:n//2, :n//2]
        b12 = B[:n//2, n//2:]
        b21 = B[n//2:, :n//2]
        b22 = B[n//2:, n//2:]

        k1 = strassen(a11 + a22, b11 + b22)
        k2 = strassen(a21 + a22, b11)
        k3 = strassen(a11, b12 - b22)
        k4 = strassen(a22, b21 - b11)
        k5 = strassen(a11 + a12, b22)
        k6 = strassen(a21 - a11, b11 + b12)
        k7 = strassen(a12 - a22, b21 + b22)

        c11 = k1 + k4 - k5 + k7
        c12 = k3 + k5
        c21 = k2 + k4
        c22 = k1 + k3 - k2 + k6

        C = np.block([ [c11, c12],
                       [c21, c22] ])
        
        return C[:init_shape_A[0], :init_shape_B[1]]

def winograd(A, B):
    init_shape_A = A.shape
    init_shape_B = B.shape

    if A.shape[1] % 2 == 1:
        A = np.append(A, [[0]] * A.shape[0], axis=1)
        B = np.append(B, [[0] * A.shape[0]], axis=0)

    m = A.shape[1] // 2

    a_term = np.sum( [ [A[i, 2*k - 1] * A[i, 2*k] for k in range(m) ]
                                                  for i in range(A.shape[0]) ],
                    axis=1)
    b_term = np.sum( [ [B[2*k, j] * B[2*k - 1, j] for k in range(m) ]
                                                  for j in range(B.shape[1]) ],
                    axis=1)

    C = np.ndarray((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C_ij = 0
            for k in range(0, m):
                C_ij += (A[i, 2*k - 1] + B[2*k, j]) * (B[2*k-1, j] + A[i, 2*k])
            C_ij -= a_term[i] + b_term[j]
            C[i,j] = C_ij
    
    return C[:init_shape_A[0], :init_shape_B[1]]


def check_multiplication(func):
    A = np.random.rand(17, 20)
    B = np.random.rand(20, 31)

    C_numpy = A @ B
    C_custom = func(A, B)

    error = abs(C_numpy - C_custom)

    if np.mean(error, (0, 1)) > 1e-5:
        return False
    return True

if __name__ == "__main__":
    print(f"Straightway correct: {check_multiplication(straightway)}") 
    print(f"Strassen correct: {check_multiplication(strassen)}") 
    print(f"Winograd correct: {check_multiplication(winograd)}")
   
#   A = np.random.rand(20, 20)
#   B = np.random.rand(20, 20)





