from __main__ import *


flag_deg = False
flag_latex = False

def fillDiagWithArray(A, a):
    for i in range(0, a.shape[0]):
        A[i, i] = a[i]
    return A


def matrixFormatLaxtex(M, num):
    C = np.zeros(M.shape)
    for row in range(0, M.shape[0]):
        for col in range(0, M.shape[1]):
            print(str(np.around(M[row, col], decimals=num)), end="& ")
        print("\\\\")


def homo(A):

    u, s, v = np.linalg.svd(A)  # default SVD function for reference and comparison to custom SVD
    s = fillDiagWithArray(np.zeros(A.shape), s)
    np.around(s, decimals=1)

    # U, S, V, x = SVD_custom(A)  # custom SVD function
    U = u
    S = s
    V = v
    # print('U:', U)
    # print('S:', S)
    # print('V:', V)

    x = v[v.shape[1] - 1,
        :]  # The last row of v is the right-singular vector corresponding to a singular value of A that is zero
    H = np.reshape(x, (3, 3))

    if flag_deg:
        print("A: " + str(A.shape))
        print("A: " + str(A))
        print("The product of USV is " + str(np.dot(u, np.dot(s, v))))
        print("which similar to A")
        print("x: " + str(x))
        print("To verify the result: ")
        print("A*x is" + str(A.dot(x)))
        print("")
    if flag_latex:
        # print("U: " + str(np.around(u, decimals=3)))
        print("U: ")
        matrixFormatLaxtex(u, 3)

        # print("S: " + str(np.around(s, decimals=0)))
        print("S: ")
        matrixFormatLaxtex(s, 3)

        # print("V: " + str(np.around(v, decimals=3)))
        print("V: ")
        matrixFormatLaxtex(v, 3)
    return H



