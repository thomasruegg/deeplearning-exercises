import numpy as np

if __name__ == '__main__':
    A = np.array([[1, 2],
                  [1, 4],
                  [1, 6]])

    b = np.array([1.8, 3.3, 4.1])

    print("exercise A")
    B = np.matmul(A.T, A)
    eigen_result = np.linalg.eig(B)
    l2, l1 = eigen_result.eigenvalues
    print(f'l1: {l1}\nl2: {l2}')

    print("\nexercise B")
    D = np.array([[np.sqrt(l1), 0],
                  [0, np.sqrt(l2)],
                  [0, 0]])
    print(f'D: {D}')

    print("\nexercise C")
    v2, v1 = eigen_result.eigenvectors[:, 0], eigen_result.eigenvectors[:, 1]
    V = np.array([v1, v2])
    print(f'V: {V}')

    print("\nexercise D")
    C = np.matmul(A, np.transpose(A))
    print(C)
    eigen_result = np.linalg.eig(C)
    print(eigen_result.eigenvalues)
    print(eigen_result.eigenvectors)
    U = eigen_result.eigenvectors
    print(f'U: {U}')

    print("\nexercise E")
    U_is_orthogonal = np.allclose(np.matmul(U, U.T), np.identity(3))
    V_is_orthogonal = np.allclose(np.matmul(V, V.T), np.identity(2))
    print(f'U is orthogonal: {U_is_orthogonal}')
    print(f'V is orthogonal: {V_is_orthogonal}')
    A_equals_UDVT = np.allclose(A, np.matmul(np.matmul(U, D), V.T))
    print(f'A = UDV^T: {A_equals_UDVT}')
    Alt_A_equals_UDVT = np.allclose(A, np.matmul(U, np.matmul(D, V.T)))
    print(f'Alt_A = UDV^T: {Alt_A_equals_UDVT}')

    print("\nexercise F")
    D_pseudo_inverse = np.linalg.pinv(D)
    print(f'D_pseudo_inverse: {D_pseudo_inverse}')

    print("\nexercise G")
    A_pseudo_inverse = np.matmul(np.matmul(V, D_pseudo_inverse), U.T)
    print(f'A_pseudo_inverse: {A_pseudo_inverse}')

    print("\nexercise H")
    most_reasonable_solution_x = np.matmul(A_pseudo_inverse, b)
    print(f'most_reasonable_solution_x: {most_reasonable_solution_x}')
