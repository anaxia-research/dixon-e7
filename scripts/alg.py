"""Real algebras by structure constants, Jordan algebras J3(B), and Tits' construction."""
import numpy as np
from itertools import product

# ---------- composition algebras ----------
def octonion_table():
    # basis e0=1, e1..e7, with e_i e_{i+1} = e_{i+3} (indices mod 7, in 1..7)
    C = np.zeros((8, 8, 8))
    C[0, :, :] = np.eye(8); C[:, 0, :] = np.eye(8)
    for i in range(1, 8):
        C[i, i, 0] = -1
    for i in range(7):
        a, b, c = (i % 7) + 1, ((i + 1) % 7) + 1, ((i + 3) % 7) + 1
        for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)]:
            C[x, y, z] = 1; C[y, x, z] = -1
    return C

def quaternion_table():
    # sub-algebra 1,e1,e2,e4 of O: e1 e2 = e4
    O = octonion_table(); idx = [0, 1, 2, 4]
    return O[np.ix_(idx, idx, idx)]

def real_table(): return np.ones((1, 1, 1))

def complex_table():
    C = np.zeros((2, 2, 2)); C[0,0,0]=1; C[0,1,1]=1; C[1,0,1]=1; C[1,1,0]=-1; return C

def tensor(A, B):
    n, m = A.shape[0], B.shape[0]
    T = np.einsum('ijk,abc->iajbkc', A, B).reshape(n*m, n*m, n*m)
    return T

def mul(C, x, y): return np.einsum('i,j,ijk->k', x, y, C)
def Lmat(C, x): return np.einsum('i,ijk->kj', x, C)   # L_x y
def Rmat(C, x): return np.einsum('j,ijk->ki', x, C)   # R_x y

def derivations(C, tol=1e-9):
    n = C.shape[0]
    # D(e_i e_j) = D(e_i) e_j + e_i D(e_j); unknown D (n x n), D[k,l] = coefficient of e_k in D(e_l)
    rows = []
    for i in range(n):
        for j in range(n):
            # sum_k C[i,j,k] D[:,k] - sum_l D[l,i] C[l,j,:] - sum_l D[l,j] C[i,l,:] = 0
            M = np.zeros((n, n, n))  # (output comp, D row, D col)
            for k in range(n): M[:, :, k] += np.diag(np.full(n, C[i, j, k]))  # D[:,k]
            for l in range(n):
                M[:, l, i] -= C[l, j, :]
                M[:, l, j] -= C[i, l, :]
            rows.append(M.reshape(n, n*n))
    A = np.vstack(rows)
    u, s, vt = np.linalg.svd(A)
    null = vt[np.sum(s > tol):]
    return [v.reshape(n, n) for v in null]

def span_basis(mats, tol=1e-9):
    if not mats: return []
    M = np.array([m.ravel() for m in mats])
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    r = np.sum(s > tol * max(1, s[0]))
    shape = mats[0].shape
    return [vt[i].reshape(shape) for i in range(r)]

# ---------- Jordan algebra H3(B) for a *-algebra B with conjugation conj (matrix) ----------
def jordan_H3(CB, conj, m=3):
    nb = CB.shape[0]
    # Hermitian X: X_ji = conj(X_ij), diagonal fixed by conj
    fixed = np.linalg.svd(conj - np.eye(nb))
    # basis of fixed space of conj
    u, s, vt = fixed
    fix = vt[np.sum(s > 1e-9):]
    basis = []
    for p in range(m):
        for f in fix:
            X = np.zeros((m, m, nb)); X[p, p] = f; basis.append(X)
    for (p, q) in [(p, q) for p in range(m) for q in range(p+1, m)]:
        for k in range(nb):
            X = np.zeros((m, m, nb)); e = np.eye(nb)[k]
            X[p, q] = e; X[q, p] = conj @ e; basis.append(X)
    def matmul(X, Y):
        Z = np.zeros((m, m, nb))
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    Z[i, j] += mul(CB, X[i, k], Y[k, j])
        return Z
    B = np.array([b.ravel() for b in basis]).T  # coords -> flat
    Binv = np.linalg.pinv(B)
    N = len(basis)
    CJ = np.zeros((N, N, N))
    for a in range(N):
        for b in range(N):
            Z = 0.5 * (matmul(basis[a], basis[b]) + matmul(basis[b], basis[a]))
            c = Binv @ Z.ravel()
            assert np.allclose(B @ c, Z.ravel()), "not closed"
            CJ[a, b] = c
    # trace (real part of diagonal sum, component e0)
    tr = np.array([sum(b[i, i, 0] for i in range(m)) for b in basis])
    return CJ, tr, basis
