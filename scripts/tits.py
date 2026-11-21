import numpy as np
from alg import *

def invariant_forms(gens, dim, tol=1e-9):
    """Symmetric bilinear forms g on R^dim with X^T g + g X = 0 for all X in gens."""
    idx = [(i, j) for i in range(dim) for j in range(i, dim)]
    basisS = []
    for (i, j) in idx:
        S = np.zeros((dim, dim)); S[i, j] = 1; S[j, i] = 1; basisS.append(S)
    rows = [np.array([(X.T @ S + S @ X).ravel() for S in basisS]).T for X in gens]
    A = np.vstack(rows)
    u, s, vt = np.linalg.svd(A)
    null = vt[np.sum(s > tol):]
    return [sum(c * S for c, S in zip(v, basisS)) for v in null]

class Tits:
    def __init__(self, CA, CJ, trJ, derA_full=True, Aprime=None, deg=3):
        self.CA, self.CJ = CA, CJ
        n, N = CA.shape[0], CJ.shape[0]
        self.n, self.N = n, N
        dA = derivations(CA)
        self.derA = dA
        # A' : orthogonal complement of 1 for Euclidean form (coordinates), as Chester et al.
        if Aprime is None:
            Aprime = np.eye(n)[1:]
        self.Ap = np.array(Aprime)          # rows: basis vectors of A'
        # J': traceless part w.r.t. trace
        one = np.linalg.lstsq(np.array([trJ]), [3.0], rcond=None)[0]
        # identity element of J
        e = self.jordan_unit()
        self.e = e
        P = np.eye(N) - np.outer(e, trJ) / deg   # projector onto traceless part
        u, s, vt = np.linalg.svd(P)
        self.Jp = vt[: np.sum(s > 1e-9)]      # rows basis of J'
        self.trJ = trJ
        self.P = P
        LJ = [Lmat(CJ, x) for x in np.eye(N)]
        self.derJ = span_basis([a @ b - b @ a for a in LJ for b in LJ])
        # forms
        self.tform = np.array([[trJ @ mul(CJ, x, y) for y in np.eye(N)] for x in np.eye(N)])

    def jordan_unit(self):
        N = self.N
        # find e with L_e = id
        M = np.array([Lmat(self.CJ, x).ravel() for x in np.eye(N)]).T
        e = np.linalg.lstsq(M, np.eye(N).ravel(), rcond=None)[0]
        assert np.allclose(M @ e, np.eye(N).ravel())
        return e

    def Dab(self, a, b):
        C = self.CA
        La, Lb, Ra, Rb = Lmat(C, a), Lmat(C, b), Rmat(C, a), Rmat(C, b)
        c = lambda X, Y: X @ Y - Y @ X
        return c(La, Lb) + c(La, Rb) + c(Ra, Rb)

    def setup_middle(self, gA):
        """gA: symmetric form on A (n x n), used as <a,b>."""
        self.gA = gA

    # element: (D (n x n), X (n x N) tensor in A (x) J, E (N x N))
    def bracket(self, u, v, alpha, beta, gamma):
        D1, X1, E1 = u; D2, X2, E2 = v
        n, N, CA, CJ = self.n, self.N, self.CA, self.CJ
        D = D1 @ D2 - D2 @ D1
        E = E1 @ E2 - E2 @ E1
        X = D1 @ X2 - D2 @ X1 + X2 @ E1.T * 0  # placeholder fixed below
        X = D1 @ X2 - D2 @ X1 + X1 @ (-E2.T) * 0
        # derivation of J acts on J index: X -> X E^T
        X = D1 @ X2 - D2 @ X1 + X2 @ E1.T - X1 @ E2.T
        # middle x middle (bilinear in the tensors)
        # X1 = sum_{i,j} X1[i,j] e_i (x) f_j
        # der_A part: alpha * sum t(f_j,f_l) D_{e_i,e_k} X1[i,j] X2[k,l]
        T = X1 @ self.tform @ X2.T          # (i,k) weights
        for i in range(n):
            for k in range(n):
                if abs(T[i, k]) > 1e-14:
                    D = D + alpha * T[i, k] * self.Dab_cache[i][k]
        G = X1.T @ self.gA @ X2             # (j,l)
        for j in range(N):
            for l in range(N):
                if abs(G[j, l]) > 1e-14:
                    E = E + beta * G[j, l] * self.LL_cache[j][l]
        # gamma * [a,b] (x) (x o y)'
        # bilinear: sum X1[i,j] X2[k,l] comm(e_i,e_k) (x) P(f_j o f_l)
        comm = self.CA - self.CA.transpose(1, 0, 2)    # (i,k,out)
        jp = np.einsum('jlm,pm->jlp', self.CJ, self.P)  # P(f_j o f_l)
        X = X + gamma * np.einsum('ij,kl,ikr,jlp->rp', X1, X2, comm, jp)
        return (D, X, E)

    def caches(self):
        n, N = self.n, self.N
        self.Dab_cache = [[self.Dab(np.eye(n)[i], np.eye(n)[k]) for k in range(n)] for i in range(n)]
        LJ = [Lmat(self.CJ, x) for x in np.eye(N)]
        self.LL_cache = [[LJ[j] @ LJ[l] - LJ[l] @ LJ[j] for l in range(N)] for j in range(N)]

    def random_element(self, rng):
        D = sum(rng.normal() * d for d in self.derA)
        E = sum(rng.normal() * d for d in self.derJ)
        a = rng.normal(size=(len(self.Ap), len(self.Jp)))
        X = self.Ap.T @ a @ self.Jp
        return (D, X, E)

def add(u, v, s=1.0): return tuple(a + s * b for a, b in zip(u, v))
def flat(u): return np.concatenate([x.ravel() for x in u])

def jacobi(T, u, v, w, p):
    b = lambda x, y: T.bracket(x, y, *p)
    r = add(add(b(b(u, v), w), b(b(v, w), u)), b(b(w, u), v))
    return flat(r)
