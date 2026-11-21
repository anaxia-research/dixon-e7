import sys, numpy as np
from alg import *
from tits import *

def build(CA, CB, conjB):
    CJ, tr, _ = jordan_H3(CB, conjB)
    T = Tits(CA, CJ, tr)
    T.caches()
    return T

def conj_std(n):
    c = -np.eye(n); c[0, 0] = 1; return c

def fit(T, gAs, ntrip=6, seed=0):
    """Jacobi residual is affine in (alpha, beta_k) at gamma=1, where <,> = sum beta_k gA_k."""
    rng = np.random.default_rng(seed)
    trips = [(T.random_element(rng), T.random_element(rng), T.random_element(rng)) for _ in range(ntrip)]
    def R(alpha, betas):
        T.setup_middle(sum(b * g for b, g in zip(betas, gAs)))
        return np.concatenate([jacobi(T, u, v, w, (alpha, 1.0, 1.0)) for (u, v, w) in trips])
    m = len(gAs)
    R0 = R(0.0, [0.0] * m)
    cols = [R(1.0, [0.0] * m) - R0] + [R(0.0, list(np.eye(m)[k])) - R0 for k in range(m)]
    M = np.array(cols).T
    sol, *_ = np.linalg.lstsq(M, -R0, rcond=None)
    res = np.linalg.norm(M @ sol + R0) / max(1, np.linalg.norm(R0))
    # affinity check
    rp = rng.normal(size=m + 1)
    lin = R0 + M @ rp
    act = R(rp[0], list(rp[1:]))
    aff = np.linalg.norm(lin - act) / np.linalg.norm(act)
    return sol, res, aff

if __name__ == '__main__':
    H, O = quaternion_table(), octonion_table()
    for name, CA, CB in [('T(H,J3(O))', H, O), ('T(O,J3(H))', O, H), ('T(H,J3(H))', H, H)]:
        T = build(CA, CB, conj_std(CB.shape[0]))
        dim = len(T.derA) + len(T.Ap) * len(T.Jp) + len(T.derJ)
        gE = np.eye(CA.shape[0])
        sol, res, aff = fit(T, [gE])
        print(name, 'dims', len(T.derA), len(T.Ap), len(T.Jp), len(T.derJ), 'total', dim,
              'alpha,beta', sol, 'rel.residual', res, 'affinity err', aff)
