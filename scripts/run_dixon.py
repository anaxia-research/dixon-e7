import numpy as np
from alg import *
from tits import *
from run_sanity import build, conj_std, fit

C, H, O = complex_table(), quaternion_table(), octonion_table()
cases = [('L3(C(x)H, O)', tensor(C, H), O), ('L3(C(x)O, H)', tensor(C, O), H)]
for name, CA, CB in cases:
    T = build(CA, CB, conj_std(CB.shape[0]))
    dim = len(T.derA) + len(T.Ap) * len(T.Jp) + len(T.derJ)
    forms = invariant_forms(T.derA, CA.shape[0])
    print(name, 'dims der A =', len(T.derA), "A' =", len(T.Ap), "J' =", len(T.Jp), 'der J =', len(T.derJ), 'total', dim)
    print('  invariant symmetric forms on A:', len(forms))
    # does every derivation preserve A' (complement of 1)?
    print('  der A kills 1:', all(np.allclose(D[:, 0], 0) for D in T.derA),
          ' preserves span(A\'):', all(np.allclose(D[0, :], 0) for D in T.derA))
    # Euclidean form
    sol, res, aff = fit(T, [np.eye(CA.shape[0])])
    print('  Euclidean <,>: best (alpha,beta)=', sol, ' rel. Jacobi residual', res)
    sol, res, aff = fit(T, forms)
    print('  all invariant <,>: best params=', np.round(sol, 4), ' rel. Jacobi residual', res, 'affinity', aff)
