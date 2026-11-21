# Variant: drop R*I from A' (A' = C (x) Im A), real Jordan factor.
import numpy as np
from alg import *
from tits import *
from run_sanity import conj_std, fit
C,H,O=complex_table(),quaternion_table(),octonion_table()
for name,CA,CB,nA in [('C(x)H with J3(O), A\'=C(x)ImH',tensor(C,H),O,4),('C(x)O with J3(H), A\'=C(x)ImO',tensor(C,O),H,8)]:
    n=CA.shape[0]
    Ap=np.eye(n)[[k for k in range(n) if k not in (0,nA)]]
    CJ,tr,_=jordan_H3(CB,conj_std(CB.shape[0]))
    T=Tits(CA,CJ,tr,Aprime=Ap); T.caches()
    forms=invariant_forms(T.derA,n)
    sol,res,aff=fit(T,forms)
    print(name,'dim',len(T.derA)+len(T.Ap)*len(T.Jp)+len(T.derJ),'best',np.round(sol,3),'rel residual',res)
