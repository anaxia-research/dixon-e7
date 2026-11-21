import numpy as np
from alg import *
from tits import *
from run_sanity import fit
C,H,O=complex_table(),quaternion_table(),octonion_table()
CB=tensor(C,H)
sigma=np.kron(np.eye(2),np.diag([1,-1,-1,-1.]))   # C-linear quaternionic conjugation
CJ,tr,basis=jordan_H3(CB,sigma)
print('dim H3(CxH,sigma)=',CJ.shape[0])
trIm=np.array([sum(b[i,i,4] for i in range(3)) for b in basis])
for variant in ['Re t = 0 (29-dim)','t_C = 0 (28-dim)']:
    T=Tits(O,CJ,tr)
    if variant.startswith('t_C'):
        e=T.e
        # I*e : multiply each diag entry by I -> in basis coords
        Bm=np.array([b.ravel() for b in basis]).T
        M=np.zeros((3,3,8)); 
        for i in range(3): M[i,i,4]=1
        Ie=np.linalg.lstsq(Bm,M.ravel(),rcond=None)[0]
        P=np.eye(CJ.shape[0])-np.outer(e,tr)/3-np.outer(Ie,trIm)/3
        u,s,vt=np.linalg.svd(P); T.Jp=vt[:np.sum(s>1e-9)]; T.P=P
    T.caches()
    forms=invariant_forms(T.derA,8)
    print(variant,'dims',len(T.derA),len(T.Ap),len(T.Jp),len(T.derJ),'total',len(T.derA)+len(T.Ap)*len(T.Jp)+len(T.derJ))
    sol,res,aff=fit(T,forms)
    print('   best',np.round(sol,4),'rel residual',res,'aff',aff)
