import numpy as np
from alg import *
from tits import *
from run_sanity import conj_std
H,O=quaternion_table(),octonion_table()
for name,CA,CB,m in [('T(H,J3(O))',H,O,3),('T(O,J3(H))',O,H,3),('T(H,J2(O))',H,O,2)]:
    CJ,tr,basis=jordan_H3(CB,conj_std(CB.shape[0]),m=m)
    T=Tits(CA,CJ,tr,deg=m)
    Bm=np.array([b.ravel() for b in basis]).T; Bp=np.linalg.pinv(Bm)
    Sg=np.diag([1.]*(m-1)+[-1.])
    th=np.array([Bp@np.einsum('ab,bck,cd->adk',Sg,b,Sg).ravel() for b in basis]).T
    aut=np.abs(np.einsum('ai,bj,abk->ijk',th,th,CJ)-np.einsum('ijk,lk->ijl',CJ,th)).max()
    # J' eigenspaces
    PJ=T.Jp; thJ=PJ@th@np.linalg.pinv(PJ)
    ev=np.linalg.eigvals(thJ); jp=int(np.sum(np.abs(ev-1)<1e-8)); jm=int(np.sum(np.abs(ev+1)<1e-8))
    # derJ eigenspaces under E -> th E th^-1
    DJ=np.array([E.ravel() for E in T.derJ]).T; DJp=np.linalg.pinv(DJ)
    M=np.array([DJp@(th@E@np.linalg.inv(th)).ravel() for E in T.derJ]).T
    ev=np.linalg.eigvals(M); dp=int(np.sum(np.abs(ev-1)<1e-8)); dm=int(np.sum(np.abs(ev+1)<1e-8))
    a=len(T.Ap); dA=len(T.derA)
    print(name,'phi automorphism err %.1e'%aut,"J'+",jp,"J'-",jm,'derJ+',dp,'derJ-',dm,
          '| fixed dim',dA+a*jp+dp,' (-1)-dim',a*jm+dm)
