import numpy as np
from alg import *
from tits import *
from run_sanity import conj_std
C,H,O=complex_table(),quaternion_table(),octonion_table()
def nullspace(T,gAs,ntrip=6,seed=1):
    rng=np.random.default_rng(seed)
    trips=[(T.random_element(rng),T.random_element(rng),T.random_element(rng)) for _ in range(ntrip)]
    def R(alpha,betas):
        T.setup_middle(sum(b*g for b,g in zip(betas,gAs)))
        return np.concatenate([jacobi(T,u,v,w,(alpha,1.0,1.0)) for (u,v,w) in trips])
    m=len(gAs); R0=R(0,[0]*m)
    M=np.array([R(1,[0]*m)-R0]+[R(0,list(np.eye(m)[k]))-R0 for k in range(m)]).T
    s=np.linalg.svd(M,compute_uv=False)
    _,_,vt=np.linalg.svd(M)
    return np.linalg.norm(R0), s, vt[np.sum(s>1e-8*s[0]):]
for name,CA,CB in [('T(H,J2(O))',H,O),('T(O,J2(H))',O,H),('L2(CxH,O)',tensor(C,H),O),('L2(CxO,H)',tensor(C,O),H)]:
    CJ,tr,_=jordan_H3(CB,conj_std(CB.shape[0]),m=2)
    T=Tits(CA,CJ,tr,deg=2); T.caches()
    forms=invariant_forms(T.derA,CA.shape[0])
    # express forms restricted to A' for readability
    r0,s,null=nullspace(T,forms)
    print(name,'dim',len(T.derA)+len(T.Ap)*len(T.Jp)+len(T.derJ),'|R0|=',r0,'sing',np.round(s,4))
    for v in null:
        print('   null vector (alpha, beta_k):',np.round(v,4))
        g=sum(b*f for b,f in zip(v[1:],forms)); Ap=T.Ap
        print('   form on A\':\n',np.round(Ap@g@Ap.T,4))
