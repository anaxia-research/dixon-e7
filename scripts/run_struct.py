import numpy as np
from alg import *
from tits import *
from run_sanity import conj_std
H,O=quaternion_table(),octonion_table()

def structure(T, alpha, beta, gamma, gA):
    T.setup_middle(gA)
    els=[]
    n,N=T.n,T.N
    for D in T.derA: els.append((D,np.zeros((n,N)),np.zeros((N,N))))
    for a in T.Ap:
        for x in T.Jp: els.append((np.zeros((n,n)),np.outer(a,x),np.zeros((N,N))))
    for E in T.derJ: els.append((np.zeros((n,n)),np.zeros((n,N)),E))
    B=np.array([flat(e) for e in els]).T
    Bp=np.linalg.pinv(B)
    d=len(els); S=np.zeros((d,d,d))
    for i in range(d):
        for j in range(i+1,d):
            r=flat(T.bracket(els[i],els[j],alpha,beta,gamma)); c=Bp@r
            assert np.allclose(B@c,r,atol=1e-8), 'not closed'
            S[i,j]=c; S[j,i]=-c
    return S, els

def analyse(S, seed=0):
    d=S.shape[0]; rng=np.random.default_rng(seed)
    ad=np.array([S[i].T for i in range(d)])
    K=np.einsum('iab,jba->ij',ad,ad)
    ev=np.linalg.eigvalsh(K)
    X=np.einsum('i,iab->ab',rng.normal(size=d),ad)
    w,V=np.linalg.eig(X); Vi=np.linalg.inv(V)
    zero=np.where(np.abs(w)<1e-7)[0]; nz=np.where(np.abs(w)>=1e-7)[0]
    # C = V M V^{-1}, M block-diagonal: free block on zero eigenspace, diagonal elsewhere
    params=[]
    for a in zero:
        for b in zero: params.append((a,b))
    for a in nz: params.append((a,a))
    cols=[]
    Ys=[np.einsum('i,iab->ab',rng.normal(size=d),ad) for _ in range(2)]
    for (a,b) in params:
        M=np.zeros((d,d),complex); M[a,b]=1; C=V@M@Vi
        cols.append(np.concatenate([(C@Y-Y@C).ravel() for Y in Ys]))
    A=np.array(cols).T
    sv=np.linalg.svd(A,compute_uv=False)
    return d, int(np.sum(ev>1e-8)), int(np.sum(ev<-1e-8)), int(np.sum(np.abs(ev)<=1e-8)), len(zero), int(np.sum(sv<1e-8*sv[0]))

if __name__=='__main__':
  for name,CA,CB,m in [('T(H,J3(O))',H,O,3),('T(O,J3(H))',O,H,3),('T(H,J2(O))',H,O,2)]:
      CJ,tr,basis=jordan_H3(CB,conj_std(CB.shape[0]),m=m)
      T=Tits(CA,CJ,tr,deg=m); T.caches()
      if m==3: S,els=structure(T,1/3,-4,1,np.eye(CA.shape[0]))
      else:
          # find nontrivial solution for J2 with Euclidean form: alpha from null space
          S,els=structure(T,-1/8,1.0,1.0,np.eye(CA.shape[0]))
      rng=np.random.default_rng(5); d=S.shape[0]
      jac=0
      for _ in range(3):
          u,v,w=rng.normal(size=(3,d)); b=lambda x,y: np.einsum('i,j,ijk->k',x,y,S)
          jac=max(jac,np.linalg.norm(b(b(u,v),w)+b(b(v,w),u)+b(b(w,u),v)))
      d,pos,neg,zero,rank,comm=analyse(S)
      print(name,'dim',d,'Jacobi err',jac,'Killing (+,-,0)=',(pos,neg,zero),'rank',rank,'commutant dim',comm)
