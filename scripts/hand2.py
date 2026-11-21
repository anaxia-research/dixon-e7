import sympy as sp
def S(i,j):
    M=sp.zeros(3); M[i,j]=1; M[j,i]=1; return M
o=lambda X,Y:(X*Y+Y*X)/2
pr=lambda X:X-X.trace()/3*sp.eye(3)
t=lambda X,Y:o(X,Y).trace()
x,y,z=S(0,1),S(1,2),S(0,2)
print('y o z =',o(y,z),' z o x =',o(z,x))
print('[L_y,L_z]x with x=S12,y=S23,z=S13:', o(y,o(z,x))-o(z,o(y,x)))
print('F parts', pr(o(pr(o(y,z)),x)), pr(o(pr(o(z,x)),y)))
print('t(y,z),t(z,x)',t(y,z),t(z,x))
# gamma=0 example
X,Y=S(0,1),S(1,2); print('t(Y,Y),t(Y,X)',t(Y,Y),t(Y,X))
# 2x2
D=sp.diag(1,-1); s=sp.Matrix([[0,1],[1,0]])
o2=lambda X,Y:(X*Y+Y*X)/2
print('J2: [L_D,L_s]D =', o2(D,o2(s,D))-o2(s,o2(D,D)))
