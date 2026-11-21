import sympy as sp
def S(i,j):
    M=sp.zeros(3); M[i,j]=1; M[j,i]=1; return M
o=lambda X,Y:(X*Y+Y*X)/2
pr=lambda X:X-X.trace()/3*sp.eye(3)
t=lambda X,Y:o(X,Y).trace()
a=sp.symbols('alpha')
x,y,z=S(0,1),S(1,2),S(0,2)
F=a*(t(y,z)*x-t(z,x)*y)+pr(o(pr(o(y,z)),x))-pr(o(pr(o(z,x)),y))
print(t(y,z),t(z,x)); print(pr(o(y,z)), pr(o(pr(o(y,z)),x)), pr(o(z,x)), pr(o(pr(o(z,x)),y)), F)
