from fenics import *
import numpy as np

L = 1.0
N = 1000
mesh = IntervalMesh(N,0.0,L)

fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

d = mesh.topology().dim()
k = Constant(1.0,name="k")
rho = Constant(0.01*L,name="rho")

nu = (4.0*k-d)/2.0
kappa = sqrt(8*nu) / rho

f = Function(V,name="rhs")

u,v = TrialFunction(V),TestFunction(V)
a = inner(grad(u),grad(v))*dx + kappa**2*u*v*dx
L = f*v*dx

M_lumped = assemble(action(v*u*dx,Constant(1.0)))

u = Function(V,name="randfield")

# set the rhs to pointwise random values
#

sparam = {"linear_solver":"cg","preconditioner":"hypre_amg"}

import math as m
sigma2  = m.pow(4*m.pi,-d/2.0)*m.pow(float(kappa),-2*float(nu))
sigma2 *= m.gamma(float(nu))/m.gamma(float(nu)+d/2)

kmax = 5
Nsamp = 1000
R = np.empty((Nsamp,V.dim()))
F = np.empty((Nsamp,V.dim()))
A = assemble(a)
b = assemble(L)
for i in range(Nsamp):
    k.assign(kmax)
    b.set_local( 1/m.sqrt(sigma2) * np.sqrt(M_lumped.get_local()) * np.random.randn(V.dim()) )
    solve(A, u.vector(), b)

    si2old = sigma2

    for kk in range(2,kmax+1):
        #k.assign(kk)

        si2  = m.pow(4*m.pi,-d/2.0)*m.pow(float(kappa),-2*float(nu))
        si2 *= m.gamma(float(nu))/m.gamma(float(nu)+d/2)

        assemble(m.sqrt(si2old/si2) * u*v*dx, tensor=b)
        assemble(a, tensor=A)

        solve(A, u.vector(), b)

        si2old = si2

    R[i,:] = u.vector().get_local()
    F[i,:] = b.get_local()

print(np.median(R.mean(axis=0)))
print(np.median(R.var(axis=0)))

#import matplotlib.pyplot as plt
#plt.hist(R.var(axis=0),50)
#plt.savefig("aaa1d.pdf")

