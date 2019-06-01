"""Smoothness parameter for random fields.

We increase N from 1 to 20 and check the effect on the solution.
At each iteration, we also evaluate the norm of the difference
between two iterates. It should decrease (slowly).
"""
from sampler import RandomFieldSPDE
from fenics import *

mesh = UnitSquareMesh(100,100)
fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

with XDMFFile(f"compare_smoothness.xdmf") as ofile:
    ofile.parameters['functions_share_mesh'] = True
    ofile.parameters['rewrite_function_mesh'] = False

    svec = []
    for N in range(1,21):
        sampler = RandomFieldSPDE(V,N=N,rho=0.05)
        sampler.set_seed(14210)
        ss = Function(V,name=f"sample_{N}")
        s = next(sampler.sample(1))
        ss.interpolate(s)
        svec.append(ss)
        ofile.write(s,float(N))

    for N,(s0,s1) in enumerate(zip(svec,svec[1:])):
        err = sqrt(assemble( (s0-s1)**2*dx ))
        print(f"{N:4d} {err:10.5e}")
