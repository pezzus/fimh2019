from sampler import RandomFieldSPDE
from fenics import *
import numpy as np
import os,sys

# prepare the mesh for simulation
#
if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} mesh.xml")

mesh = Mesh(sys.argv[1])

fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

s1 = Function(V,name="s1")
s2 = Function(V,name="s2")
s = Function(V,name="s")

sampl1 = RandomFieldSPDE(V,N=5,rho=10.0)
sampl2 = RandomFieldSPDE(V,N=5,rho=1.0)

with XDMFFile("samples.xdmf") as f:
    f.parameters['functions_share_mesh'] = True
    f.parameters['rewrite_function_mesh'] = False
    for i,(ss1,ss2) in enumerate(zip(sampl1.sample(5),sampl2.sample(5))):
        print(f"Sample {i}")
        s1.assign(ss1)
        s2.assign(ss2)
        s.assign(0.5*s1+0.5*s2)
        f.write(s1,float(i))
        f.write(s2,float(i))
        f.write(s,float(i))

