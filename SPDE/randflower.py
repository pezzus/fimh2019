from sampler import RandomFieldSPDE
from fenics import *
from ufl import tanh
import numpy as np

mesh = Mesh("flower.xml")
bfun = MeshFunction("size_t",mesh,mesh.topology().dim()-1)
File("flower_facet_region.xml") >> bfun

fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

d = mesh.topology().dim()
u,v = TrialFunction(V),TestFunction(V)

# harmonic interpolation
dist = Function(V,name="distance")
bcs = [DirichletBC(V,0.0,bfun,3),DirichletBC(V,1.0,bfun,2)]
solve(inner(grad(u),grad(v))*dx==Constant(0.0)*v*dx(mesh),dist,bcs)

dist2 = Function(V,name="distance2")
normalized = lambda g: g/sqrt(dot(g,g))
solve(inner(grad(u),grad(v))*dx==inner(normalized(grad(dist)),grad(v))*dx,
      dist2,DirichletBC(V,0.0,bfun,3))

sf = Constant(10.0,name="sf")
st = Constant(1.0,name="st")
t = normalized(grad(dist2))
fib = as_vector([t[1],-t[0]])

ff = outer(fib,fib)
R0 = Constant(0.5)
ee = Constant(0.05)
sfhump = st + (sf-st)/2.0*(1-tanh((dist2-R0)/ee))
sthump = st/sfhump
D = sfhump*ff + sthump*(Identity(d) - ff)

with XDMFFile(f"rand_flower.xdmf") as ofile:
    ofile.parameters['functions_share_mesh'] = True
    ofile.parameters['rewrite_function_mesh'] = False
    for N in [1,5,10]:
        sampler = RandomFieldSPDE(V,N=N,rho=0.05,D=D)
        s = next(sampler.sample(1))
        ofile.write(s,float(N))

