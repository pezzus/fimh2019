from sampler import RandomFieldSPDE
from fenics import *
import numpy as np

L = 1.0
N = 100
mesh = BoxMesh(Point(0.0,0.0,0.0),Point(L,L,L),N,N,N)

fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

sf = Constant(10.0,name="sf")
st = Constant(0.1,name="st")
sn = Constant(0.1,name="sn")

X = SpatialCoordinate(mesh)
alpha = DOLFIN_PI/3 * (2*X[2]-1)
e1 = Constant((1.0,0.0,0.0))
e2 = Constant((0.0,1.0,0.0))
e3 = Constant((0.0,0.0,1.0))
ff = +cos(alpha)*e1 + sin(alpha)*e2
ss = -sin(alpha)*e1 + cos(alpha)*e2
nn = e3

D = sf*outer(ff,ff) + st*outer(ss,ss) + sn*outer(nn,nn)

sampler = RandomFieldSPDE(V,N=10,rho=0.05*L,D=D)

with XDMFFile("cube_aniso.xdmf") as ofile:
    s = next(sampler.sample(1))
    ofile.write(s,0.0)

