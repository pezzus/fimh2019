"""Random field with SPDE on hollow domain.

In addition to expectation and standard deviation,
we also evaluate correlation along the geodesic
connecting points (0.25,0.5) and (0.75,0.5).
"""
from sampler import RandomFieldSPDE
from mshr import Rectangle,generate_mesh
from fenics import *
from time import time
import numpy as np

def piecewise_segment(*pts,N=21):
    t = np.linspace(0.0,1.0,N)
    ns = len(pts)-1
    s = np.empty((ns*(N-1)+1,2))
    for i,(p0,p1) in enumerate(zip(pts,pts[1:])):
        s[i*(N-1):(i+1)*(N-1)+1,:] = np.outer((1-t),p0) + np.outer(t,p1)
    L = np.linalg.norm(s - np.roll(s,-1,axis=0),axis=1)
    L = np.r_[0.0,np.cumsum(L)[:-1]]

    return s,L

N   = 20   # smoothness of RF
rho = 0.2  # correlation length
nsamples = 1000

# hollow square domain
#
domain =   Rectangle(Point(0,0),Point(1,1)) \
         - Rectangle(Point(0.45,0.1),Point(0.55,0.9))
mesh = generate_mesh(domain, 100)

fe = FiniteElement("P",mesh.ufl_cell(),1)
V = FunctionSpace(mesh,fe)

sampler = RandomFieldSPDE(V,N=N,rho=rho)

smean = Function(V,name="mean")
sstd  = Function(V,name="std")
svec = np.empty((nsamples,V.dim()))

seg1,L1 = piecewise_segment([0.25,0.5],[0.45,0.1],[0.55,0.1],[0.75,0.5],N=51)
seg2,L2 = piecewise_segment([0.25,0.25],[0.25,0.75],N=101)
V1 = np.empty((nsamples,seg1.shape[0]))
V2 = np.empty((nsamples,seg2.shape[0]))

with XDMFFile(f"randblock.xdmf") as ofile:
    ofile.parameters['functions_share_mesh'] = True
    ofile.parameters['rewrite_function_mesh'] = False
    t0 = time()
    for i,s in enumerate(sampler.sample(nsamples)):
        print(f"Sample {i}")
        svec[i,:] = s.vector().get_local()
        V1[i,:] = [s(p) for p in seg1]
        V2[i,:] = [s(p) for p in seg2]
        # write only first 4 samples
        if i<4: ofile.write(s,float(i))
        if i%10 == 0:
            print(f"Execution time {time()-t0}")
            t0 = time()

    C1 = [np.cov(V1[:,0],V1[:,i])[0,1] for i in range(V1.shape[1])]
    C2 = [np.cov(V2[:,0],V2[:,i])[0,1] for i in range(V2.shape[1])]
    np.savez(f"randblock_segment.npz",L1=L1,C1=C1,L2=L2,C2=C2)
    smean.vector().set_local(svec.mean(axis=0))
    sstd.vector().set_local(svec.std(axis=0))
    ofile.write(smean,0.0)
    ofile.write(sstd,0.0)
