from pyccmc.anatomy import create_block
from pyccmc import cPropeiko,igb_write
from textwrap import dedent
from lrchol import lowRankCholesky

import numpy as np

class GeodesicMeter(object):
    def __init__(self,N=100):

        afun = lambda x,y,z: np.pi/3.0*(2*z-1)
        nelm=(N,N,2)

        #cfun = lambda x,y,z: 1-np.logical_and(np.sin(np.pi*(x-0.05)/0.1)>=0.0,np.abs(y-0.5)<0.3)
        cfun = lambda x,y,z: 1-np.logical_and(np.abs(x-0.5)<0.02,np.abs(y-0.5)<0.4)
        #tfun = lambda x,y,z: 1.0 + 0.9*np.sin(4*np.pi*x)*np.sin(4*np.pi*y)

        create_block("varcube",alpha=afun,nelm=nelm,cell=cfun)
        #create_block("varcube",alpha=afun,nelm=nelm,theta=tfun)

        parfile = dedent("""\
        dir_input = .
        dir_output = .
        logfile = -

        hx = {h}
        hy = {h}
        hz = {h}

        fname_alpha = varcube-a
        fname_cell  = varcube-c
        fname_gamma = varcube-g
        fname_phi   = varcube-p
        #fname_theta = varcube-t

        substance[1].name = ventricle
        substance[1].sigma_el = 2.0
        substance[1].sigma_il = 2.0
        substance[1].sigma_et = 2.0
        substance[1].sigma_it = 2.0
        substance[1].theta = 1.0
        substance[1].beta = 1.0
        substance[1].rho2 = -1.0
        """.format(h=1/N))

        with open("Eikonal.par","w") as f:
            f.write(parfile)

        self.propeiko = cPropeiko("Eikonal.par")
        self.propeiko.set_pacingsite(0,0,0,0,0.0)
        self.propeiko.run_dtime()

        valid = np.isfinite(np.swapaxes(self.propeiko.get_dtime(),0,2))
        self.xi = np.asarray(np.where(valid))
        self.h = 1/N

    def __call__(self,idx):

        # keep this to compare to exact solution
        return self.h * np.linalg.norm(self.xi.T - self.xi.T[idx,:],axis=1)

        # initial point
        self.propeiko.set_pacingsite(0,*self.xi[:,idx],0.0)
        self.propeiko.run_dtime()

        dtime = np.swapaxes(self.propeiko.get_dtime(),0,2)

        return dtime[tuple(self.xi[:,:])]


N = 100
dist = GeodesicMeter(N)

alpha = 0.0
sigma = 0.2
ker = lambda d: dist.h**3 * np.exp(-(d/sigma)**2)
rowfun = lambda i: ker(dist(i)) + alpha*np.eye(dist.xi[0].size,1,i)[0,:]

dA = ker(np.zeros(dist.xi[0].size)) + alpha
Lchol,M = lowRankCholesky(dA,rowfun,5000,1e-4)

print(f"\n\n{M}\n\n")

A = np.dot(Lchol,Lchol.T)
w,vv = np.linalg.eigh(A)
v = np.dot(Lchol.T,vv)

#for i,mode in enumerate(v.T):
#    vfun = dist.propeiko.get_dtime().copy()
#    vfun[tuple(dist.xi[:,:])] = mode
#    igb_write(np.swapaxes(vfun,0,2),f"foo_mode_{i}.igb")

np.random.seed(23401)
Z = np.random.randn(M,4)
v = v * np.sqrt(w)
Y = np.dot(v,Z)

for i,Yval in enumerate(Y.T):
    sfun = np.swapaxes(dist.propeiko.get_dtime(),0,2).copy()
    sfun[tuple(dist.xi[:,:])] = Yval
    igb_write(np.swapaxes(sfun,0,2),f"foo_rand_{i}.igb")

