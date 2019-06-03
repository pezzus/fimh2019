from pyccmc.anatomy import create_block
from pyccmc import cPropeiko,igb_write
from textwrap import dedent
from lrchol import lowRankCholesky

import numpy as np

class GeodesicMeter(object):
    def __init__(self,N=100):

        afun = lambda x,y,z: np.pi/3.0*(2*z-1)
        nelm=(N,N,1)

        cfun = lambda x,y,z: 1-np.logical_and(np.abs(x-0.5)<0.05,np.abs(y-0.5)<0.4)

        create_block("varcube",alpha=afun,nelm=nelm,cell=cfun)
        #create_block("varcube",alpha=afun,nelm=nelm,cell=1)
        h = 1.0/N

        parfile = dedent(f"""\
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

        substance[1].name = ventricle
        substance[1].sigma_el = 2.0
        substance[1].sigma_il = 2.0
        substance[1].sigma_et = 2.0
        substance[1].sigma_it = 2.0
        substance[1].theta = 1.0
        substance[1].beta = 1.0
        substance[1].rho2 = -1.0""")

        with open("Eikonal.par","w") as f:
            f.write(parfile)


def save_sample(Yval,dist,fname):
    sfun = np.empty_like(dist.propeiko.get_dtime())
    sfun.fill(np.nan)
    sfun.ravel()[dist.propeiko.nod2ver_view] = Yval
    igb_write(sfun,fname)

N = 100
dist = GeodesicMeter(N)
dim = dist.get_dim()
npts = dist.get_number_of_points()

save_sample(dist.bcs/(2**dim),dist,"foobcs.igb")
#exit(0)

if False:
    mu = np.zeros(npts)
    #pts = dist.get_points()
    #from scipy.spatial.distance import pdist,squareform
    #D = h * squareform(pdist(pts))
    #C = h**dim * np.exp(-(D/sigma)**2)
    #print(C)
    C = np.array([ rowfun(i) for i in range(npts) ])
    (u, s, v) = np.linalg.svd(C)
    print(s)
    Y = np.random.multivariate_normal(mu,C,size=1000)
    for i,Yval in enumerate(Y):
        save_sample(Yval,dist,f"foo_rand_{i}.igb")
        if i >= 4: break
    save_sample(Y.mean(axis=0),dist,f"foo_mean.igb")
    save_sample(Y.std(axis=0),dist,f"foo_std.igb")
    print( Y.shape )
    exit(0)

Lchol,M = lowRankCholesky(diagon,rowfun,5000,1e-8)
Lchol = Lchol.T

print(f"\n\n{M}\n\n")

# lumped mass-matrix
Binv = 1.0/b
Binv = Binv[:,None]
A = Lchol.T @ (Binv * Lchol)
w,vv = np.linalg.eigh(A)
v = Binv * np.dot(Lchol,vv.T)

#
#exit(0)

# normalize with respect to L2 norm
v = v / np.linalg.norm(np.sqrt(diagon)[:,None] * v, axis=0)

#for i,mode in enumerate(v.T):
#    vfun = dist.propeiko.get_dtime().copy()
#    vfun[tuple(dist.xi[:,:])] = mode
#    igb_write(np.swapaxes(vfun,0,2),f"foo_mode_{i}.igb")

np.random.seed(23401)
Z = np.random.randn(M,10000)
# Y = np.empty(1000,npts)
# v is npts x M
# Z is M x N

#Y[i,j] = sqrt(w[k]) * Z[k,i] * v[j,k]
Y = (v @ (np.sqrt(w)[:,None] * Z)).T

for i,Yval in enumerate(Y):
    save_sample(Yval,dist,f"foo_rand_{i}.igb")
    if i >= 4: break

save_sample(Y.mean(axis=0),dist,f"foo_mean.igb")
save_sample(Y.std(axis=0),dist,f"foo_std.igb")


