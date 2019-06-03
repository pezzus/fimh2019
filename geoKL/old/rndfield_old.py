from fenics import *
from lrchol import lowRankCholesky
import numpy as np

__all__ = ["RandomFieldSampler"]

class RandomFieldSampler:
    "Normal random field with squared-exponential spatial correlation."

    def __init__(self,V,mu=1.0,std=0.0,sigma=1.0):
        "Initialize with mean mu, standard deviation std, correlation sigma."

        # in this version of the code we ignore connectivity, we just assume regular
        # lattice in R^3. Correlation kernel is evaluated pointwise.
        x = V.tabulate_dof_coordinates()
        n = x.shape[0]
        #h = project(CellVolume(V.mesh()),V).vector().get_local()
        dim = x.shape[1]
        h = project(MinCellEdgeLength(V.mesh()),V).vector().get_local().mean()

        #dval = h**3 * 1.0/np.sqrt(2*np.pi*sigma**2)
        ker = lambda w: h**dim * np.exp( -(w/sigma)**2 )
        rowfun = lambda i: ker(np.linalg.norm(x-x[i,:],axis=1))
        dval = ker(0)*np.ones(n)

        Lchol,M = lowRankCholesky(dval,rowfun,10000,1e-4)

        A = np.dot(Lchol,Lchol.T)
        w,vv = np.linalg.eigh(A)
        v = np.dot(Lchol.T,vv)
        # normalize wrt to L2 norm
        for m in range(M):
            mode = v[:,m]
            vfun = Function(V)
            vfun.vector().set_local(mode)
            nrm = sqrt(assemble(vfun**2*dx))
            v[:,m] /= nrm

        self.mu  = mu
        self.std = std
        self.v   = v
        self.w   = w
        self.V   = V
        self.dval = dval[0]

    def sample(self, nsamples=1):
        "Sample the field"

        M = self.w.size
        dval = self.dval
        while True:
            Z = np.random.randn(M,nsamples)
            v = self.v * np.sqrt(self.w)
            Y = np.dot(v,Z)
            print(Y.shape,v.shape)
            # renormalize so that Y.var = std
            #Y /= np.sqrt(dval)
            #Y = Y*np.sqrt(self.w)
            S = self.mu + self.std*Y
            if (S > 0.0).all():
                break
            else:
                print("!!! Skip sample")

        for i,Sval in enumerate(S.T):
            sfun = Function(self.V)
            sfun.vector().set_local(Sval)
            yield sfun

    def get_number_modes(self):
        "Return size of KL expansion"

        return self.v.shape[1]

    def get_modes(self):
        "Return KL eigenfunctions"

        for mode in self.v.T:
            vfun = Function(self.V)
            vfun.vector().set_local(mode)
            yield vfun

    def set_seed(self,seed):
        "Fix the seed for random number generator"
        
        np.random.seed(seed)

if __name__ == "__main__":

    L = 1.0
    N = 200
    mesh = RectangleMesh(Point(0,0),Point(L,L),N,N)
    P1 = FiniteElement("CG",mesh.ufl_cell(),1)
    V = FunctionSpace(mesh,P1)

    sampler = RandomFieldSampler(V,10.0,1.0,0.1)
    sampler.set_seed(34892)

    for s in sampler.sample(10):
        print( s.vector().get_local().std() )

    print(f"Number of modes: {sampler.get_number_modes()}")


