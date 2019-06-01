from fenics import *
import numpy as np
import math as m

class RandomFieldSPDE:

    def __init__(self,V,N=20,rho=0.2,D=None):
        self.V = V
        mesh = V.mesh()

        N = Constant(N,name="N")
        rho = Constant(rho,name="rho")

        d = mesh.topology().dim()
        nu = 2*N - d/2.0
        kappa = sqrt(4*nu) / rho
        eta = Constant(1.0,name="eta")

        f = Function(V,name="rhs")
        u,v = TrialFunction(V),TestFunction(V)

        D = D or Identity(d)
        a = inner(D*grad(u),grad(v))*dx + kappa**2*u*v*dx
        L = eta*f*v*dx

        self.a = a
        self.L = L
        self.N = N
        self.rho = rho
        self.f = f
        self.eta = eta

    def _compute_mass_cholesky(self):

        V = self.V
        mesh = V.mesh()
        dmap = V.dofmap()
        a = TrialFunction(V)*TestFunction(V)*dx
        M = PETScMatrix()
        cpp.fem.Assembler().init_global_tensor(M,Form(a))

        for c in cells(mesh):
            print(c.index())
            Mloc = assemble_local(a,c)
            Lloc = np.linalg.cholesky(Mloc)
            cdof = dmap.cell_dofs(c.index())
            M.mat().setValuesLocal(cdof,cdof,Lloc,addv=True)

        M.apply("add")
        #print(M.array())
        #print(np.linalg.cholesky(assemble(a).array()))

    def get_sigma2(self,N,rho):
        d = self.V.mesh().topology().dim()
        nu = 2*N - d/2.0
        kappa = m.sqrt(4*nu) / rho
        sigma2  = m.pow(4*m.pi,-d/2.0)*m.pow(kappa,-2*nu)
        sigma2 *= m.gamma(nu)/m.gamma(nu+d/2)

        return sigma2

    def sample_white_noise(self,nsamples=1,lumped=False):

        V = self.V
        mesh = V.mesh()
        dmap = V.dofmap()
        a = TrialFunction(V)*TestFunction(V)*dx

        #from time import time
        #t = time()
        if lumped:
            Ml = assemble(action(a,Constant(1.0))).get_local()
            Wvec = 1.0/np.sqrt(Ml) * np.random.randn(nsamples,V.dim())
            w = Function(V)
            for n in range(nsamples):
                w.vector().set_local(Wvec[n,:])
                Wvec[n,:] = assemble(TestFunction(V)*w*dx).get_local()
        else:
            Wvec = np.zeros(shape=(nsamples,V.dim()))
            for c in cells(mesh):
                Mloc = assemble_local(a,c)
                mu = np.zeros(Mloc.shape[0])
                b = np.random.multivariate_normal(mu,Mloc,size=nsamples)
                Wvec[:,dmap.cell_dofs(c.index())] += b
        #print(time()-t)

        for n in range(nsamples):
            W = Function(V,name=f"whitenoise_{n}")
            W.vector().set_local(Wvec[n,:])
            yield W

    def set_seed(self,num):
        np.random.seed(num)

    def sample(self,nsamples=1,lumped=True):

        sparam = {"linear_solver":"cg","preconditioner":"hypre_amg"}
        u = Function(self.V,name="sample")

        A,b = assemble_system(self.a, self.L)
        for W in self.sample_white_noise(nsamples,lumped=lumped):

            Nmax = int(float(self.N))
            sigma2 = self.get_sigma2(float(self.N),float(self.rho))
            eta = 1.0/m.pow(sigma2,1.0/(2*Nmax))
            self.eta.assign(eta)

            b.set_local(eta*W.vector().get_local())
            solve(A,u.vector(),b,"cg","hypre_amg")

            for n in range(1,Nmax):
                self.f.assign(u)
                assemble(self.L,tensor=b)
                #assemble_system(self.a, self.L,A_tensor=A,b_tensor=b)
                solve(A,u.vector(),b,"cg","hypre_amg")
            yield u

