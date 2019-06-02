from fenics import *
from rndfield import *

class SqExpFEniCS:

    def __init__(self):
        mesh = UnitSquareMesh(100,100)
        P1 = FiniteElement("P",mesh.ufl_cell(),1)
        V = FunctionSpace(mesh,P1)

        u,v = TrialFunction(V),TestFunction(V)

        m = u*v*dx
        self.Ml = assemble(action(m,Constant(1.0))).get_local()

        # i-th row
        dee = Expression("sqrt(pow(x[0]-x0,2)+pow(x[1]-x1,2))",x0=0.0,x1=0.0,element=P1)
        kee = Expression("exp(-pow(d/sigma,2))",sigma=0.2,d=dee,element=P1)
        k = Function(V,name="distance")

        self.k = k
        self.dee = dee
        self.kee = kee
        self.d2v = dof_to_vertex_map(V)
        self.mesh = mesh
        self.V = V

    def get_mass_matrix(self):
        return self.Ml

    def get_diagonal(self):
        return self.Ml**2

    def get_row(self,i):
        # eval the kernel from x_i
        xyz = self.mesh.coordinates()[self.d2v[i]]
        self.dee.x0 = xyz[0]
        self.dee.x1 = xyz[1]
        self.k.interpolate(self.kee)
        # assemble the row
        #r = assemble(Constant(self.Ml[i])*self.k*TestFunction(self.V)*dx)
        r = self.Ml*self.Ml[i]*self.k.vector().get_local()

        return r

if __name__ == "__main__":

    ker = SqExpFEniCS()
    sampler = RandomFieldSampler(ker,tol=1e-10)
    sampler.set_seed(10402)

    import numpy as np

    #S = sampler.sample(1).squeeze()
    sfun = Function(ker.V,name="sample")
    #sfun.vector().set_local(S)

    with XDMFFile("sampleFE_modes.xdmf") as f:
        mode = Function(ker.V,name="mode")
        for i,m in enumerate(sampler.get_modes()):
            mode.vector().set_local(m)
            f.write(mode,float(i))

    with XDMFFile("sampleFE_sfun.xdmf") as f:
        m = sampler.get_number_modes()
        Z = np.random.randn(m)
        V = sampler.V
        lmbda = sampler.lmbda
        print(V.shape)
        for i in range(20,m):
            S = sum(Z[k]*V[:,k] for k in range(0,i))
            sfun.vector().set_local( S )
            f.write(sfun,float(i))

    plot(sfun)
    from matplotlib.pyplot import show
    show()


