from pyccmc import cPropeiko
from pyccmc import igb_write
from pyccmc.anatomy import create_block
from textwrap import dedent
import numpy as np

__all__ = ["GeodesicDistance","create_cube","create_square","create_hollow_square"]

class GeodesicDistance:
    "Compute geodesic distance with Propeiko"

    def __init__(self,parfile,force2d=False):
        "Init Propeiko with parfile"

        self.propeiko = cPropeiko(parfile,quiet=True)
        self.propeiko.mesh_init()

        self.pts = self.get_points()
        self.bcs = self.get_boundary(self.pts)
        self.force2d = force2d
        self.backend = "propeiko"

    def get_boundary(self,pts):
        "Number of incident cubes for each node in the mesh"

        nx = self.propeiko.mesh.nx
        ny = self.propeiko.mesh.ny
        nz = self.propeiko.mesh.nz

        pts = self.get_points()

        neig = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
        bcs = np.zeros(pts.shape[0],dtype=np.int8)
        for nn in neig:
            # cubes in one direction
            cubs = pts - nn
            # exclude boundary cubes
            val = np.logical_and(cubs<[nx,ny,nz], cubs>=[0,0,0]).all(axis=1)
            cubs = cubs[val,:]
            # get voxel idx
            v = cubs[:,0] + nx*(cubs[:,1] + ny*cubs[:,2])
            # check if is cube (therefore inside the domain)
            val[val] = (self.propeiko.vox2cub_view[v] >= 0)
            bcs += val

        return bcs

    def get_mass_matrix(self,lumped=True):

        if lumped:
            dim = self.get_dim()
            return self.get_voxel_volume() * self.bcs / (2**dim)
        else:
            raise NotImplementedError

    def convert_to_function(self,Yval,fill_value=np.nan):

        sfun = np.empty_like(self.propeiko.get_dtime())
        sfun.fill(fill_value)
        sfun.ravel()[self.propeiko.nod2ver_view] = Yval
        return sfun

    def save_function_igb(self,fname,Yval):

        sfun = self.convert_to_function(Yval)
        igb_write(sfun,fname)

    def get_number_of_points(self):
        return self.propeiko.mesh.nnod

    def get_voxel_volume(self):

        hx = self.propeiko.mesh.hx
        hy = self.propeiko.mesh.hy
        hz = self.propeiko.mesh.hz

        if self.force2d:
            hx = hx if self.propeiko.mesh.nx > 1 else 1.0
            hy = hy if self.propeiko.mesh.ny > 1 else 1.0
            hz = hz if self.propeiko.mesh.nz > 1 else 1.0

        return hx*hy*hz

    def get_dim(self):
        # dimension of the problem
        # we always solve in 3D, but if in one direction the
        # number of voxels is 1, then is lower dimentional
        d = sum(getattr(self.propeiko.mesh,s) > 1 for s in ["nx","ny","nz"])
        if self.force2d:
            return d
        else:
            return 3

    def get_point(self,idx):
        return self.get_points(idx)[0]

    def point_to_node(self,x,y,z):
        nx = self.propeiko.mesh.nx + 1
        ny = self.propeiko.mesh.ny + 1
        v = x + nx*(y + ny*z)
        idx = self.propeiko.ver2nod_view[v]
        return idx

    def get_points(self,idx=None):
        nx = self.propeiko.mesh.nx + 1
        ny = self.propeiko.mesh.ny + 1

        idx = idx or slice(None)
        v = self.propeiko.nod2ver_view[idx]
        x = v % nx
        y = (v // nx) % ny
        z = v // (nx*ny)

        return np.c_[x,y,z]

    def __call__(self,xyz):

        if self.backend == "euclidean":
            pts = self.get_points()
            dtime = self.propeiko.mesh.hx * np.linalg.norm(pts - xyz,axis=1)

        elif self.backend == "fmm":
            raise NotImplementedError

        else:
            self.propeiko.set_pacingsite(0,*xyz,0.0)
            self.propeiko.run_dtime()
            nod2ver = self.propeiko.nod2ver_view
            dtime_full = self.propeiko.get_dtime()
            dtime = np.ascontiguousarray(dtime_full.ravel()[nod2ver])

        return dtime

def create_cube():
    pass


def create_hollow_square(prefix,N=50,xsize=0.05,ysize=0.4):
    cfun = lambda x,y,z: 1-np.logical_and(np.abs(x-0.5)<xsize,np.abs(y-0.5)<ysize)
    return create_masked_square(prefix,N,cfun=cfun)

def create_square(prefix,N=50):
    return create_masked_square(prefix,N,cfun=1)

def create_masked_square(prefix,N=50,cfun=1):
    "Hollow square geometry"

    h = 1.0/N
    create_block(prefix,alpha=np.pi/2.0,cell=cfun,nelm=(N,N,1))

    parfile = f"{prefix}.par"

    with open(parfile,"w") as f:
        f.write(dedent(f"""\
            dir_input = .
            dir_output = .
            logfile = -

            hx = {h}
            hy = {h}
            hz = {h}

            fname_alpha = {prefix}-a
            fname_cell  = {prefix}-c
            fname_gamma = {prefix}-g
            fname_phi   = {prefix}-p

            substance[1].name = ventricle
            substance[1].sigma_el = 2.0
            substance[1].sigma_il = 2.0
            substance[1].sigma_et = 2.0
            substance[1].sigma_it = 2.0
            substance[1].theta = 1.0
            substance[1].beta = 1.0
            substance[1].rho2 = -1.0"""))

    return GeodesicDistance(parfile,force2d=True)

if __name__ == "__main__":

    # simple test with hollow square
    #
    gdist = create_hollow_square("varcube")

    print(gdist.get_number_of_points())
    pt = gdist.get_point(53)
    print(pt)
    print(gdist(pt))
    print(gdist.get_voxel_volume())

