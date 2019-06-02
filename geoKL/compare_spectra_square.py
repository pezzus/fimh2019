from rndfield import *
from geodist import *
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    gdist = create_square("varcube",N=10)
    #gdist.backend = "euclidean"

    ker = SquaredExponentialKernel(gdist,sigma=0.5,alpha=0.0)
    Afull = ker.get_full_matrix()
    Bfull = gdist.get_mass_matrix()

    w,v = eigh(Afull,np.diag(Bfull))
    print(w.min())

    ker.alpha = 1e-2

    sampler = RandomFieldSampler(ker,tol=1e-6)

    plt.plot(w[::-1])
    plt.plot(sampler.lmbda)
    plt.grid()
    plt.show()

