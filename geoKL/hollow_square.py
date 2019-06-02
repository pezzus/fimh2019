from rndfield import *
from geodist import *
from pyccmc.igb2xdmf import write_xdmf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from time import time

def piecewise_segment(*pts,N=21):
    t = np.linspace(0.0,1.0,N)
    ns = len(pts)-1
    s = np.empty((ns*(N-1)+1,2))
    for i,(p0,p1) in enumerate(zip(pts,pts[1:])):
        s[i*(N-1):(i+1)*(N-1)+1,:] = np.outer((1-t),p0) + np.outer(t,p1)
    L = np.linalg.norm(s - np.roll(s,-1,axis=0),axis=1)
    L = np.r_[0.0,np.cumsum(L)[:-1]]

    return s,L


if __name__ == "__main__":

    N = 100
    prefix = "hollow_square"
    gdist = create_hollow_square(prefix,N=N)
    #gdist.backend = "euclidean"

    sns.set()

    alpha = 0.0
    t0 = time()
    ker = SquaredExponentialKernel(gdist,sigma=0.2,alpha=alpha)
    sampler = RandomFieldSampler(ker,tol=1e-10)
    t1 = time()
    print(f"======= KL TIME {t1-t0} =======")
    print()

    t2 = time()
    sampler.set_seed(34892)
    S = sampler.sample(10000)
    print(f"===== SAMPLE TIME {t2-t1} =====")
    print()

    # variogram along geodesic path
    #
    x = np.arange(N+1)/N
    y = np.arange(N+1)/N
    seg,L = piecewise_segment([0.25,0.5],[0.45,0.1],[0.55,0.1],[0.75,0.5],N=101)
    #seg,L = piecewise_segment([0.25,0.5],[0.75,0.5],N=101)
    V = np.empty((S.shape[0],seg.shape[0]))
    for i in range(S.shape[0]):
        Sxy = gdist.convert_to_function(S[i,:],fill_value=0.0)[0,:,:].T
        g = RegularGridInterpolator((x,y),Sxy)
        V[i,:] = g(seg)
    C = [np.cov(V[:,0],V[:,i])[0,1] for i in range(V.shape[1])]

    np.savez(f"{prefix}_cov_geo.npz",C=C,L=L)
    plt.plot(L,C)
    plt.plot(L,np.exp(-(L/ker.sigma)**2))
    plt.xlabel("Distance")
    plt.ylabel("Covariance")
    plt.tight_layout()
    plt.show()

    std_exact = (sampler.V**2).sum(axis=1)

    # variogram
    # line from (0.25,0.25) to (0.25,0.75)
    idx = [gdist.point_to_node(N//4,i,0) for i in range(N//4,3*N//4)]
    C_exact = (sampler.V[idx[0],:] * sampler.V[idx,:]).sum(axis=1)
    C = [np.cov(S[:,idx[0]],S[:,idx[i]])[0,1] for i in range(len(idx))]
    H = 1/N*np.arange(len(C))
    np.savez(f"{prefix}_cov.npz",C=C,H=H,C_exact=C_exact)

    plt.plot(H,C)
    plt.plot(H,np.exp(-(H/ker.sigma)**2))
    plt.plot(H,C_exact)
    plt.xlabel("Distance")
    plt.ylabel("Covariance")
    plt.tight_layout()
    plt.show()

    mean = S.mean(axis=0)
    std  = S.std(axis=0)

    print("Average mean",mean.mean())
    print("Average std ",std.mean())

    olist = []
    for i in range(10):
        gdist.save_function_igb(f"{prefix}_rfun{i}.igb",S[i,:])
        olist += [f"{prefix}_rfun{i}.igb",f"rfun{i}"]
    gdist.save_function_igb(f"{prefix}_mean.igb",mean)
    gdist.save_function_igb(f"{prefix}_std.igb",std)
    gdist.save_function_igb(f"{prefix}_std_exact.igb",std_exact)
    olist += [f"{prefix}_mean.igb","mean",
              f"{prefix}_std.igb","std",
              f"{prefix}_std_exact.igb","std_exact",
              f"{prefix}-c.igb","cell"]
    write_xdmf("hollow_square.xdmf",*olist)

    fig = plt.figure(figsize=(10,5))
    axs = fig.subplots(1,3)
    axs[0].contourf(gdist.convert_to_function(S[0,:])[0,:,:])
    axs[1].contourf(gdist.convert_to_function(std_exact)[0,:,:])
    axs[2].contourf(gdist.convert_to_function(std)[0,:,:])
    plt.show()

