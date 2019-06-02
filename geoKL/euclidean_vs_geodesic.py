"""Random field with and without geodesic distance on non-convex domain.
"""
from rndfield import *
from geodist import *

N = 400
rho = 0.2

prefix = "geoKL_hollow"
gdist = create_hollow_square(prefix,N=N,xsize=0.02)
ker = SquaredExponentialKernel(gdist,sigma=rho)
Sgeo = RandomFieldSampler(ker,tol=1e-10).sample(5)

gdist.backend = "euclidean"
Seuc = RandomFieldSampler(ker,tol=1e-10).sample(5)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
fig = plt.figure(figsize=(10,5))
axs = fig.subplots(2,5,sharex=True,sharey=True)
for i in range(5):
    axs[0][i].contourf(gdist.convert_to_function(Sgeo[i,:])[0,:,:],corner_mask=False)
    axs[0][i].set_title(f"Sample {i+1}")
    axs[1][i].contourf(gdist.convert_to_function(Seuc[i,:])[0,:,:],corner_mask=False)
    axs[1][i].set_xticklabels([])
axs[0][0].set_yticklabels([])
axs[1][0].set_yticklabels([])
axs[0][0].set_ylabel("Geodesic")
axs[1][0].set_ylabel("Euclidean")

fig.tight_layout()
fig.subplots_adjust(hspace=0.03,wspace=0.03)
fig.savefig("eucVSgeo.pdf")
plt.show()

