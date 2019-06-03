# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataSPDE = np.load("randblock_vario.npz")
dataKL = np.load("hollow_square_cov_geo.npz")

sns.set()
rho = 0.2

h = dataSPDE['L1']
c = np.exp(-(h/rho)**2)

fig = plt.figure(figsize=(4.3,5))
ax = fig.add_subplot(111)
ax.plot(dataSPDE['L1'],dataSPDE['C1'],label="SPDE")
ax.plot(dataKL['L'],dataKL['C'],label="geoKL")
ax.plot(h,c,'k--',label="exact")
ax.set_xlabel("Distance")
ax.set_ylabel("Covariance")
ax.legend()
fig.tight_layout()
fig.savefig("cov_along_geodesic.pdf")
plt.show()
