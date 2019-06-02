"""Low-rank approximation of stationary covariance function.

It uses a lazy-evaluation of the correlation kernel, to improve
efficiency when kernel computation is expensive. This is necessary
for kernels depending on the geodesic distance.

Adapted from pyccmc/lowrank
"""

import numpy as np

def lowRankCholesky(dA,rowfun,M=200,tol=1e-4):
    """Compute low-rank pivoted Cholesky factorization.

    :param dA:     initial diagonal
    :param rowfun: function returning row i of matrix A
    :param M:      initial guess of reduced rank (M << N)
    :param tol:    tolerance
    """

    # diagonal
    d = dA.copy()

    # dimension of the matrix
    N = d.size

    # Guess reduced rank
    M = min(N,M)

    # Allocate space for efficiency
    L = np.zeros((M,N))

    # permutation list for pivoting
    p = np.arange(0,N)

    # initial error
    error = np.linalg.norm(d,1)

    # index from 0 and not from 1
    m = 0

    while error > tol:
        i = m + np.argmax( d[p[m:N]] )

        p[[m,i]] = p[[i,m]]

        L[m,p[m]] = np.sqrt(d[p[m]])

        # evaluate row p[m] of the full matrix
        a = rowfun(p[m])
        s = np.dot(L[0:m,p[m]],L[0:m,p[m+1:N]])
        L[m,p[m+1:N]] = (a[p[m+1:N]] - s)/L[m,p[m]]
        d[p[m+1:N]] -= L[m,p[m+1:N]]**2

        error = d[p[m+1:N]].sum()
        print(m,error)

        if m+1 == M: break
        m += 1

    # we transpose so that the matrix is N x M
    L = np.delete(L,range(m,M),axis=0).T

    return L,m


if __name__ == "__main__":

    sigma = 0.1

    N = 1001
    M = N
    x = np.linspace(0,1,N)
    dval = 1.0/(N-1) * 1.0/np.sqrt(2*np.pi*sigma**2)
    rowfun = lambda i: dval*np.exp(-(x-x[i])**2/sigma**2)

    # 2d
    if True:
        N = 11
        x = np.linspace(0,1,N)
        X,Y = np.meshgrid(x,x,indexing='ij')
        x = np.c_[X.flatten(),Y.flatten()]
        dval = (1.0/(N-1))**2 * 1.0/np.sqrt(2*np.pi*sigma**2)
        #dval = 1.0
        #rowfun = lambda i: dval*np.exp(-np.linalg.norm(x-x[i,:],axis=1)**2/sigma**2)
        rowfun = lambda i: dval*np.exp(-np.linalg.norm(x-x[i,:],1,axis=1)/sigma)

    L,M = lowRankCholesky(dval*np.ones(x.shape[0]),rowfun,1e6,1e-6)

    print(M)
    Afull = np.r_[[rowfun(i) for i in range(x.shape[0])]]

    A = np.dot(L,L.T)
    w,vv = np.linalg.eigh(A)
    v = np.dot(L.T,vv)

    #Afull = np.dot(Lfull,Lfull.T)
    from scipy.linalg import eigvalsh
    wfull = eigvalsh(Afull)#,eigvals=(x.shape[0]-w.size,x.shape[0]-1))

    w = w[::-1]
    wfull = wfull[::-1]

    #print("Error on eigenvalues")
    #print(w-wfull)

