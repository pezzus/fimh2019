from lrchol import lowRankCholesky
from geodist import *
from scipy.linalg import eigh
import numpy as np

__all__ = ["RandomFieldSampler","SquaredExponentialKernel"]

class SquaredExponentialKernel:

    def __init__(self,gdist,sigma=0.2,alpha=0.0):

        self.gdist = gdist
        self.sigma = sigma
        self.alpha = alpha

        # mass matrix
        self.Ml = gdist.get_mass_matrix(lumped=True)

    def get_mass_matrix(self):
        return self.Ml

    def get_diagonal(self):
        "Returns the diagonal of the kernel matrix"

        # k(x,x) = 1.0, so diagonal is just square of
        # lumped mass matrix
        # NOTE: Au = l Bu  => (A+alpha*B)u = (l+alpha)Bu
        # => (A+alpha*B)u = rBu, l = r-alpha
        d = self.Ml**2 + self.alpha*self.Ml

        return d

    def get_row(self,i):
        "Returns i-th row of the kernel matrix"

        sigma = self.sigma
        xyz = self.gdist.get_point(i)

        Ml = self.Ml
        d  = self.gdist(xyz)
        k  = np.exp(-(d/sigma)**2)
        r  = Ml*k * Ml[i]
        r[i] += self.alpha*Ml[i]

        return r

    def get_full_matrix(self):
        "Returns full kernel matrix. Expensive!"

        n = self.gdist.get_number_of_points()
        C = np.array([ self.get_row(i) for i in range(n) ])
        return C


class RandomFieldSampler:
    "Random field with given spatial correlation."

    def __init__(self,kernel,tol=1e-10):
        "Initialize with given correlation kernel."

        # low-rank Cholesky
        d = kernel.get_diagonal()
        r = kernel.get_row
        Lm,m = lowRankCholesky(d,r,10000,tol)

        # lumped mass matrix (as diagonal matrix)
        B = kernel.get_mass_matrix()
        # inverse mass--we use the fact that B is diagonal
        Binv = 1.0/B
        # B^{-1} * Lm
        BiLm = Binv[:,None] * Lm

        # eigenvalue problem of size m x m
        # NOTE: eigenvectors are columns of Vhat, that is Vhat[:,i]
        # is the i-th eigenvector
        #
        Am = Lm.T @ BiLm
        lmbda,Vhat = eigh(Am)
        # reorder from larger to smallest eigenvalue and shift with alpha
        lmbda = lmbda[::-1] - kernel.alpha
        Vhat = Vhat[:,::-1]

        assert (lmbda > 0).all()

        # back to full rank
        V = BiLm @ Vhat

        # NOTE:
        # the B-norm of eigenvectors in V is exactly lmbda
        #
        #    (B*v_k, v_k) =
        #    = (B*B^{-1}*Lm*vhat_k, B^{-1}*Lm*vhat_k) =
        #    = (vhat_k, Lm^T*B^{-1}*Lm*vhat_k) = lmbda_k
        #
        # to sample, we have the expansion
        #
        #   sqrt(lmbda_k) vnorm_k Z_k = v_k Z_k
        #
        # where vnorm_k is B-normalized and exactly equal to
        #
        #   vnorm_k := 1/sqrt(lmbda_k) v_k
        #

        self.lmbda = lmbda
        self.V = V


    def sample(self, nsamples=1):
        "Sample the field (each row is a sample)"

        V = self.V
        l = self.lmbda
        m = V.shape[1]
        Z = np.random.randn(nsamples,m)

        # size is nsamples x N
        #S = Z @ (np.sqrt(l)[:,None] * V.T)
        S = Z @ V.T

        return S

    def get_number_modes(self):
        "Return size of KL expansion"

        return self.V.shape[1]

    def get_modes(self):
        "Return KL eigenfunctions"

        # columns of V are eigenfunctions
        for mode in self.V.T:
            yield mode

    def set_seed(self,seed):
        "Fix the seed for random number generator"

        np.random.seed(seed)

