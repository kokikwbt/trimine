""" Python implementation of TriMine @ KDD'12 """

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma
from tqdm import trange

# https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/decomposition/_online_lda.py#L135


class TriMine(object):
    def __init__(self, k, u, v, n, outputdir):
        # statuses
        self.k = k  # of topics
        self.u = u  # of objects
        self.v = v  # of actors
        self.n = n  # data duration
        self.outputdir = outputdir
        self.train_log = []
        self.max_alpha = 0.001
        self.max_beta  = 10
        self.max_gamma = 10
        self.init_params()


    def init_params(self, alpha=None, beta=None, gamma=None):
        """ Initialize model parameters """

        # if parameter > 1: pure
        # if parameter < 1: mixed
        self.alpha = 0.0001  #self.u
        self.beta  = 10  #self.v
        self.gamma = 10  #self.n

        self.O = np.zeros((self.k, self.u))  # Object matrix
        self.A = np.zeros((self.k, self.v))  # Actor matrix
        self.C = np.zeros((self.k, self.n))  # Time matrix


    def init_status(self):
        self.Nk = np.zeros(self.k, dtype=int)
        self.Nu = np.zeros(self.u, dtype=int)
        self.Nku = np.zeros((self.k, self.u), dtype=int)
        self.Nkv = np.zeros((self.k, self.v), dtype=int)
        self.Nkn = np.zeros((self.k, self.n), dtype=int)
        self.Z = np.full((self.u, self.v, self.n), -1)


    def get_params(self, **kwargs):
        return self.alpha, self.beta, self.gamma


    def get_factors(self):
        return self.O, self.A, self.C


    def infer(self, tensor, n_iter=10, tol=1.e-8,
              init=True, verbose=True):
        """
        Given: a tensor (actors * objects * time)
        Find: matrices, O, A, C
        """
        if init == True:
            self.init_status()

        for iteration in range(n_iter):
            # Sampling hidden topics z, i.e., Equation (1)
            self.Z = self.sample_topic(tensor, self.Z)

            # Update parameters
            self.update_alpha()
            self.update_beta()
            self.update_gamma()
            self.compute_factors()

            # Compute log-likelihood
            llh = self.loglikelihood()
            self.train_log.append(llh)

            # Early break
            if iteration > 0:
                if np.abs(self.train_log[-1] - self.train_log[-2]) < tol:
                    print('Early stopping')
                    break

            if verbose == True:
                # Print learning log
                print('Iteration', iteration + 1)
                print('loglikelihood=\t', llh)
                print(f'| alpha\t| {self.alpha:.3f}')
                print(f'| beta \t| {self.beta:.3f} ')
                print(f'| gamma\t| {self.gamma:.3f}')
                # Save learning log
                plt.plot(self.train_log)
                plt.xlabel('Iterations')
                plt.ylabel('Log-likelihood')
                plt.savefig(self.outputdir + 'train_log.png')
                plt.close()


    def loglikelihood(self):
        """ Compute Log-likelihood """

        # Symmetric dirichlet distribution
        # https://en.wikipedia.org/wiki/Dirichlet_distribution

        llh = 0
        llh = loggamma(self.alpha * self.k) - self.k * loggamma(self.alpha)
        # llh += loggamma(self.alpha * self.u) - self.u * loggamma(self.alpha)
        llh += loggamma(self.beta * self.k) - self.k * loggamma(self.beta)
        llh += loggamma(self.gamma * self.k) - self.k * loggamma(self.gamma)

        for i in range(self.k):
            llh += (self.alpha - 1) * sum([np.log(self.O[j, i]) for j in range(self.u)]) / self.u
            llh += (self.beta  - 1) * sum([np.log(self.A[j, i]) for j in range(self.v)]) / self.v
            llh += (self.gamma - 1) * sum([np.log(self.C[j, i]) for j in range(self.n)]) / self.n

        return llh


    def sample_topic(self, X, Z):
        """
        X: event tensor
        Z: topic assignments of the previous iteration
        """

        self.Nu = X.sum(axis=(1, 2))


        for t in trange(self.n, desc='#### Infering Z'):
            for i in range(self.u):
                for j in range(self.v):
                    # for each non-zero event entry,
                    # assign latent topic, z
                    topic, count = Z[i, j, t], X[i, j, t]

                    if count == 0:
                        continue

                    if not topic == -1:
                        self.Nk[topic] -= count
                        self.Nku[topic, i] -= count
                        self.Nkv[topic, j] -= count
                        self.Nkn[topic, t] -= count

                        if ((self.Nk  < 0).sum() > 0 or
                            (self.Nkv < 0).sum() > 0 or
                            (self.Nku < 0).sum() > 0 or
                            (self.Nkn < 0).sum() > 0):
                            print("Invalid counter N has been found")
                            exit()

                    """ compute posterior distribution """
                    posts = np.zeros(self.k)
                    # print(self.Nku[:, i])
                    # print(self.Nkv[:, j])
                    # print(self.Nkn[:, t])

                    for r in range(self.k):
                        # NOTE: Nk[r] = Nkv[r, :].sum() = Nkn[r, :].sum()
                        O = A = C = 1
                        O = (self.Nku[r, i] + self.alpha) / (self.Nu[i] + self.alpha * self.k)
                        A = (self.Nkv[r, j] + self.beta)  / (self.Nk[r] + self.beta  * self.v)
                        C = (self.Nkn[r, t] + self.gamma) / (self.Nk[r] + self.gamma * self.n)
                        posts[r] = O * A * C

                    posts = posts / posts.sum()  # normalize
                    topic = np.argmax(np.random.multinomial(1, posts))
                    # print(topic, '<-', posts)

                    Z[i, j, t] = topic
                    self.Nk[topic] += count
                    self.Nku[topic, i] += count
                    self.Nkv[topic, j] += count
                    self.Nkn[topic, t] += count

        return Z


    def update_alpha(self):
        # https://www.techscore.com/blog/2015/06/16/dmm/
        num = -1 * self.u * self.k * digamma(self.alpha)

        den = -1 * self.u * self.k * digamma(self.alpha * self.u)
        for i in range(self.k):
            den += self.u * digamma(self.Nk[i] + self.alpha * self.u)
            for j in range(self.u):
                num += digamma(self.Nku[i, j] + self.alpha)

        # den = -1 * self.u * self.k * digamma(self.alpha * self.k)
        # for i in range(self.u):
        #     den += self.k * digamma(self.Nu[i] + self.alpha * self.k)
        #     for j in range(self.k):
        #         num += digamma(self.Nku[j, i] + self.alpha)

        self.alpha *= num / den

        if self.alpha > self.max_alpha:
            self.alpha = self.max_alpha
        if self.alpha < 1.e-8:
            self.alpha = 1.e-8


    def update_beta(self):
        num = -1 * self.k * self.v * digamma(self.beta)
        den = -1 * self.k * self.v * digamma(self.beta * self.v)

        for i in range(self.k):
            den += self.v * digamma(self.Nk[i] + self.beta * self.v)
            for j in range(self.v):
                num += digamma(self.Nkv[i, j] + self.beta)

        self.beta *= num / den

        if self.beta > self.max_beta:
            self.beta = self.max_beta


    def update_gamma(self):
        num = -1 * self.k * self.n * digamma(self.gamma)
        den = -1 * self.k * self.n * digamma(self.gamma * self.n)

        for i in range(self.k):
            den += self.n * digamma(self.Nk[i] + self.gamma * self.n)
            for j in range(self.n):
                num += digamma(self.Nkn[i, j] + self.gamma)

        self.gamma *= num / den

        if self.gamma > self.max_gamma:
            self.gamma = self.max_gamma


    def compute_factors(self):
        """ Generate three factors/matrices, O, A, and C
        """
        self.O = np.zeros((self.u, self.k))
        self.A = np.zeros((self.v, self.k))
        self.C = np.zeros((self.n, self.k))

        for i in range(self.k):
            for j in range(self.u):
                self.O[j, i] = (
                    (self.Nku[i, j] + self.alpha)
                    / (self.Nu[j] + self.alpha * self.k))
            for j in range(self.v):
                self.A[j, i] = (
                    (self.Nkv[i, j] + self.beta)
                    / (self.Nk[i] + self.v * self.beta))
            for j in range(self.n):
                self.C[j, i] = (
                    (self.Nkn[i, j] + self.gamma)
                    / (self.Nk[i] + self.n * self.gamma))

        # print(self.O.sum(axis=1))
        # print(self.A.sum(axis=0))
        # print(self.C.sum(axis=0))
        return self.O, self.A, self.C


    def save_model(self):
        """ Save all of parameters for TriMine
        """
        with open(self.outputdir + 'params.txt', 'w') as f:
            f.write(f'topic,{self.k}\n')
            f.write(f'alpha,{self.alpha}\n')
            f.write(f'beta, {self.beta}\n')
            f.write(f'gamma,{self.gamma}\n')

        np.savetxt(self.outputdir + 'O.txt', self.O)
        np.savetxt(self.outputdir + 'A.txt', self.A)
        np.savetxt(self.outputdir + 'C.txt', self.C)
        np.savetxt(self.outputdir + 'train_log.txt', self.train_log)
