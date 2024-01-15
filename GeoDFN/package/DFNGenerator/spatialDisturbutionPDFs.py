import random
import math
import numpy as np
from numba import jit
from scipy.stats import vonmises

class spatialDisturbutionPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if  self.distribution_type == "Log-Normal":
            return logNormal(self.params["mu"], self.params["sigma"], self.params["max distance"])
        elif self.distribution_type == "Power-law":
            return negativePowerLaw(self.params["alpha"], self.params["max distance"])
        elif self.distribution_type == "Uniform":
            return uniform( self.params["max distance"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def compute_mode(self):
        if self.distribution_type == "Log-Normal":
            mode = np.exp(self.params["mu"] - self.params["sigma"] ** 2)
            return mode
        elif self.distribution_type == "Power-law":
            # The mode for  power-law is typically the minimum value
            return 100
        elif self.distribution_type == "Uniform":
            # for uniform dist every valuue is mode, so it is considered zero here
            return 0
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


@jit(nopython=True)
def logNormal( mu,sigma, max=None):
    #for plotting, remove the -mu
    val=random.lognormvariate(mu, sigma)
    while val>max:
        val= random.lognormvariate(mu, sigma)
    return val
#@jit(nopython=True)
def negativePowerLaw(alpha, max=None):
    min=10
    if alpha <= 1:
        raise ValueError("Alpha must be greater than 1 for the distribution to be normalizable.")
    if max is None:
        max = np.inf
    cdf_min = 1 - min ** (-alpha + 1)
    cdf_max = 1 - max ** (-alpha + 1)
    u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
    val = min / (1 - u) ** (1 / (alpha - 1))
    max_attempts = 1000
    attempts = 0
    while val > max and attempts < max_attempts:
        u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
        val = min / (1 - u) ** (1 / (alpha - 1))
        attempts += 1
    if attempts >= max_attempts:
        raise ValueError("Unable to generate a value within the specified range after many attempts.")
    return val

@jit(nopython=True)
def uniform(max):
    min = 0
    return random.uniform(min,max)

