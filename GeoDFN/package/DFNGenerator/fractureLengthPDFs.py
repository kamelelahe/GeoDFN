import random
import math
import numpy as np
from numba import jit
from scipy.stats import vonmises

class fractureLengthPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if  self.distribution_type == "Log-Normal":
            return logNormal(self.params["mu"], self.params["sigma"], self.params["Lmax"],self.params["Lmin"])
        elif self.distribution_type == "Truncated power-law":
            return negativePowerLaw(self.params["alpha"], self.params["Lmax"],self.params["Lmin"])
        elif self.distribution_type == "Exponential":
            return exponential(self.params["lambda"], self.params["Lmax"],self.params["Lmin"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")



@jit(nopython=True)
def logNormal( mu,sigma, max=None, min=None):
    #for plotting, remove the -mu
    val=random.lognormvariate(mu, sigma)
    while val>max or val<min:
        val= random.lognormvariate(mu, sigma)
    return val
#@jit(nopython=True)
def negativePowerLaw(alpha, max=None, min=None):
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
def exponential(lambdaf, max=None, min=None):
    if lambdaf <= 0:
        raise ValueError("Lambda must be greater than 0.")
    val = -math.log(1.0 - random.random()) / lambdaf
    while val > max or val<min:
        val = -math.log(1.0 - random.random()) / lambdaf
    return val



