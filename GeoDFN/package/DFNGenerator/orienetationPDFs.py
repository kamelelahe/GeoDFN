import random
import math
import numpy as np
from numba import jit
from scipy.stats import vonmises

class orientationPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if self.distribution_type == "Constant":
            return self.params["theta"]
        elif self.distribution_type == "Uniform":
            return uniform(self.params["min theta"], self.params["max theta"])
        elif  self.distribution_type == "Von-Mises":
            return  np.degrees(vonmisesImp(self.params["loc"], self.params["kappa"]))
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


@jit(nopython=True)
def uniform(min,max):
    return random.uniform(min,max)

def vonmisesImp(loc,kappa):
    return vonmises(loc=loc, kappa=kappa).rvs()

