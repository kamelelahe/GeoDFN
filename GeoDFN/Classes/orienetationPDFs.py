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
            return uniform(self.params["thetaMin"], self.params["thetaMax"])
        elif  self.distribution_type == "Von-Mises":
            theta_min = self.params.get("thetaMin", None)
            theta_max = self.params.get("thetaMax", None)
            if theta_max is None:
                theta_max = np.radians(np.inf)
            if theta_min is None:
                theta_min = np.radians(-np.inf)


            return  np.degrees(vonmisesImp(self.params["loc"], self.params["kappa"],theta_min, theta_max))
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


@jit(nopython=True)
def uniform(min,max):
    return random.uniform(min,max)

def vonmisesImp(loc,kappa,min,max):
    value=vonmises(loc=loc, kappa=kappa).rvs()
    while value<min or value>max:
        value = vonmises(loc=loc, kappa=kappa).rvs()
    return value

