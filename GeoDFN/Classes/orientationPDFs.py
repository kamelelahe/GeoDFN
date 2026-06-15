import math
import numpy as np
from scipy.stats import vonmises


class OrientationPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if self.distribution_type == "Constant":
            return self.params["theta"]
        elif self.distribution_type == "Uniform":
            return np.random.uniform(self.params["thetaMin"], self.params["thetaMax"])
        elif self.distribution_type == "Von-Mises":
            theta_min = self.params.get("thetaMin", None)
            theta_max = self.params.get("thetaMax", None)
            if theta_max is None:
                theta_max = np.radians(np.inf)
            if theta_min is None:
                theta_min = np.radians(-np.inf)
            return np.degrees(_vonmises_sample(self.params["loc"], self.params["kappa"], theta_min, theta_max))
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


def _vonmises_sample(loc, kappa, min_val, max_val):
    value = vonmises(loc=loc, kappa=kappa).rvs()
    while value < min_val or value > max_val:
        value = vonmises(loc=loc, kappa=kappa).rvs()
    return value
