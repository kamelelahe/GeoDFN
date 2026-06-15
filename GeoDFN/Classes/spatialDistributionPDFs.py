import math
import numpy as np


class SpatialDistributionPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if self.distribution_type == "Log-Normal":
            return _log_normal(self.params["mu"], self.params["sigma"], self.params["max distance"])
        elif self.distribution_type == "Power-law":
            return _negative_power_law(self.params["alpha"], self.params["min distance"], self.params["max distance"])
        elif self.distribution_type == "Uniform":
            return np.random.uniform(0, self.params["max distance"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def compute_mode(self):
        if self.distribution_type == "Log-Normal":
            return np.exp(self.params["mu"] - self.params["sigma"] ** 2)
        elif self.distribution_type == "Power-law":
            return 100
        elif self.distribution_type == "Uniform":
            return 0
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


def _log_normal(mu, sigma, max_val=None):
    val = np.random.lognormal(mu, sigma)
    while val > max_val:
        val = np.random.lognormal(mu, sigma)
    return val


def _negative_power_law(alpha, min_val=None, max_val=None):
    if max_val is None:
        max_val = np.inf
    cdf_min = 1 - min_val ** (-alpha + 1)
    cdf_max = 1 - max_val ** (-alpha + 1)
    u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
    val = min_val / (1 - u) ** (1 / (alpha - 1))
    max_attempts = 1000
    attempts = 0
    while val > max_val and attempts < max_attempts:
        u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
        val = min_val / (1 - u) ** (1 / (alpha - 1))
        attempts += 1
    if attempts >= max_attempts:
        raise ValueError("Unable to generate a value within the specified range after many attempts.")
    return val
