import math
import numpy as np


class fractureLengthPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if self.distribution_type == "Constant":
            return self.params["L"]
        elif self.distribution_type == "Log-Normal":
            return _log_normal(self.params["mu"], self.params["sigma"], self.params["Lmax"], self.params["Lmin"])
        elif self.distribution_type == "Power-law":
            return _negative_power_law(self.params["alpha"], self.params["Lmax"], self.params["Lmin"])
        elif self.distribution_type == "Exponential":
            return _exponential(self.params["lambda"], self.params["Lmax"], self.params["Lmin"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")


def _log_normal(mu, sigma, max_val=None, min_val=None):
    val = np.random.lognormal(mu, sigma)
    while val > max_val or val < min_val:
        val = np.random.lognormal(mu, sigma)
    return val


def _negative_power_law(alpha, max_val=None, min_val=None):
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


def _exponential(lambdaf, max_val=None, min_val=None):
    if lambdaf <= 0:
        raise ValueError("Lambda must be greater than 0.")
    val = -math.log(1.0 - np.random.random()) / lambdaf
    while val > max_val or val < min_val:
        val = -math.log(1.0 - np.random.random()) / lambdaf
    return val
