from .Classes.DFNGenerator import DFNGenerator
import numpy as np
from multiprocessing import freeze_support

numOfRealizations = 1
DFNname = 'Brazil-aperture'
domainLengthX = 300
domainLengthY = 600

apertureCalculationParameters = {
    "method": 'Barton-Bandis',
    "JCS": 140,
    "JRC": 15,
    "sigma_Hmax": 100,
    "sigma_c": 130,
    "strike": 0,
}

set1 = {
    'I': 0.01,
    'fractureLengthPDF': "Log-Normal",
    'fractureLengthPDFParams': {"mu": 2.4, "sigma": 0.73, "Lmin": 2.59, "Lmax": 57.48},
    'spatialDistributionPDF': "Power-law",
    'spatialDistributionPDFParams': {"alpha": 0.51, "min distance": 1, "max distance": 600},
    'orientationDistributionPDF': "Von-Mises",
    'orientationDistributionPDFParams': {"kappa": 8.55, "loc": 1.4, 'thetaMin': np.radians(30), 'thetaMax': np.radians(120)},
    'bufferZone': {
        "constant": 1.4,
        "method": "constant"
    }
}

set2 = {
    'I': 0.0435,
    'fractureLengthPDF': "Log-Normal",
    'fractureLengthPDFParams': {"mu": 2.73, "sigma": 0.68, "Lmin": 2.23, "Lmax": 114.92},
    'spatialDistributionPDF': "Power-law",
    'spatialDistributionPDFParams': {"alpha": 0.74, "min distance": 7.5, "max distance": 600},
    'orientationDistributionPDF': "Von-Mises",
    'orientationDistributionPDFParams': {"kappa": 24.5, "loc": 2.75, 'thetaMin': np.radians(120), 'thetaMax': np.radians(175)},
    'bufferZone': {
        "constant": 0.8,
        "method": "constant"
    }
}

set3 = {
    'I': 0.0256,
    'fractureLengthPDF': "Log-Normal",
    'fractureLengthPDFParams': {"mu": 3.06, "sigma": 0.66, "Lmin": 1.2, "Lmax": 121.62},
    'spatialDistributionPDF': "Power-law",
    'spatialDistributionPDFParams': {"alpha": 0.8, "min distance": 7.5, "max distance": 600},
    'orientationDistributionPDF': "Von-Mises",
    'orientationDistributionPDFParams': {"kappa": 58.16, "loc": 0.063, 'thetaMin': np.radians(-5), 'thetaMax': np.radians(30)},
    'bufferZone': {
        "constant": 1.7,
        "method": "constant"
    }
}

IsMultipleStressAzimuths = True
stressAzimuth = [0, 45, 90]


def main():
    DFNGenerator(domainLengthX, domainLengthY, [set1, set2, set3], apertureCalculationParameters, DFNname,
                 numOfRealizations, IsMultipleStressAzimuths=IsMultipleStressAzimuths, stressAzimuth=stressAzimuth)


if __name__ == "__main__":
    freeze_support()
    main()
