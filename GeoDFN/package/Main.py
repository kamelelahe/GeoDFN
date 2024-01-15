from .DFNGenerator.DFNGenerator import DFNGenerator
import numpy as np
from multiprocessing import freeze_support
# Initialize parameters for each distribution

DFNname = '01'

## based on selected method, fill the required parmeters
apertureCalculationParameters={
    "method":'Bisdom', #"constant",'Bisdom','Lepillier','subLinear'

    #constant
    "aperture":10e-4,

    # subLinear
    "scalingCoefficient": 0.0001,  # scaling coefficient
    "scalingExponent": 0.4,  # scaling_exponent for sub-linear

    #Bisdom
    "JCS": 140,
    "JRC":15,
    "sigma_Hmax": 140,
    "sigma_c": 140, #uniaxial compressive strength
    #"strike": 95,

    #Lepillier
    "strike": 95,
    "S_Hmax":1.8e8,
    "S_hmin":0.7e8,
    "E": 15e9,  # Young's modulus in Pa
    "nu": 0.22,  # Poisson ratio
}

set1={
    'I':0.03,
    'fractureLengthPDF': 'Truncated power-law',
    'fractureLengthPDFParams': {"alpha": 2.5, "Lmin": 5},
    'spatialDisturbutionPDF': "Log-Normal",#"Uniform",
    'spatialDisturbutionPDFParams':{"mu": np.log(250), "sigma": 0.5, "max distance":1000},# {"min distance": 2, "max distance":1000},
    'orientationDisturbutonPDF': "Von-Mises",
    'orientationDisturbutonPDFParams': {"kappa":1000 , "loc":np.radians(0)},#direction with North
}

set2={
    'I':0.03,
    'fractureLengthPDF': 'Truncated power-law',
    'fractureLengthPDFParams': {"alpha": 2.5, "Lmin": 5},
    'spatialDisturbutionPDF': "Log-Normal",
    'spatialDisturbutionPDFParams': {"mu": np.log(250), "sigma": 0.5, "max distance":1000},
    'orientationDisturbutonPDF':"Von-Mises",
    'orientationDisturbutonPDFParams': {"kappa":1000 , "loc":np.radians(60)},#direction with North
}

set3={
    'I':0.015,
    'fractureLengthPDF': 'Exponential',
    'fractureLengthPDFParams': {"lambda":0.1, "Lmin": 2},
    'spatialDisturbutionPDF': "Log-Normal",
    'spatialDisturbutionPDFParams': {"mu": np.log(250), "sigma": 0.5, "max distance":1000},
    'orientationDisturbutonPDF':"Uniform",
    'orientationDisturbutonPDFParams': {"max theta":7,
                                        "min theta":11},#direction with North
}

bufferZone={
    "constant": 0.3,
    "method": "constant" #constantlinearRelationshipLength
}

IsMultipleStressAzimuths=False
stressAzimuth = [0,45,90]
domainLengthX=1000
domainLengthY=1000


def main():
    DFNGenerator(domainLengthX, domainLengthY, [set1,set2],apertureCalculationParameters,DFNname,bufferZone,IsMultipleStressAzimuths=IsMultipleStressAzimuths, stressAzimuth=stressAzimuth)


if __name__ == "__main__":
    freeze_support()
    main()