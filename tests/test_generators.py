import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GeoDFN.Classes.DFNGenerator import DFNGenerator


MINIMAL_APERTURE_PARAMS = {
    "method": "subLinear",
    "scalingCoefficient": 0.001,
    "scalingExponent": 0.5,
    "aperture": 1e-3,
    "JCS": 140,
    "JRC": 15,
    "sigma_Hmax": 100,
    "sigma_c": 130,
    "strike": 0,
    "S_Hmax": 1.8e8,
    "S_hmin": 0.7e8,
    "E": 15e9,
    "nu": 0.22,
}

MINIMAL_SET = {
    'I': 0.005,
    'fractureLengthPDF': "Log-Normal",
    'fractureLengthPDFParams': {"mu": 2.0, "sigma": 0.5, "Lmin": 2.0, "Lmax": 30.0},
    'spatialDistributionPDF': "Power-law",
    'spatialDistributionPDFParams': {"alpha": 0.51, "min distance": 1, "max distance": 50},
    'orientationDistributionPDF': "Von-Mises",
    'orientationDistributionPDFParams': {"kappa": 8.0, "loc": 1.4, 'thetaMin': np.radians(30), 'thetaMax': np.radians(120)},
    'bufferZone': {"constant": 1.0, "method": "constant"}
}


def test_dfn_generator_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gen = DFNGenerator(
        domainLengthX=50,
        domainLengthY=50,
        sets=[MINIMAL_SET],
        apertureCalculationParameters=MINIMAL_APERTURE_PARAMS,
        DFNName='test_run',
        numOfRealizations=1,
        savePic=False,
    )
    assert gen is not None


def test_realizations_are_accessible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gen = DFNGenerator(
        domainLengthX=200,
        domainLengthY=200,
        sets=[MINIMAL_SET],
        apertureCalculationParameters=MINIMAL_APERTURE_PARAMS,
        DFNName='test_realizations',
        numOfRealizations=2,
        savePic=False,
    )
    assert hasattr(gen, 'realizations')
    assert len(gen.realizations) == 2
    first_realization = gen.realizations[0]
    assert len(first_realization) == 1
    fractures = first_realization[0]
    assert isinstance(fractures, list)
    assert len(fractures) > 0
    assert 'x_start' in fractures[0]
    assert 'fracture length' in fractures[0]


def test_output_dir_parameter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom_dir = str(tmp_path / 'my_output')
    gen = DFNGenerator(
        domainLengthX=50,
        domainLengthY=50,
        sets=[MINIMAL_SET],
        apertureCalculationParameters=MINIMAL_APERTURE_PARAMS,
        DFNName='test_dir',
        numOfRealizations=1,
        savePic=False,
        output_dir=custom_dir,
    )
    assert os.path.isdir(os.path.join(custom_dir, 'test_dir'))


def test_compute_intensity():
    gen = object.__new__(DFNGenerator)
    gen.xmax = 100.0
    gen.ymax = 100.0
    fractures = [{'fracture length': 10.0}, {'fracture length': 20.0}]
    intensity = gen._compute_intensity(fractures)
    assert abs(intensity - 30.0 / 10000.0) < 1e-10


def test_sort_fractures_descending():
    gen = object.__new__(DFNGenerator)
    fractures = [
        {'fracture length': 5.0},
        {'fracture length': 15.0},
        {'fracture length': 10.0},
    ]
    sorted_fracs = gen._sort_fractures(fractures)
    lengths = [f['fracture length'] for f in sorted_fracs]
    assert lengths == sorted(lengths, reverse=True)
    assert sorted_fracs[0]['number'] == 0
    assert sorted_fracs[1]['number'] == 1
    assert sorted_fracs[2]['number'] == 2
