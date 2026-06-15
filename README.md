# GeoDFN

**GeoDFN** is a Python library for generating stochastic 2D Discrete Fracture Networks (DFNs) for geological and hydrogeological simulations. It models fracture length, orientation, spatial distribution, and aperture using real-world statistical data, and supports multi-realization Monte Carlo runs.

![GeoDFN Logo](logoGeoDFN.png)

---

## Features

- Generate multi-set fracture networks with user-defined statistical properties
- Fracture length distributions: Log-Normal, Power-law
- Orientation distributions: Von-Mises, Uniform
- Spatial distributions: Power-law, Uniform
- Aperture models: constant, sub-linear scaling, Barton-Bandis, Lepillier
- Stress-dependent aperture correction for multiple stress azimuths
- Seed-point constraints (fix fracture nucleation locations)
- Exclusion zones (prevent fractures from entering defined regions)
- Buffer zone spacing between fractures
- Multi-realization output for uncertainty quantification
- Outputs: fracture coordinates, apertures, orientations, stereographic plots, statistics

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/GeoDFN.git
cd GeoDFN
pip install -r requirements.txt
```

Or install as a package (editable mode):

```bash
pip install -e .
```

---

## Quick Start

Run one of the provided examples from the project root:

```bash
python -m GeoDFN.Example-BrazilRandomSeeds
python -m GeoDFN.Example-BrazilFixedSeeds
python -m GeoDFN.Example-BrazilFixedSeedsAndExclusion
python -m GeoDFN.Example-BrazilAperture
```

Output is written to `DFNs/<DFNName>/` in the working directory.

---

## Usage

### Basic example (random seed placement)

```python
import numpy as np
from GeoDFN.Classes.DFNGenerator import DFNGenerator

set1 = {
    'I': 0.01,
    'fractureLengthPDF': "Log-Normal",
    'fractureLengthPDFParams': {"mu": 2.4, "sigma": 0.73, "Lmin": 2.59, "Lmax": 57.48},
    'spatialDisturbutionPDF': "Power-law",
    'spatialDisturbutionPDFParams': {"alpha": 0.51, "min distance": 1, "max distance": 600},
    'orientationDisturbutonPDF': "Von-Mises",
    'orientationDisturbutonPDFParams': {
        "kappa": 8.55, "loc": 1.4,
        "thetaMin": np.radians(30), "thetaMax": np.radians(120)
    },
    'bufferZone': {"constant": 1.4, "method": "constant"},
}

aperture_params = {
    "method": "subLinear",
    "scalingCoefficient": 0.001,
    "scalingExponent": 0.5,
}

DFNGenerator(
    domainLengthX=300,
    domainLengthY=600,
    sets=[set1],
    apertureCalculationParameters=aperture_params,
    DFNName="my_first_dfn",
    numOfRealizations=5,
)
```

### With fixed seed points

```python
from GeoDFN.Classes.DFNGeneratorWithSeed import DFNGeneratorWithSeed

seeds = [(150, 300), (100, 200), (200, 400)]  # (x, y) fracture nucleation points

DFNGeneratorWithSeed(
    domainLengthX=300,
    domainLengthY=600,
    sets=[set1],
    apertureCalculationParameters=aperture_params,
    DFNName="seeded_dfn",
    seeds=seeds,
    numOfRealizations=5,
)
```

---

## Fracture Set Parameters

| Key | Description |
|-----|-------------|
| `I` | Fracture intensity (P21, total fracture length per unit area) |
| `fractureLengthPDF` | Length distribution: `"Log-Normal"` or `"Power-law"` |
| `fractureLengthPDFParams` | Parameters for the chosen distribution (mu, sigma, Lmin, Lmax, etc.) |
| `spatialDisturbutionPDF` | Spatial placement distribution: `"Power-law"` or `"Uniform"` |
| `spatialDisturbutionPDFParams` | Parameters for spatial distribution (alpha, min/max distance) |
| `orientationDisturbutonPDF` | Orientation distribution: `"Von-Mises"` or `"Uniform"` |
| `orientationDisturbutonPDFParams` | Parameters (kappa, loc, thetaMin, thetaMax in radians) |
| `bufferZone` | Minimum spacing between fractures (`method`: `"constant"`) |

## Aperture Calculation Methods

| Method | Description |
|--------|-------------|
| `"constant"` | Fixed aperture value for all fractures |
| `"subLinear"` | Power-law scaling with fracture length: `a = C * L^n` |
| `"Barton-Bandis"` | Mechanical aperture model using JRC, JCS, and stress |
| `"Lepillier"` | Stress-dependent aperture using elastic rock properties |

---

## Output Structure

For each run, GeoDFN creates `DFNs/<DFNName>/` containing:

```
DFNs/<DFNName>/
├── fractureCoordinates/     # Start/end coordinates of each fracture
├── aperture/                # Fracture aperture values
├── fractureSet/             # Per-set fracture lists
├── inputProperties/         # Input parameters used
├── orientationStereographic/# Stereonet plots
├── outputPropertiesPerSet/  # Statistics per fracture set
├── outputPropertiesTotal/   # Total network statistics
├── pics/                    # Network visualizations
└── tries/                   # Placement attempt logs
```

---

## Dataset

The `Datasets/Brazil/Apodi.txt` file contains real fracture data from the Apodi carbonate formation (Brazil), used to calibrate the example runs.

---

## Requirements

- Python >= 3.8
- numpy
- scipy
- matplotlib
- numba

---

## Citation

If you use GeoDFN in your research, please cite:

> Kamel Targhi, E., et al. "From outcrop observations to dynamic simulations: an efficient workflow for generating ensembles of geologically plausible fracture networks and assessing their impact on flow and transport." *Geoenergy* 3.1 (2025): geoenergy2025-028.

---

## License

MIT License — Copyright (c) 2025 Elahe Kamel Targhi. See [LICENSE](LICENSE) for details.
