# GeoDFN

**GeoDFN** is an open-source Python tool for generating stochastic 2D Discrete Fracture Networks (DFNs) calibrated from geological field observations. It supports statistical characterisation of fracture length, orientation, spatial distribution, and aperture, and produces multi-realization ensembles suitable for flow and transport simulations.

![GeoDFN Logo](logoGeoDFN.png)

[![CI](https://github.com/kamelelahe/GeoDFN/actions/workflows/ci.yml/badge.svg)](https://github.com/kamelelahe/GeoDFN/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [GUI vs Python API — Which Should I Use?](#gui-vs-python-api--which-should-i-use)
- [Using the GUI](#using-the-gui)
- [Using the Python API](#using-the-python-api)
- [Fracture Set Parameters](#fracture-set-parameters)
- [Aperture Methods](#aperture-methods)
- [Output Structure](#output-structure)
- [For Developers](#for-developers)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

---

## Overview

GeoDFN bridges the gap between field observations and numerical simulation. Starting from outcrop statistics, it generates geologically plausible fracture network ensembles that can be directly used in flow and transport modelling workflows.

**Key capabilities:**

- Multi-set fracture networks with independent statistical controls per set
- Fracture length: Log-Normal, Power-law, Exponential, Constant
- Orientation: Von-Mises, Uniform, Constant
- Spatial distribution: Power-law, Log-Normal, Uniform
- Aperture models: constant, sub-linear scaling, Barton-Bandis, Lepillier
- Stress-dependent aperture correction across multiple stress azimuths
- Fixed seed-point and exclusion-zone constraints
- Monte Carlo multi-realization generation
- Programmatic access to results for downstream processing

---

## Getting Started

GeoDFN offers two ways to work:

| | **Desktop GUI** | **Python API** |
|---|---|---|
| Setup | Download folder, double-click | `pip install -r requirements.txt` |
| Best for | Visual exploration, parameter tuning | Batch generation, synthetic datasets, scripting |
| Output | Files + interactive plots | Files + `gen.realizations` in memory |

---

## GUI vs Python API — Which Should I Use?

### Use the Desktop GUI when you:
- Are exploring parameter combinations visually
- Need to generate and inspect a small number of networks
- Want to share the tool with colleagues who are not Python users
- Are in an early calibration phase, adjusting distributions to match field observations

### Use the Python API when you:
- Are generating large ensembles for synthetic studies (hundreds to thousands of realizations)
- Need to integrate DFN generation into a broader simulation or analysis pipeline
- Want to sweep parameters programmatically across a range of values
- Are building datasets for machine learning or uncertainty quantification workflows

> **Note:** For large-scale synthetic studies, we recommend the Python API directly rather than the GUI. Manually adjusting parameters in the GUI for each configuration is impractical at scale, whereas the API allows full automation through standard Python scripting.

---

## Using the GUI

### Desktop App (no Python required)

1. Open the `App/` folder
2. Double-click `GeoDFN.exe`
3. Your browser opens automatically with the interface

The sidebar controls domain size, aperture method, and the number of fracture sets. Each set has its own tab for configuring length, orientation, and spatial distributions. Click **Generate DFN** to run and view the results.

### From source (Python required)

```bash
pip install streamlit
streamlit run app.py
```

---

## Using the Python API

### Installation

```bash
git clone https://github.com/kamelelahe/GeoDFN.git
cd GeoDFN
pip install -r requirements.txt
```

### Basic example

```python
import numpy as np
from GeoDFN.Classes.DFNGenerator import DFNGenerator

set1 = {
    'I': 0.01,                              # fracture intensity (P21, m⁻¹)
    'fractureLengthPDF': 'Log-Normal',
    'fractureLengthPDFParams': {
        'mu': 2.4, 'sigma': 0.73,
        'Lmin': 2.59, 'Lmax': 57.48,
    },
    'spatialDistributionPDF': 'Power-law',
    'spatialDistributionPDFParams': {
        'alpha': 0.51, 'min distance': 1, 'max distance': 600,
    },
    'orientationDistributionPDF': 'Von-Mises',
    'orientationDistributionPDFParams': {
        'kappa': 8.55, 'loc': 1.4,
        'thetaMin': np.radians(30), 'thetaMax': np.radians(120),
    },
    'bufferZone': {'method': 'constant', 'constant': 1.4},
}

aperture_params = {
    'method': 'subLinear',
    'scalingCoefficient': 0.001,
    'scalingExponent': 0.5,
}

gen = DFNGenerator(
    domainLengthX=300,
    domainLengthY=600,
    sets=[set1],
    apertureCalculationParameters=aperture_params,
    DFNName='my_first_dfn',
    numOfRealizations=10,
    output_dir='DFNs',
)

# Access results programmatically
for i, realization in enumerate(gen.realizations):
    fractures = realization[0]   # fractures from set 1
    print(f'Realization {i+1}: {len(fractures)} fractures')
```

### With fixed seed points

```python
from GeoDFN.Classes.DFNGeneratorWithSeed import DFNGeneratorWithSeed

set1_seeded = {**set1, 'seed': {'X': 150, 'Y': 300}}

gen = DFNGeneratorWithSeed(
    domainLengthX=300,
    domainLengthY=600,
    sets=[set1_seeded],
    apertureCalculationParameters=aperture_params,
    DFNName='seeded_dfn',
    num_realizations=10,
)
```

### With exclusion zones

```python
from GeoDFN.Classes.DFNGeneratorWithSeedAndExclusion import DFNGeneratorWithSeedAndExclusion

gen = DFNGeneratorWithSeedAndExclusion(
    domainLengthX=300,
    domainLengthY=600,
    sets=[set1_seeded],
    apertureCalculationParameters=aperture_params,
    DFNName='excluded_dfn',
    num_realizations=10,
)
```

### Controlling the output directory

```python
gen = DFNGenerator(..., output_dir='/path/to/my/results')
```

### Running the provided examples

```bash
python -c "import sys; sys.modules['__package__'] = 'GeoDFN'; exec(open('GeoDFN/Example-BrazilRandomSeeds.py').read())"
```

Or use the helper scripts directly in any IDE or Jupyter environment by importing the classes as shown above.

---

## Fracture Set Parameters

Each fracture set is a Python dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `I` | float | Fracture intensity P21 (total length per unit area, m⁻¹) |
| `fractureLengthPDF` | str | `'Log-Normal'`, `'Power-law'`, `'Exponential'`, `'Constant'` |
| `fractureLengthPDFParams` | dict | Parameters for the chosen distribution (see below) |
| `spatialDistributionPDF` | str | `'Power-law'`, `'Log-Normal'`, `'Uniform'` |
| `spatialDistributionPDFParams` | dict | Parameters for spatial placement |
| `orientationDistributionPDF` | str | `'Von-Mises'`, `'Uniform'`, `'Constant'` |
| `orientationDistributionPDFParams` | dict | Parameters for orientation distribution |
| `bufferZone` | dict | Minimum spacing between fractures |

**Length PDF parameters:**

| PDF | Required keys |
|-----|--------------|
| `Log-Normal` | `mu`, `sigma`, `Lmin`, `Lmax` |
| `Power-law` | `alpha`, `Lmin`, `Lmax` |
| `Exponential` | `lambda`, `Lmin`, `Lmax` |
| `Constant` | `L` |

**Orientation PDF parameters:**

| PDF | Required keys |
|-----|--------------|
| `Von-Mises` | `loc` (mean, radians), `kappa`, `thetaMin`, `thetaMax` |
| `Uniform` | `thetaMin`, `thetaMax` (radians) |
| `Constant` | `theta` (radians) |

**Spatial PDF parameters:**

| PDF | Required keys |
|-----|--------------|
| `Power-law` | `alpha`, `min distance`, `max distance` |
| `Log-Normal` | `mu`, `sigma`, `max distance` |
| `Uniform` | `max distance` |

---

## Aperture Methods

| Method | Required keys | Description |
|--------|--------------|-------------|
| `constant` | `aperture` | Fixed aperture for all fractures |
| `subLinear` | `scalingCoefficient`, `scalingExponent` | Power-law scaling with length: `a = C · Lⁿ` |
| `Barton-Bandis` | `JRC`, `JCS`, `sigma_Hmax`, `sigma_c`, `strike` | Mechanical aperture from joint roughness and normal stress |
| `Lepillier` | `aperture`, `S_Hmax`, `S_hmin`, `E`, `nu`, `strike` | Stress-dependent aperture using elastic rock properties |

---

## Output Structure

Results are written to `<output_dir>/<DFNName>/`:

```
<output_dir>/<DFNName>/
├── fractureCoordinates/      # Start and end (x, y) coordinates per fracture
├── aperture/                 # Aperture values per fracture
├── fractureSet/              # Full fracture list per set
├── inputProperties/          # Copy of input parameters used for each run
├── orientationStereographic/ # Stereonet plots of fracture orientations
├── outputPropertiesPerSet/   # Statistics broken down per fracture set
├── outputPropertiesTotal/    # Total network statistics
├── pics/                     # DFN visualizations
└── tries/                    # Placement iteration logs
```

Results are also accessible in memory via `gen.realizations`:

```python
gen.realizations[i]      # realization i — list of fracture sets
gen.realizations[i][j]   # set j in realization i — list of fracture dicts
gen.realizations[i][j][k]['x_start']          # fracture coordinates
gen.realizations[i][j][k]['fracture length']  # fracture length (m)
gen.realizations[i][j][k]['fracture aperture'] # aperture (m)
```

---

## For Developers

### Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

### Run the test suite

```bash
pytest tests/
```

### Build the desktop app (.exe)

Requires PyInstaller:

```bash
pip install pyinstaller
python -m PyInstaller geodfn.spec --noconfirm
```

The distributable is generated in `dist/GeoDFN/`. Share the entire `dist/GeoDFN/` folder — users double-click `GeoDFN.exe` and the app opens in their browser automatically.

### Project structure

```
GeoDFN/
├── Classes/
│   ├── DFNGenerator.py                  # Random-seed fracture generator
│   ├── DFNGeneratorWithSeed.py          # Fixed seed-point generator
│   ├── DFNGeneratorWithSeedAndExclusion.py  # Generator with exclusion zones
│   ├── _validation.py                   # Input validation
│   ├── fractureLengthPDFs.py
│   ├── orientationPDFs.py
│   ├── spatialDistributionPDFs.py
│   ├── apertureCalculator.py
│   └── bufferZoneCalculator.py
├── Example-BrazilRandomSeeds.py
├── Example-BrazilFixedSeeds.py
├── Example-BrazilFixedSeedsAndExclusion.py
├── Example-BrazilAperture.py
├── Examples.ipynb
└── PercolationAnalysis.ipynb
app.py          # Streamlit GUI
launcher.py     # Desktop app entry point
geodfn.spec     # PyInstaller build spec
```

---

## Dataset

`Datasets/Brazil/Apodi.txt` contains real fracture trace data from the Apodi carbonate formation (Potiguar Basin, NE Brazil), used to calibrate the statistical distributions in the provided examples.

---

## Citation

If you use GeoDFN in your research, please cite:

> Kamel Targhi, E., et al. "From outcrop observations to dynamic simulations: an efficient workflow for generating ensembles of geologically plausible fracture networks and assessing their impact on flow and transport." *Geoenergy* 3.1 (2025): geoenergy2025-028.

---

## License

MIT License — © 2025 Elahe Kamel Targhi. See [LICENSE](LICENSE) for details.
