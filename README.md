<div align="center">

<img src="logoGeoDFN.png" width="180" alt="GeoDFN logo"/>

# GeoDFN

### Stochastic Discrete Fracture Network Generator

Generate geologically realistic fracture networks from field statistics - with a point-and-click interface or a Python API.

<br/>

[![Download for Windows](https://img.shields.io/badge/⬇%20Download%20Desktop%20App%20(Windows)-v2.0.0-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/kamelelahe/GeoDFN/releases/latest/download/GeoDFN-v2.0.0-Windows.zip)

<br/>

[![CI](https://github.com/kamelelahe/GeoDFN/actions/workflows/ci.yml/badge.svg)](https://github.com/kamelelahe/GeoDFN/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

---

## What is GeoDFN?

GeoDFN generates stochastic 2D Discrete Fracture Networks calibrated from geological field observations. Starting from outcrop statistics, it produces geologically plausible fracture network ensembles ready for flow, transport, and geomechanical simulations.

It ships in two forms - a **desktop app** for interactive use, and a **Python library** for scripting and large-scale batch generation.

---

## Get Started in 3 Steps

### Option A - Desktop App (no Python required)

1. Click the **Download** button above
2. Unzip the file
3. Double-click **GeoDFN.exe** → your browser opens with the interface

### Option B - Python API

```bash
git clone https://github.com/kamelelahe/GeoDFN.git
cd GeoDFN
pip install -r requirements.txt
```

```python
import numpy as np
from GeoDFN.Classes.DFNGenerator import DFNGenerator

gen = DFNGenerator(
    domainLengthX=300, domainLengthY=600,
    sets=[{
        'I': 0.01,
        'fractureLengthPDF': 'Log-Normal',
        'fractureLengthPDFParams': {'mu': 2.4, 'sigma': 0.73, 'Lmin': 2.59, 'Lmax': 57.48},
        'spatialDistributionPDF': 'Power-law',
        'spatialDistributionPDFParams': {'alpha': 0.51, 'min distance': 1, 'max distance': 600},
        'orientationDistributionPDF': 'Von-Mises',
        'orientationDistributionPDFParams': {'kappa': 8.55, 'loc': 1.4,
                                              'thetaMin': np.radians(30), 'thetaMax': np.radians(120)},
        'bufferZone': {'method': 'constant', 'constant': 1.4},
    }],
    apertureCalculationParameters={'method': 'subLinear', 'scalingCoefficient': 0.001, 'scalingExponent': 0.5},
    DFNName='my_dfn',
    numOfRealizations=10,
)

print(f"{len(gen.realizations)} realizations generated")
```

---

## GUI or Python API?

| | Desktop GUI | Python API |
|---|---|---|
| **Best for** | Visual exploration, parameter tuning, quick results | Large ensembles, batch generation, simulation pipelines |
| **Setup** | Download & double-click | `pip install -r requirements.txt` |
| **Realizations** | Interactive, one run at a time | Hundreds to thousands, fully automated |

> For synthetic dataset generation and uncertainty quantification studies, we recommend the Python API. It allows full automation over parameter spaces that would be impractical to configure manually through an interface.

---

## Capabilities

| Feature | Options |
|---|---|
| **Fracture length** | Log-Normal · Power-law · Exponential · Constant |
| **Orientation** | Von-Mises · Uniform · Constant |
| **Spatial distribution** | Power-law · Log-Normal · Uniform |
| **Aperture model** | Constant · Sub-linear scaling · Barton-Bandis · Lepillier |
| **Stress correction** | Multi-azimuth stress-dependent aperture |
| **Output** | Coordinates · Apertures · Statistics · Stereonets · Visualizations |

---

## Output

Each run writes results to `DFNs/<name>/`:

```
DFNs/<name>/
├── fractureCoordinates/       # Start/end (x, y) of each fracture
├── aperture/                  # Aperture values
├── fractureSet/               # Full fracture list per set
├── orientationStereographic/  # Stereonet plots
├── outputPropertiesTotal/     # Network statistics
└── pics/                      # DFN visualizations
```

Results are also available in memory after generation:

```python
fractures = gen.realizations[0][0]           # realization 0, set 0
print(fractures[0]['fracture length'])        # length in metres
print(fractures[0]['fracture aperture'])      # aperture in metres
```

---

## For Developers

<details>
<summary>Installation, tests, and building the desktop app</summary>

### Install with dev dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/
```

### Build the desktop app

```bash
pip install pyinstaller
python -m PyInstaller geodfn.spec --noconfirm
```

Distributable is generated in `dist/GeoDFN/`. Share the entire folder - users double-click `GeoDFN.exe`.

### Project structure

```
GeoDFN/
├── Classes/
│   ├── DFNGenerator.py                      # Random-seed generator
│   ├── DFNGeneratorWithSeed.py              # Fixed seed-point generator
│   ├── DFNGeneratorWithSeedAndExclusion.py  # Generator with exclusion zones
│   ├── _validation.py                       # Input validation
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

</details>

---

## Citation

If you use GeoDFN in your research, please cite:

> Kamel Targhi, E., et al. "From outcrop observations to dynamic simulations: an efficient workflow for generating ensembles of geologically plausible fracture networks and assessing their impact on flow and transport." *Geoenergy* 3.1 (2025): geoenergy2025-028.

---

## License

MIT License - © 2025 Elahe Kamel Targhi. See [LICENSE](LICENSE) for details.
