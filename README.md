# Spacefaring envelope of habitable super-Earths

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20805502.svg)](https://doi.org/10.5281/zenodo.20805502)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis code for the paper:

> **Multistage Rocket Optimization, Geophysics, and the Spacefaring Envelope of Habitable Super-Earths**
> Sanjoy M. Som (Blue Marble Space), *Proceedings IAU Symposium No. 404* (2024),
> eds. J. Haqq-Misra & R. Kopparapu.

This repository contains a coupled **geophysical–atmospheric–astronautical**
framework that maps a *spacefaring envelope* onto planetary mass and surface air
pressure. "Spacefaring capability" is defined operationally as placing a
1000 kg ("Voyager-class") payload on an escape trajectory using chemical
propulsion. The model optimizes a multistage chemical rocket whose stage count
minimizes the reliability-weighted expected launch mass, validates against flown
launch vehicles, and finds that **gravity, staging, and engine clustering — not
atmospheric drag or geophysics — set the envelope**, with chemical escape of the
benchmark payload becoming impractical above approximately
11.5 M⊕ (a ~100 F-1-class-engine first-stage clustering limit).

## Repository layout

| Path | Description |
|------|-------------|
| `spacefaring.py` | Main entry point: mass–pressure survey, CSV export, launch-mass and engine-count line plots (Figs. 3 & 4). |
| `rocket_optimizer.py` | Multistage rocket sizing and reliability-weighted stage-count selection (Tsiolkovsky core). |
| `terrestrial_planet.py` | Valencia et al. (2006) mass–radius, surface gravity, escape velocity. |
| `atmospheric_generator.py` | Isothermal atmosphere and altitude-dependent drag-loss integrator. |
| `engineering_validation.py` | Turbopump-power / F-1-class engine-count estimate (Eq. for `N_eng`). |
| `geophysics.py` | Mantle Rayleigh number and core magnetic Reynolds number vs mass (Fig. 1). |
| `mission_dv.py` | Mission Δv budgets (LEO / TLI / escape / Mars) for the validation panel. |
| `validation_launch_vehicles.py` | Validation against real vehicles, Saturn V → Electron (Fig. 2). |
| `validation_prior_literature.py` | Reproduces Hippke (2018) benchmark table (Table 4). |
| `validation_data_sources.py` | Bibliographic URLs/DOIs for the reference-vehicle data. |
| `dv_breakdown_bars.py` | Supplementary Δv-component breakdown plot (not in the paper). |
| `constants.py` | Shared physical constants. |
| `csv/` | Generated grid data (feasibility, Δv, stages, engines, launch mass, reliability). |

The `paper/`, `validation/`, and `heatmap/` directories are **not tracked** in
git (see `.gitignore`); the scripts recreate `validation/` (and write figure
copies into `paper/`) when run.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
# Mass–pressure survey + Figs. 3 & 4 + CSVs (choose option 2 when prompted)
python spacefaring.py

# Geophysics panels (Fig. 1)
python geophysics.py --plot-ra
python geophysics.py --plot-dynamo

# Launch-vehicle validation (Fig. 2) and printed validation tables
python validation_launch_vehicles.py
```

## Reproducing the paper figures and tables

See **[REPRODUCING.md](REPRODUCING.md)** for the exact command for each figure
and table, including output filenames and where they are written.

## Citing

If you use this code, please cite the paper above and the archived software
release. Version 1.0.1 is archived on Zenodo with DOI
[10.5281/zenodo.20805502](https://doi.org/10.5281/zenodo.20805502); citation
metadata is in [CITATION.cff](CITATION.cff).

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

AI-assisted software (the large language model Claude, Opus 4.8, Anthropic,
accessed through the Cursor IDE) was used intermittently during development of
this analysis code and manuscript. The scientific methods and underlying model
were devised independently by the author, who reviewed and verified all code and
output. See the paper's Acknowledgments for the full statement.
