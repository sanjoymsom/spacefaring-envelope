# Reproducing the paper figures and tables

All commands assume the dependencies are installed (`pip install -r
requirements.txt`) and are run from the repository root.

Figures are written both to a local `validation/` directory (for inspection) and
to `paper/` (the directory the LaTeX source includes). Both directories are
git-ignored and are created automatically when the scripts run. Generated grid
data is written to `csv/`.

> Note: the paper's launch-mass benchmark uses the mean Apollo 8–17 Saturn V
> gross lift-off mass computed in `mission_dv.py`, so figures and tables are
> internally consistent across scripts.

## Figure 1 — Geophysical screen (`fig:geophys`)

Mantle Rayleigh number and core magnetic Reynolds number versus planetary mass.

```bash
python geophysics.py --plot-ra
python geophysics.py --plot-dynamo
```

Outputs:
- `paper/geophysics_rayleigh.png`, `validation/geophysics_rayleigh.png` (panel A)
- `paper/geophysics_dynamo.png`, `validation/geophysics_dynamo.png` (panel B)
- `csv/geophysics_rayleigh.csv`, `csv/geophysics_dynamo.csv`

## Figure 2 — Launch-vehicle validation (`fig:val`)

Mass ratio vs mission Δv, absolute launch mass, and first-stage engine count for
six real vehicles (Saturn V, Falcon Heavy, SLS Block 1, Atlas V 551, N1 (1964),
Electron). Also prints the validation tables (including the F-1 turbopump check).

```bash
python validation_launch_vehicles.py
```

Outputs:
- `paper/launch_vehicle_validation.png`, `validation/launch_vehicle_validation.png`
- Console: panels (A) mass-ratio, (B) absolute-mass/engines, and turbopump check.

## Figures 3 & 4 — Launch mass and engine count (`fig:lm`, `fig:eng`)

Run the main survey and choose **option 2** (mass–pressure grid survey + line
plots) at the prompt:

```bash
python spacefaring.py
# choose: 2
```

Outputs:
- `paper/launchmass_lines.png`, `validation/launchmass_lines.png` (Fig. 3)
- `paper/engines_lines.png`, `validation/engines_lines.png` (Fig. 4)
- `csv/feasibility_grid.csv`, `csv/deltav_grid.csv`, `csv/stages_grid.csv`,
  `csv/engines_grid.csv`, `csv/launchmass_grid.csv`, `csv/reliability_grid.csv`

## Table 3 — Reliability-optimal vehicle (`tab:results`)

The per-mass architecture numbers (escape velocity, stages, engines, mission
reliability, launch mass at 1 and 10 bar, drag fraction) are printed by the main
survey. Run **option 2** as above and read the console summary, or **option 1**
to print a detailed per-mass table at a chosen surface pressure:

```bash
python spacefaring.py
# choose: 1   (then enter a surface pressure, e.g. 1.0)
```

## Table 4 — Hippke (2018) benchmark comparison (`tab:benchmark`)

Reproduces Hippke's quoted values against this model (idealized vs full-model
columns) and writes a LaTeX snippet.

```bash
python validation_prior_literature.py
```

Outputs:
- Console: benchmark comparison table.
- `validation/prior_literature_table.tex` (LaTeX snippet for `tab:benchmark`).

## Supplementary — Δv-component breakdown (not in the paper)

```bash
python dv_breakdown_bars.py
```

Output: `validation/deltav_breakdown_stacked.png` (stacked escape / gravity /
drag Δv vs surface pressure for selected planetary masses).

## Module self-tests

Several modules print a quick self-test when run directly, useful for sanity
checks:

```bash
python terrestrial_planet.py      # Valencia mass-radius, g, v_esc for sample masses
python rocket_optimizer.py        # interactive single-planet staging table
python engineering_validation.py  # Earth turbopump / engine-count check
```
