# Electrode-Geometry Control of Normal Electric-Field Distributions and Electrostatic Loading on Sessile-Droplet Interfaces

Source code, data and figure scripts supporting the manuscript:

> F. S. M. Obaid and M. A. Khan, *Electrode-Geometry Control of Normal Electric-Field
> Distributions and Electrostatic Loading on Sessile-Droplet Interfaces: A
> Finite-Difference Study with a Finite-Element Cross-Check and Surrogate-Assisted
> Design Exploration.*

**DOI:** *(to be inserted on publication of this deposit)*
**Version:** v1.0
**Licence:** MIT for code, CC-BY-4.0 for data and figures (see `LICENSE`).

---

## 1. What this deposit contains

Everything needed to reproduce every number, table and figure in the manuscript:
the two finite-difference Laplace solvers, the independent finite-element
cross-check, the Young-Laplace shape solver, the surrogate and Bayesian-optimisation
pipelines, the raw computed data, and the figure-generation scripts.

```
ehd-electrode-geometry/
├── README.md              this file
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_all.sh             reproduces the published numbers end to end
├── src/                   solver library (import path for all scripts)
│   ├── ehd_geom.py             analytic geometry: droplet interfaces (spherical cap
│   │                           and Young-Laplace profile), rounded/flat pins, finite
│   │                           disc, plate; implicit functions and bisection crossings
│   ├── ehd_axisym.py           axisymmetric cylindrical FD Laplace solver: flux-
│   │                           conserving 1/r stencil, Shortley-Weller embedded
│   │                           boundary, asymptotic Robin far field, air-side
│   │                           surface-field extraction at fixed offsets
│   ├── ehd_solver3d.py         3-D Cartesian FD Laplace solver: same embedded
│   │                           boundary, mirror symmetry, AMG-CG, plus the two
│   │                           peak detectors (see Section 4 below)
│   ├── ehd_config.py           electrode configurations and material constants
│   └── ehd_young_laplace.py    augmented Young-Laplace sessile-shape solver
├── scripts/               one script per study (see Section 3)
├── data/                  computed results as JSON (inputs to the tables)
└── figures/
    ├── published/         the figures exactly as they appear in the manuscript
    └── script_output/     plots regenerated from the deposited data by the
                           make_figures scripts (see figures/README.md)
```

## 2. Requirements

Python 3.10 with:

```
numpy>=1.26   scipy>=1.15   scikit-learn>=1.7   xgboost>=3.2
pyamg>=5.3    scikit-fem>=12.0   matplotlib>=3.10
```

Install with `pip install -r requirements.txt`.

The scripts resolve `src/`, `data/` and `figures/` relative to their own location,
so the package runs from any directory with no configuration.

## 3. Reproducing the published results

`bash run_all.sh` runs the full chain. Individual studies:

| Script | Reproduces | Approx. runtime |
|---|---|---|
| `verify_analytic.py` | Exact-solution verification: 0.07% apex error, observed order p ~ 2 (Table 1) | ~30 s |
| `verify_3d_vs_axi.py` | Cartesian versus axisymmetric agreement at d = 0 (~1%, Table 1) | ~10 s |
| `run_axi_convergence.py` | Axisymmetric GCI 0.21% / 0.12% and domain independence (Table 1) | ~1 min |
| `run3d_offset.py 0 1 ... 9` | Table 2 single-pin sweep (E_apex, E_n,max, primary-detector Delta_s, V1) | ~1 min |
| `run3d_gci.py s0.30 s0.225 s0.169 D0.30 D0.225 D0.169` | 3-D three-mesh GCI: p ~ 2.6, GCI ~ 6.2% (Table 1) | ~1 min |
| `run3d_double.py` | Bipolar double-pin: two maxima, Ca_E ~ 0.18 at +/-20 kV (Figure 8) | ~10 s |
| `run_shape_steering.py hemi:6 hemiprof:6 70:6 90:6 110:6` | Table 3b shape sensitivity (robust detector) | ~1 min |
| `fem_compare.py d6` / `fem_compare.py double` | Independent finite-element cross-check (Figure 4) | ~1 min |
| `run_ml_pareto.py` | Table 4 surrogate metrics; steering-voltage trade-off | ~10 s |
| `run_bo.py 0 1 2 3` | Four Bayesian-optimisation trials versus random sampling (Figure 6) | ~2 min |
| `make_figures.py`, `make_figures2.py` | Regenerate the figures | ~1 min |

Runtimes are for the two-core workstation quoted in Table 5 of the manuscript.

## 4. Two peak detectors (important)

The steering displacement `Delta_s` is a sub-grid quantity and is therefore detector
dependent. The manuscript is explicit about this, and both detectors are provided in
`src/ehd_solver3d.py` so that every published value is reproducible:

* `Solver3D.peak_primary` — parabolic refinement of the raw profile, 1.2 mm window,
  no smoothing. **Reproduces the Table 2 `Delta_s` column** (for example
  d = 3 mm gives 0.709 mm, d = 6 mm gives 0.848 mm, quoted as 0.71 and 0.85).
* `Solver3D.peak_robust` — Hann-smoothed, pin-side search including the apex, 0.5 mm
  window, magnitude taken from the raw profile. **Reproduces the Table 3b column**
  (hemisphere 1.055 mm and Young-Laplace 90 degrees 1.213 mm, quoted as 1.1 and 1.2).

The ~0.2 mm spread between the two detectors at the same offset is the detector
sensitivity discussed in Section 3.3 of the manuscript. It is the reason the exact
steering distance is reported as not converged, and why only the existence,
direction and broad plateau of the off-apex maximum are claimed.

## 5. Modelling notes for reviewers

* **Boundary condition.** The bottom electrode is the true finite 20 mm disc, and an
  asymptotic Robin condition on the outer boundary represents decay towards earth at
  infinity. This is well posed and domain independent (<0.1% axisymmetric, <0.5%
  three-dimensional). An infinite biased plane with open side walls is *not* well
  posed for the pin cell: it leaves the droplet-surface field dependent on where the
  outer boundary is placed.
* **Pin tip.** A finite fillet radius `r_tip = 0.10 mm` removes the sharp-edge
  idealisation, but it is not resolved by the production 3-D grid. Pin-tip fields are
  therefore *not* converged and are used only for order-of-magnitude corona screening.
* **Profile geometry control.** `run_shape_steering.py hemiprof:6` expresses the
  hemisphere as a numerical profile and reproduces the analytic spherical-cap result
  (1.055 mm versus 1.055 mm), which validates the Young-Laplace geometry pathway
  against the analytic one.
* **Scope.** The interface is prescribed and undeformed. These calculations supply
  electrostatic loading only; they cannot predict deformation, stability loss, cone
  formation or jetting.

## 6. Data files

`data/ALL.json` consolidates the principal results. Per-study files:

| File | Contents |
|---|---|
| `axi_production.json`, `axi_mesh.json`, `axi_domain.json`, `axi_conv.json` | Axisymmetric fields, GCI and domain studies |
| `off_sweep.json` | Table 2 single-pin sweep |
| `gci3d.json`, `mesh3d.json`, `dom3d_d6.json` | 3-D grid and domain sensitivity |
| `double_pin.json`, `shape_doublepin.json` | Double-pin maxima |
| `young_laplace.json`, `shape_steering.json`, `shape_table3_fixed.json` | Shape studies (Tables 3a, 3b) |
| `femcmp_d6.json`, `femcmp_double.json`, `fem_d6.json` | Finite-element cross-check profiles |
| `ml_pareto.json` | Table 4 surrogate metrics and trade-off sets |
| `bo_results.json`, `bo_summary.json`, `bo_cache.json` | Bayesian-optimisation histories and solver cache |
| `master.json`, `rev2.json` | Consolidated derived quantities (V1, Ca_E, corona) |

Random seeds are fixed throughout (`random_state = 42` for the tree models,
`random_state = 0` for the Gaussian process, Bayesian-optimisation seeds 0 to 3).

## 7. Known limitations of this deposit

* Absolute 3-D field magnitudes carry a reported GCI of ~6.2%; an 8% conservative
  band is used in the exploratory trade-off plot.
* The steering displacement is detector dependent (Section 4) and is not converged.
* The finite-element cross-check uses a coarse (~0.5 mm) tetrahedral mesh and a
  grounded outer box rather than the Robin condition. It tests normalized topology
  only, not absolute magnitude.
* No experimental data are included. None were available for this work.
* **Figure provenance.** `figures/published/` holds the figures of record. The
  authors' final plotting code is not included in this deposit; the
  `make_figures` scripts regenerate equivalent plots from the deposited data but
  differ in styling and, for Figure 2, in the plotted quantity. See
  `figures/README.md`. Adding the authors' figure scripts would make the deposit
  fully self-reproducing.
