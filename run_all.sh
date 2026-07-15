#!/usr/bin/env bash
# Reproduce the published results end to end.
# Usage: bash run_all.sh
set -e
cd "$(dirname "$0")/scripts"
echo "=== 1/8 exact-solution verification (Table 1) ==="
python3 verify_analytic.py
echo "=== 2/8 Cartesian vs axisymmetric at d=0 (Table 1) ==="
python3 verify_3d_vs_axi.py
echo "=== 3/8 axisymmetric GCI and domain independence (Table 1) ==="
python3 run_axi_convergence.py
echo "=== 4/8 single-pin offset sweep, primary detector (Table 2) ==="
python3 run3d_offset.py 0 1 2 3 4 5 6 7 8 9
echo "=== 5/8 3-D three-mesh GCI (Table 1) ==="
python3 run3d_gci.py s0.30 s0.225 s0.169 D0.30 D0.225 D0.169
echo "=== 6/8 shape sensitivity, robust detector (Table 3b) + double pin ==="
python3 run_shape_steering.py hemi:6 hemiprof:6 70:6 90:6 110:6
python3 run3d_double.py
echo "=== 7/8 finite-element cross-check (Figure 4) ==="
python3 fem_compare.py d6
python3 fem_compare.py double
echo "=== 8/8 surrogates, trade-off and Bayesian optimisation (Table 4, Figure 6) ==="
python3 run_ml_pareto.py
python3 run_bo.py 0 1 2 3
echo "=== figures ==="
python3 make_figures.py
python3 make_figures2.py
echo "DONE. Results in ../data, figures in ../figures"
