#!/usr/bin/env python3
"""GNSS figure - F_per_seq_improvement (i2Nav per-sequence RMSE improvement,
KF-GINS baseline vs adapter vs DRiFt). Reads the committed summary JSON at
data/precomputed/gnss/figures/summary_both_subsets.json and writes
figures/F_per_seq_improvement.pdf. Only the per-sequence figure is produced
here (the trajectory / error-time panels need raw .npz not shipped in this
repo). No GPU / dataset needed."""
import os
import sys
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ['DRIFT_GNSS_DATA'] = str(REPO / 'data/precomputed/gnss')
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(REPO / 'scripts/gnss_imu/runners'))
import plot_i2nav_drift_pub as M

out = REPO / 'figures'
out.mkdir(exist_ok=True)
M.OUT = out                       # redirect output into figures/
S = M.load_summary()
M.fig_per_seq_improvement(S)
print('wrote', out / 'F_per_seq_improvement.pdf')
