#!/usr/bin/env python3
"""Fig. 2 - UWB height tracking on const4-trial6-tdoa2-traj3 (GT vs ESKF vs
Adapter-only vs WRAP). Reads committed slim trajectories under
data/precomputed/fig2_height/ and writes figures/height_tracking_C4_traj3_upper.pdf.
No GPU / dataset / C++ needed."""
import sys
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/estimation'))
import matplotlib
matplotlib.use('Agg')
import plot_8way as P

D = REPO / 'data/precomputed/fig2_height'
SEQ = 'const4-trial6-tdoa2-traj3'
summary = {'datasets': [{
    'dataset_name': SEQ, 'const_name': 'const4',
    'methods': {
        'eskf': {'file': str(D / 'eskf.npz')},
        'mamba_eskf': {'file': str(D / 'mamba_eskf.npz')},
        'mamba_dr': {'file': str(D / 'mamba_dr.npz')},
    }}]}
out = REPO / 'figures'
out.mkdir(exist_ok=True)
P.plot_height_tracking(summary, str(out), dataset_name=SEQ, upper_only=True,
                       out_name='height_tracking_C4_traj3_upper.pdf')
print('wrote', out / 'height_tracking_C4_traj3_upper.pdf')
