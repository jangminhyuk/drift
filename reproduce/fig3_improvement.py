#!/usr/bin/env python3
"""Fig. 3 - UWB RMSE improvement bars (Mamba / GRU / Transformer adapter, with
and without the DR layer) vs the ESKF baseline. Reads the committed per-sequence
RMSE table and writes figures/improvement_tdoa2.pdf. No GPU / dataset needed."""
import sys
import pathlib
import pandas as pd
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/estimation'))
import matplotlib
matplotlib.use('Agg')
import plot_8way as P

v = pd.read_csv(REPO / 'data/precomputed/uwb_rmse_per_sequence.csv')
v = v[v.tdoa == 2]
COL = {'eskf': 'ESKF', 'dr_eskf': 'DR-only', 'mamba_eskf': 'Adapter-only',
       'mamba_dr': 'DRiFt', 'transformer_eskf': 'Transformer-only',
       'transformer_dr': 'Transformer-DRiFt', 'gru_eskf': 'GRU-only',
       'gru_dr': 'GRU-DRiFt'}
datasets = []
for _, r in v.iterrows():
    datasets.append({
        'dataset_name': r['sequence'], 'const_name': r['constellation'],
        'methods': {mk: {'rms_all': float(r[c])}
                    for mk, c in COL.items() if c in r and pd.notna(r[c])}})
out = REPO / 'figures'
out.mkdir(exist_ok=True)
P.plot_headline_improvement({'datasets': datasets}, str(out),
                            out_name='improvement_tdoa2.pdf')
print('wrote', out / 'improvement_tdoa2.pdf')
