#!/usr/bin/env python3
"""Table I - UWB (TDOA2) RMSE per anchor constellation (pooled over all
sequences), mean +/- std. Reads the committed per-sequence RMSE table and
writes tables/uwb_rmse_table.tex. No GPU / dataset needed."""
import pathlib
import numpy as np
import pandas as pd
REPO = pathlib.Path(__file__).resolve().parents[1]
v = pd.read_csv(REPO / 'data/precomputed/uwb_rmse_per_sequence.csv')
v = v[v.tdoa == 2]
METH = ['ESKF', 'DR-only', 'Adapter-only', 'DRiFt']


def row(label, g):
    m = {k: g[k].mean() for k in METH}
    s = {k: np.std(g[k].values) for k in METH}
    best = min(m, key=m.get)
    cells = ['%s%.3f\\,$\\pm$\\,%.3f%s' % ('\\textbf{' if k == best else '',
             m[k], s[k], '}' if k == best else '') for k in METH]
    return '%s & %s \\\\' % (label, ' & '.join(cells))


L = [r'\begin{table}[t]', r'\centering',
     r'\caption{UWB (TDOA2) localization RMSE (m), mean\,$\pm$\,std over all '
     r'sequences per anchor constellation. Adapter = Mamba.}',
     r'\label{tab:uwb_rmse}', r'\begin{tabular}{lcccc}', r'\toprule',
     'Const. & ESKF & DR-only & Adapter-only & DRiFt \\\\', r'\midrule']
for c in ['const1', 'const2', 'const3', 'const4']:
    L.append(row('\\#' + c[-1], v[v.constellation == c]))
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
out = REPO / 'tables'
out.mkdir(exist_ok=True)
(out / 'uwb_rmse_table.tex').write_text('\n'.join(L) + '\n')
print('wrote', out / 'uwb_rmse_table.tex')
print('\n'.join(L[6:12]))
