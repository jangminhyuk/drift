#!/usr/bin/env python3
"""v47 paper-artifact stage (runs after eval_v47_aggregate). Produces the
complete v47-family paper set as *_v47 SIBLINGS — the current paper files
(v43 family) are not touched; swap-in is a rename once the user adopts v47.

  Fig 2  height_tracking_C4_traj3_upper_v47.pdf   (v47 Mamba vs ESKF vs GT)
  Fig 3  improvement_tdoa2.pdf                    (already written by the
         aggregate job in results/eval_table_v47_final{,_autonomous})
  Fig 4  FIG_REGIMES_UWB_v47.{png,pdf}            (regimes map, v47 gains)
  Table  uwb_rmse_table_tdoa2_paper_split_v47.tex (6-row split format)

All gains use the validation-theta protocol (the aggregate already applied
it to the per-sequence CSV).
"""
import json
import os
import subprocess
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path('.')
ART = ROOT / 'research/dr_conclusions/artifacts'
FINAL = ROOT / 'results/eval_table_v47_final'
FINAL_AUTO = ROOT / 'results/eval_table_v47_final_autonomous'
SHARDS = pathlib.Path(os.path.join(os.environ.get('WRAP_SHARD_ROOT','shards'), 'eval_table_v47_final'))
V1_SHARDS = pathlib.Path(os.path.join(os.environ.get('WRAP_SHARD_ROOT','shards'), 'eval_table_v1'))
os.chdir(ROOT)

# ---------- 1. v47 guideline units (diagnostics unchanged, gains v47) ----
diag = pd.read_csv(ART / 'guideline_units_alltdoa2.csv')
diag = diag[diag.ds == 'UWB']
v = pd.read_csv(FINAL / 'uwb_rmse_per_sequence.csv')
v = v[v.tdoa == 2].rename(columns={'sequence': 'unit'})
u = diag.merge(v[['unit', 'ESKF', 'DR-only', 'Adapter-only', 'WRAP']], on='unit')
u['pct_adapter'] = 100 * (1 - u['Adapter-only'] / u['ESKF'])
u['pct_dr_alone'] = 100 * (1 - u['DR-only'] / u['ESKF'])
u['pct_both'] = 100 * (1 - u['WRAP'] / u['ESKF'])
u[diag.columns].to_csv(ART / 'guideline_units_alltdoa2_v47.csv', index=False)
print(f'[1] wrote guideline_units_alltdoa2_v47.csv ({len(u)} seqs)')

# ---------- 2. Fig 4 sibling: regimes map with v47 gains ----------------
# v47 adopted as canonical (2026-07-12): no suffix, group-fill default style
env = dict(os.environ,
           GUIDELINE_UNITS_CSV='guideline_units_alltdoa2_v47.csv',
           FIG_SUFFIX='')
r = subprocess.run([sys.executable,
                    'research/dr_conclusions/scripts/fig_drift_guideline_v5_uwb.py'],
                   env=env, capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, 'regimes figure failed'
print('[2] FIG_REGIMES_UWB rendered')

# ---------- 3. Fig 2 sibling: height tracking with the v47 Mamba --------
sys.path.insert(0, str(ROOT / 'scripts/estimation'))
import matplotlib
matplotlib.use('Agg')
import plot_8way as P

SEQ = 'const4-trial6-tdoa2-traj3'
eskf_sum = json.load(open(V1_SHARDS / f'{SEQ}__b0/eval_8way_summary.json'))


def _abs(md, shard_dir):
    f = md.get('file', '')
    if f and not os.path.isabs(f) and not os.path.isfile(f):
        cand = os.path.join(shard_dir, f)
        if os.path.isfile(cand):
            md = dict(md, file=cand)
    return md


ds_e = next(d for d in eskf_sum['datasets'] if d['dataset_name'] == SEQ)
fig_dir = FINAL / 'figures'   # all-sequence scope is canonical (2026-07-12)
fig_dir.mkdir(parents=True, exist_ok=True)
# all three backbones (user 2026-07-12); each backbone's methods are aliased
# to the mamba_* keys plot_height_tracking expects — the figure's labels
# ("Adapter-only"/"WRAP") are backbone-agnostic
for bb, sfx in [('mamba', ''), ('gru', '_GRU'), ('transformer', '_XFMR')]:
    shard = SHARDS / f"{SEQ}__{'xfmr' if bb == 'transformer' else bb}"
    bsum = json.load(open(shard / 'eval_8way_summary.json'))
    ds_m = next(d for d in bsum['datasets'] if d['dataset_name'] == SEQ)
    summary = {'datasets': [{
        'dataset_name': SEQ, 'const_name': 'const4',
        'methods': {
            'eskf': _abs(ds_e['methods']['eskf'], str(V1_SHARDS / f'{SEQ}__b0')),
            'mamba_eskf': _abs(ds_m['methods'][f'{bb}_eskf'], str(shard)),
            'mamba_dr': _abs(ds_m['methods'][f'{bb}_dr'], str(shard)),
        }}]}
    P.plot_height_tracking(summary, str(fig_dir), dataset_name=SEQ,
                           upper_only=True,
                           out_name=f'height_tracking_C4_traj3_upper{sfx}.pdf')
print('[3] height tracking (Fig 2, mamba + GRU + XFMR) rendered')

# ---------- 4. Paper table: one row per constellation, manuals pooled ----
# (user 2026-07-12; the split-row version remains on disk as
# uwb_rmse_table_tdoa2_paper_split_v47.tex for reference)
v = pd.read_csv(FINAL / 'uwb_rmse_per_sequence.csv')
v = v[v.tdoa == 2]
METH = ['ESKF', 'DR-only', 'Adapter-only', 'WRAP']


def row(label, g):
    means = {k: g[k].mean() for k in METH}
    stds = {k: np.std(g[k].values) for k in METH}
    best = min(means, key=means.get)
    cells = ['%s%.3f\\,$\\pm$\\,%.3f%s' % ('\\textbf{' if k == best else '',
             means[k], stds[k], '}' if k == best else '') for k in METH]
    return '%s & %s \\\\' % (label, ' & '.join(cells))


L = [r'\begin{table}[t]', r'\centering',
     r'\caption{UWB (TDOA2) localization RMSE (m), mean\,$\pm$\,std over all sequences per anchor constellation. DR radii $(\theta_w,\theta_v)$ are selected once per constellation on the validation trials and held fixed.}',
     r'\label{tab:uwb_rmse}', r'\setlength{\tabcolsep}{4pt}',
     r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lcccc}', r'\toprule',
     'Const. & ESKF & DR-only & Adapter-only (Mamba) & WRAP \\\\', r'\midrule']
for c in ['const1', 'const2', 'const3', 'const4']:
    L.append(row('\\#' + c[-1], v[v.constellation == c]))
L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
out_tex = FINAL / 'uwb_rmse_table_tdoa2_paper.tex'
out_tex.write_text('\n'.join(L) + '\n')
print(f'[4] wrote {out_tex}')
print('\n'.join(L[9:15]))

# ---------- 4b. Sibling tables for the GRU and Transformer adapters ------
# (user 2026-07-12) same pooled format; ESKF/DR-only are backbone-agnostic,
# only the adapter and WRAP columns change per backbone.
def row_bb(label, g, meth):
    means = {k: g[k].mean() for k in meth}
    stds = {k: np.std(g[k].values) for k in meth}
    best = min(means, key=means.get)
    cells = ['%s%.3f\\,$\\pm$\\,%.3f%s' % ('\\textbf{' if k == best else '',
             means[k], stds[k], '}' if k == best else '') for k in meth]
    return '%s & %s \\\\' % (label, ' & '.join(cells))


for bb, adp_col, dr_col, fname in [
        ('GRU', 'GRU-only', 'GRU-WRAP', 'uwb_rmse_table_tdoa2_paper_gru.tex'),
        ('Transformer', 'Transformer-only', 'Transformer-WRAP',
         'uwb_rmse_table_tdoa2_paper_xfmr.tex')]:
    meth = ['ESKF', 'DR-only', adp_col, dr_col]
    Lb = [r'\begin{table}[t]', r'\centering',
          r'\caption{UWB (TDOA2) localization RMSE (m), mean\,$\pm$\,std over all sequences per anchor constellation, with the %s adapter. DR radii $(\theta_w,\theta_v)$ are selected once per constellation on the validation trials and held fixed.}' % bb,
          r'\label{tab:uwb_rmse_%s}' % bb.lower(),
          r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\linewidth}{!}{%',
          r'\begin{tabular}{lcccc}', r'\toprule',
          'Const. & ESKF & DR-only & Adapter-only (%s) & WRAP \\\\' % bb,
          r'\midrule']
    for c in ['const1', 'const2', 'const3', 'const4']:
        Lb.append(row_bb('\\#' + c[-1], v[v.constellation == c], meth))
    Lb += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (FINAL / fname).write_text('\n'.join(Lb) + '\n')
    print(f'[4b] wrote {FINAL / fname}')
    print('\n'.join(Lb[9:14]))

# ---------- 5. summary for the decision report ---------------------------
print('\n=== v47 family, val-theta, all-sequence per-const means ===')
for c in ['const1', 'const2', 'const3', 'const4']:
    g = v[v.constellation == c]
    print('%s: ESKF %.3f | DR %.3f | adp %.3f | WRAP %.3f' %
          (c, g.ESKF.mean(), g['DR-only'].mean(),
           g['Adapter-only'].mean(), g.WRAP.mean()))
cb = v.groupby('constellation')[METH].mean().mean()
for k in METH[1:]:
    print('%14s %+.1f%%' % (k, 100 * (cb.ESKF - cb[k]) / cb.ESKF))
