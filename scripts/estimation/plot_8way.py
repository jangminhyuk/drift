'''
Publication-quality plots for the 12-method comparison
(4 base x {none, TAC, CDC} DR variants).

Reads eval_8way_summary.json and generates the same set of figures as
the original 8-method version, with TAC vs CDC clearly distinguished
in legends, colors, hatch patterns, and per-method heatmaps.

Each DR method has its own theta-sweep heatmap:
  TAC: theta_w x theta_v
  CDC: theta_x x theta_v

Usage:
    python plot_8way.py --results_dir results/eval_8way_v4 --save_only
'''
import argparse
import ast
import json
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Publication-quality defaults (matching plot_adaptive.py)
# ------------------------------------------------------------------ #
# Paper style (2026-07-09): DejaVu Sans + the FIG_REGIMES_UWB palette so all
# paper figures share one look (see research/dr_conclusions/scripts/
# fig_drift_guideline_v5_uwb.py). Palette: adapter #e34948 / DR #2a78d6 /
# DRiFt #4a3aa7 / baseline gray #898781 / ink #0b0b0b / chrome #c3c2b7.
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.edgecolor': '#c3c2b7',
    'xtick.color': '#52514e',
    'ytick.color': '#52514e',
    'grid.color': '#e1e0d9',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
})

SINGLE_COL_W = 3.5
DOUBLE_COL_W = 7.16

# ------------------------------------------------------------------ #
#  12-method color/style scheme
#  Color encodes predictor type; hatch '' = no DR, '///' = TAC, 'xxx' = CDC.
#  group: 'eskf' (no DR), 'tac' (TAC-DR), or 'cdc' (CDC-DR).
# ------------------------------------------------------------------ #
METHOD_META = {
    'eskf':                {'color': '#4878CF', 'hatch': '',    'label': 'ESKF',           'group': 'eskf', 'dr_style': None},
    'dr_eskf':             {'color': '#4878CF', 'hatch': '///', 'label': 'DR-ESKF (TAC)',  'group': 'tac',  'dr_style': 'tac'},
    'dr_eskf_cdc':         {'color': '#4878CF', 'hatch': 'xxx', 'label': 'DR-ESKF (CDC)',  'group': 'cdc',  'dr_style': 'cdc'},
    'mamba_eskf':          {'color': '#2ca02c', 'hatch': '',    'label': 'Mamba',          'group': 'eskf', 'dr_style': None},
    'mamba_dr':            {'color': '#2ca02c', 'hatch': '///', 'label': 'Mamba+DR (TAC)', 'group': 'tac',  'dr_style': 'tac'},
    'mamba_cdc':           {'color': '#2ca02c', 'hatch': 'xxx', 'label': 'Mamba+DR (CDC)', 'group': 'cdc',  'dr_style': 'cdc'},
    'gru_eskf':            {'color': '#d62728', 'hatch': '',    'label': 'GRU',            'group': 'eskf', 'dr_style': None},
    'gru_dr':              {'color': '#d62728', 'hatch': '///', 'label': 'GRU+DR (TAC)',   'group': 'tac',  'dr_style': 'tac'},
    'gru_cdc':             {'color': '#d62728', 'hatch': 'xxx', 'label': 'GRU+DR (CDC)',   'group': 'cdc',  'dr_style': 'cdc'},
    'transformer_eskf':    {'color': '#ff7f0e', 'hatch': '',    'label': 'Transformer',           'group': 'eskf', 'dr_style': None},
    'transformer_dr':      {'color': '#ff7f0e', 'hatch': '///', 'label': 'Transformer+DR (TAC)',  'group': 'tac',  'dr_style': 'tac'},
    'transformer_cdc':     {'color': '#ff7f0e', 'hatch': 'xxx', 'label': 'Transformer+DR (CDC)',  'group': 'cdc',  'dr_style': 'cdc'},
    # Classical (statistical) adaptive baselines (TAC variants only)
    'sh_eskf':             {'color': '#9467bd', 'hatch': '',    'label': 'Sage-Husa',             'group': 'eskf', 'dr_style': None},
    'sh_dr':               {'color': '#9467bd', 'hatch': '///', 'label': 'Sage-Husa+DR (TAC)',    'group': 'tac',  'dr_style': 'tac'},
    'vb_eskf':             {'color': '#8c564b', 'hatch': '',    'label': 'VB-AKF',                'group': 'eskf', 'dr_style': None},
    'vb_dr':               {'color': '#8c564b', 'hatch': '///', 'label': 'VB-AKF+DR (TAC)',       'group': 'tac',  'dr_style': 'tac'},
}

PREDICTOR_COLORS = {
    'mamba': '#2ca02c',
    'gru': '#d62728',
    'transformer': '#ff7f0e',
    'sh': '#9467bd',
    'vb': '#8c564b',
}

PREDICTOR_LABELS = {
    'mamba': 'Mamba',
    'gru': 'GRU',
    'transformer': 'Transformer',
    'sh': 'Sage-Husa',
    'vb': 'VB-AKF',
}


def _pred_label(p):
    return PREDICTOR_LABELS.get(p, p.capitalize())


def _theta_axes(summary, dr_style):
    '''Return (t1_list, t2_list, t1_short, t2_short, t1_label, t2_label)
    for plotting the heatmap of a method with the given dr_style.'''
    if dr_style == 'cdc':
        t1 = summary.get('theta_x_list') or summary.get('theta_w_list', [])
        t2 = summary.get('theta_v_list', [])
        return t1, t2, 'tx', 'tv', r'$\theta_x$', r'$\theta_v$'
    # default TAC
    t1 = summary.get('theta_w_list', [])
    t2 = summary.get('theta_v_list', [])
    return t1, t2, 'tw', 'tv', r'$\theta_w$', r'$\theta_v$'


def _grid_key(dr_style, t1, t2):
    if dr_style == 'cdc':
        return 'tx%.3g_tv%.3g' % (t1, t2)
    return 'tw%.3g_tv%.3g' % (t1, t2)

# ------------------------------------------------------------------ #
#  Data loading helpers
# ------------------------------------------------------------------ #
def load_summary(results_dir):
    '''Load eval_8way_summary.json.'''
    path = os.path.join(results_dir, 'eval_8way_summary.json')
    with open(path) as f:
        summary = json.load(f)
    # Convert 'inf' strings back to float
    def _destring(obj):
        if isinstance(obj, str) and obj in ('inf', 'nan'):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _destring(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_destring(v) for v in obj]
        return obj
    return _destring(summary)


def _get_const_rms(summary, method_key):
    '''Get per-constellation mean RMSE for a method.
    Returns dict: const_name -> mean_rms_all.'''
    const_vals = {}
    for ds in summary['datasets']:
        cn = ds['const_name']
        mdata = ds['methods'].get(method_key, {})
        rms = mdata.get('rms_all', float('inf'))
        if isinstance(rms, str):
            rms = float('inf')
        const_vals.setdefault(cn, []).append(rms)
    return {cn: np.mean(vs) for cn, vs in const_vals.items()}


def _parse_csv_list(s):
    '''Parse a list-encoded CSV column like "[1.0, 2.0, 3.0]".'''
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    s = str(s).strip()
    if s.startswith('['):
        try:
            return np.array(ast.literal_eval(s), dtype=float)
        except Exception:
            pass
    return np.array([float(s)])


# ------------------------------------------------------------------ #
#  Plot 1: Main RMSE bar chart (2-row: ESKF-based / DR-based)
# ------------------------------------------------------------------ #
def plot_rmse_bar(summary, fig_dir):
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))
    n_const = len(const_names)

    eskf_methods = [m for m in methods_run if METHOD_META.get(m, {}).get('group') == 'eskf']
    tac_methods = [m for m in methods_run if METHOD_META.get(m, {}).get('group') == 'tac']
    cdc_methods = [m for m in methods_run if METHOD_META.get(m, {}).get('group') == 'cdc']

    n_panels = sum(1 for s in (eskf_methods, tac_methods, cdc_methods) if s)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(n_panels, 1, figsize=(DOUBLE_COL_W, 1.8 * n_panels + 0.6),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    panel_specs = [(eskf_methods, 'No DR (vanilla ESKF stage)'),
                   (tac_methods,  r'TAC DR-ESKF — BW ball on $(\Sigma_w, \Sigma_v)$'),
                   (cdc_methods,  r'CDC DR-ESKF — BW ball on $(X_{prior}, \Sigma_v)$')]
    panel_iter = iter(axes)
    ax1 = ax2 = None
    for method_set, title in panel_specs:
        if not method_set:
            continue
        ax = next(panel_iter)
        if ax1 is None:
            ax1 = ax
        ax2 = ax
        n_methods = len(method_set)
        if n_methods == 0:
            ax.set_visible(False)
            continue

        bar_w = 0.7 / max(n_methods, 1)
        x = np.arange(n_const)

        for i, mkey in enumerate(method_set):
            meta = METHOD_META.get(mkey, {})
            const_rms = _get_const_rms(summary, mkey)
            vals = [const_rms.get(cn, float('inf')) for cn in const_names]
            offset = (i - (n_methods - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w * 0.9,
                          color=meta.get('color', '#999'),
                          hatch=meta.get('hatch', ''),
                          edgecolor='white', linewidth=0.5,
                          label=meta.get('label', mkey))
            # Value annotations
            for bar, v in zip(bars, vals):
                if v < 10:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            '%.3f' % v, ha='center', va='bottom', fontsize=6, rotation=0)

        ax.set_ylabel('RMSE [m]')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_xticks(x)
        ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])

    # Shared y-axis limits
    all_rms = []
    for mkey in methods_run:
        for cn in const_names:
            cr = _get_const_rms(summary, mkey)
            v = cr.get(cn, float('inf'))
            if v < 10:
                all_rms.append(v)
    if all_rms:
        ymax = max(all_rms) * 1.15
        for ax in axes:
            ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rmse_bar_8way.pdf'))
    plt.close(fig)
    print('  Saved rmse_bar_8way.pdf')


# ------------------------------------------------------------------ #
#  Plot 2: Improvement over ESKF baseline
# ------------------------------------------------------------------ #
def plot_improvement(summary, fig_dir, include_cdc=True, strip_tac=False,
                     out_name='improvement_8way.pdf', font_scale=1.0, show_title=True,
                     xlabel='Improvement over ESKF baseline [%]'):
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    # Compute overall mean RMSE per method
    eskf_rms = _get_const_rms(summary, 'eskf')
    eskf_mean = np.mean(list(eskf_rms.values()))
    if eskf_mean == 0 or math.isinf(eskf_mean):
        print('  Skipping improvement plot (no valid ESKF baseline)')
        return

    improvements = []
    for mkey in methods_run:
        if mkey == 'eskf':
            continue
        if not include_cdc and METHOD_META.get(mkey, {}).get('group') == 'cdc':
            continue
        cr = _get_const_rms(summary, mkey)
        method_mean = np.mean(list(cr.values()))
        pct = (eskf_mean - method_mean) / eskf_mean * 100
        improvements.append((mkey, pct))

    # Sort by improvement descending
    improvements.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 0.45 * len(improvements) + 1.0))
    y_pos = np.arange(len(improvements))
    labels = []
    colors = []
    vals = []
    for mkey, pct in improvements:
        meta = METHOD_META.get(mkey, {})
        label = meta.get('label', mkey)
        if strip_tac:
            label = label.replace(' (TAC)', '')
        labels.append(label)
        colors.append(meta.get('color', '#999'))
        vals.append(pct)

    bars = ax.barh(y_pos, vals, color=colors, edgecolor='white', height=0.6)
    for bar, mkey_pct in zip(bars, improvements):
        mkey, pct = mkey_pct
        meta = METHOD_META.get(mkey, {})
        if meta.get('hatch'):
            bar.set_hatch(meta['hatch'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9 * font_scale)
    ax.set_xlabel(xlabel, fontsize=11 * font_scale)
    ax.tick_params(axis='x', labelsize=9 * font_scale)
    ax.axvline(0, color='#333', linewidth=0.8, zorder=0)
    ax.invert_yaxis()

    # Value annotations
    for i, v in enumerate(vals):
        ax.text(v + 0.3 if v >= 0 else v - 0.3,
                i, '%+.1f%%' % v, va='center',
                ha='left' if v >= 0 else 'right', fontsize=8 * font_scale)

    if show_title:
        ax.set_title('RMSE Improvement Over ESKF Baseline (mean across constellations)',
                     fontsize=11 * font_scale)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print('  Saved %s' % out_name)


# ------------------------------------------------------------------ #
#  Plot 2b: Headline improvement figure (curated subset, publication)
#  Flat bars sorted within three groups — DRiFt (adapter + DR),
#  Adapter-only (adapter, no DR), and the classical DR-only baseline —
#  with short backbone tick labels and vertical group brackets on the left.
# ------------------------------------------------------------------ #
def plot_headline_improvement(summary, fig_dir, out_name='improvement.pdf'):
    eskf_rms = _get_const_rms(summary, 'eskf')
    eskf_mean = np.mean(list(eskf_rms.values())) if eskf_rms else float('inf')
    if eskf_mean == 0 or math.isinf(eskf_mean):
        print('  Skipping headline improvement (no valid ESKF baseline)')
        return

    def _imp(mkey):
        cr = _get_const_rms(summary, mkey)
        if not cr:
            return None
        return (eskf_mean - np.mean(list(cr.values()))) / eskf_mean * 100.0

    backbones = ['mamba', 'transformer', 'gru']
    # variant carries the color (backbone is named by the y-tick label):
    # DRiFt violet / adapter red / DR blue — same semantics as FIG_REGIMES_UWB
    drift_color, adapter_color, dr_only_color = '#4a3aa7', '#e34948', '#2a78d6'

    # Three contiguous blocks, each sorted by improvement (desc) so the group
    # brackets always span a contiguous, ranked set of bars.
    drift = [(p, _imp(p + '_dr')) for p in backbones]
    drift = sorted([(p, v) for p, v in drift if v is not None],
                   key=lambda t: t[1], reverse=True)
    adapt = [(p, _imp(p + '_eskf')) for p in backbones]
    adapt = sorted([(p, v) for p, v in adapt if v is not None],
                   key=lambda t: t[1], reverse=True)
    dr_only = _imp('dr_eskf')

    # rows: (value, color, tick_label, group_label)
    rows = []
    for p, v in drift:
        rows.append((v, drift_color, PREDICTOR_LABELS.get(p, p), 'DRiFt'))
    for p, v in adapt:
        rows.append((v, adapter_color, PREDICTOR_LABELS.get(p, p), 'Adapter-only'))
    if dr_only is not None:
        rows.append((dr_only, dr_only_color, 'DR-only', None))
    if not rows:
        print('  Skipping headline improvement (no methods)')
        return

    vals = [r[0] for r in rows]
    n = len(rows)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, max(2.6, 0.46 * n + 0.5)))
    # light fill + solid colored edge, like FIG_REGIMES_UWB panel (b)
    ax.barh(y, vals, color=[to_rgba(r[1], 0.35) for r in rows],
            edgecolor=[r[1] for r in rows], height=0.66, linewidth=1.6)

    hi = max(vals + [0.0])
    for i, v in enumerate(vals):
        ax.text(v + 0.012 * hi, i, '%+.1f%%' % v, va='center', ha='left',
                fontsize=9, color='#0b0b0b', fontweight='bold')

    ax.set_xlim(0, hi * 1.16)
    ax.set_ylim(n - 0.5, -0.5)          # row 0 (best) at the top
    ax.set_yticks(y)
    ax.set_yticklabels([r[2] for r in rows], fontsize=11, color='#0b0b0b')
    ax.set_xlabel('RMSE improvement over ESKF baseline [%]', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)

    # Reserve left room, then measure the rendered y-tick-label extent so the
    # group brackets/labels sit clear of the longest tick (e.g. "Transformer")
    # instead of relying on a fixed offset that can overlap it.
    fig.subplots_adjust(left=0.32, right=0.97, bottom=0.16, top=0.97)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    tick_left = 0.0
    for lbl in ax.get_yticklabels():
        bb = lbl.get_window_extent(renderer=renderer)
        tick_left = min(tick_left, inv.transform((bb.x0, bb.y0))[0])
    bracket_x = tick_left - 0.03      # vertical bracket, just left of the ticks
    label_x = bracket_x - 0.05        # rotated group label, left of the bracket

    # Vertical group labels + bracket line for each contiguous learned block.
    trans = ax.get_yaxis_transform()
    groups = {}
    for i, r in enumerate(rows):
        if r[3] is not None:
            groups.setdefault(r[3], []).append(i)
    group_color = {'DRiFt': drift_color, 'Adapter-only': adapter_color}
    for gname, idxs in groups.items():
        lo_i, hi_i = min(idxs), max(idxs)
        gtext = 'Adapter-\nonly' if gname == 'Adapter-only' else gname
        ax.text(label_x, (lo_i + hi_i) / 2.0, gtext, transform=trans, rotation=90,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=group_color.get(gname, '#0b0b0b'))
        ax.plot([bracket_x, bracket_x], [lo_i - 0.33, hi_i + 0.33], transform=trans,
                color='#52514e', lw=1.0, clip_on=False)

    fig.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print('  Saved %s' % out_name)


# ------------------------------------------------------------------ #
#  Per-constellation RMSE TABLE (CSV + LaTeX).
#  This is the raw data behind improvement.pdf (which is its column-mean),
#  so a bar figure of it would be redundant -- a compact table is the right
#  medium. Bold = best (lowest) per column.
# ------------------------------------------------------------------ #
def write_constellation_table(summary, fig_dir):
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))
    spec = [('eskf', 'ESKF'), ('dr_eskf', 'DR-only'),
            ('mamba_eskf', 'Adapter-only (Mamba)'), ('mamba_dr', 'DRiFt (Mamba)'),
            ('gru_eskf', 'Adapter-only (GRU)'), ('gru_dr', 'DRiFt (GRU)'),
            ('transformer_eskf', 'Adapter-only (Transformer)'),
            ('transformer_dr', 'DRiFt (Transformer)')]
    spec = [(k, l) for k, l in spec if k in summary['methods_run']]
    if not spec:
        print('  Skipping constellation table (no methods)')
        return
    cols = [c.replace('const', 'C') for c in const_names] + ['Mean']
    rows = []
    for mkey, lab in spec:
        cr = _get_const_rms(summary, mkey)
        vals = [cr.get(c, float('nan')) for c in const_names]
        finite = [v for v in vals if v < 10]
        rows.append((lab, vals + [np.mean(finite) if finite else float('nan')]))
    best = [int(np.nanargmin([r[1][j] for r in rows])) for j in range(len(cols))]

    with open(os.path.join(fig_dir, 'per_constellation_rmse.csv'), 'w') as f:
        f.write('method,' + ','.join(cols) + '\n')
        for lab, vals in rows:
            f.write(lab + ',' + ','.join('%.3f' % v for v in vals) + '\n')

    with open(os.path.join(fig_dir, 'per_constellation_rmse.tex'), 'w') as f:
        f.write('% Position RMSE [m] per constellation. Bold = best per column.\n')
        f.write('\\begin{tabular}{l' + 'c' * len(cols) + '}\n\\toprule\n')
        f.write('Method & ' + ' & '.join(cols) + ' \\\\\n\\midrule\n')
        for i, (lab, vals) in enumerate(rows):
            cells = [('\\textbf{%.3f}' % v) if i == best[j] else ('%.3f' % v)
                     for j, v in enumerate(vals)]
            f.write(lab + ' & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    print('  Saved per_constellation_rmse.{csv,tex}')


# ------------------------------------------------------------------ #
#  Plot 2d: Error-distribution figure (the headline "insight" figure)
#  Position error vs percentile, pooled over all sequences. DRiFt is
#  lowest at every percentile (~30% reduction). Three interchangeable
#  styles: 'curve' (quantile curve), 'dumbbell', or 'bars'.
# ------------------------------------------------------------------ #
def plot_headline_error_levels(summary, fig_dir, style='curve', out_name=None):
    out_name = out_name or {'curve': 'error_curve.pdf',
                            'dumbbell': 'error_dumbbell.pdf',
                            'bars': 'error_bars.pdf'}.get(style, 'error_levels.pdf')

    def _pool(mkey):
        out = []
        for ds in summary['datasets']:
            f = ds['methods'].get(mkey, {}).get('file', '')
            if not f or not os.path.isfile(f):
                continue
            try:
                d = np.load(f, allow_pickle=True)
            except Exception:
                continue
            if 'pos_error' in d:
                out.append(np.linalg.norm(d['pos_error'], axis=1))
        return np.concatenate(out) if out else np.array([])

    methods = [('ESKF', 'eskf', '#777777'),
               ('DR-only', 'dr_eskf', '#4878CF'),
               ('Adapter-only (Mamba)', 'mamba_eskf', '#2ca02c'),
               ('DRiFt (Mamba)', 'mamba_dr', '#d95f02')]
    arr = {lab: _pool(mk) for lab, mk, _ in methods}
    methods = [(lab, mk, c) for lab, mk, c in methods if len(arr[lab])]
    if len(methods) < 2:
        print('  Skipping error-level figure (no per-step error data)')
        return
    eskf_arr = arr[methods[0][0]]
    drift = methods[-1]
    title = 'DRiFt lowers position error by ~30%, including the worst case'

    if style == 'curve':
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        g = np.linspace(0, 99.5, 200)
        for lab, mk, c in methods:
            ax.plot(g, np.percentile(arr[lab], g), color=c, lw=2.3, label=lab)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, float(np.percentile(eskf_arr, 99.5)) * 1.05)
        ax.set_xlabel('Percentage of time')
        ax.set_ylabel('Position error [m]')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
        ax.grid(alpha=0.3)

    elif style == 'dumbbell':
        pcts = [50, 75, 90, 95, 99]
        yy = np.arange(len(pcts))[::-1]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for yi, p in zip(yy, pcts):
            vals = [(c, float(np.percentile(arr[lab], p))) for lab, mk, c in methods]
            ax.plot([min(v for _, v in vals), max(v for _, v in vals)], [yi, yi],
                    color='#cccccc', lw=2, zorder=1)
            for c, v in vals:
                ax.scatter([v], [yi], color=c, s=70, zorder=3,
                           edgecolors='white', linewidths=0.5)
            e = float(np.percentile(eskf_arr, p))
            f = float(np.percentile(arr[drift[0]], p))
            ax.text(e + 0.012, yi, '%.2f' % e, va='center', fontsize=7.5, color='#555')
            ax.text(f - 0.012, yi, '%.2f' % f, va='center', ha='right',
                    fontsize=7.5, color=drift[2], fontweight='bold')
        ax.set_yticks(yy)
        ax.set_yticklabels(['%dth' % p for p in pcts])
        ax.set_ylabel('Percentile')
        ax.set_xlabel('Position error [m]')
        ax.set_xlim(0, float(np.percentile(eskf_arr, 99)) * 1.18)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                                  markersize=9, label=lab) for lab, mk, c in methods],
                  fontsize=9, loc='upper right', framealpha=0.95)
        ax.grid(axis='x', alpha=0.3)

    else:  # bars
        pcts = [50, 75, 90, 95, 99]
        fig, ax = plt.subplots(figsize=(7.0, 4.1))
        x = np.arange(len(pcts))
        w = 0.26
        for i, (lab, mk, c) in enumerate(methods):
            vals = [float(np.percentile(arr[lab], p)) for p in pcts]
            bars = ax.bar(x + (i - 1) * w, vals, w, color=c, edgecolor='white', label=lab)
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, '%.2f' % v,
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(['%dth' % p for p in pcts])
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Position error [m]')
        ax.set_ylim(0, float(np.percentile(eskf_arr, 99)) * 1.18)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3)

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print('  Saved %s' % out_name)


# ------------------------------------------------------------------ #
#  Plot 2e: Height tracking + vertical-error showcase (paper)
#  One representative hard sequence: estimated height vs ground truth
#  (top) and the vertical position error (bottom) for ESKF / Adapter-only
#  / DRiFt. Shows DRiFt removing ESKF's systematic vertical bias.
# ------------------------------------------------------------------ #
def plot_height_tracking(summary, fig_dir,
                         dataset_name='const4-trial6-tdoa2-traj3',
                         out_name='height_tracking_C4_traj3.pdf', window_s=0.8,
                         bottom='vertical', upper_only=False):
    from matplotlib.lines import Line2D
    ds = next((x for x in summary['datasets']
               if x['dataset_name'] == dataset_name), None)
    if ds is None:
        print('  Skipping height tracking (sequence %s not found)' % dataset_name)
        return

    def _load(mk):
        f = ds['methods'].get(mk, {}).get('file', '')
        return np.load(f, allow_pickle=True) if f and os.path.isfile(f) else None

    E, A, F = _load('eskf'), _load('mamba_eskf'), _load('mamba_dr')
    if E is None or A is None or F is None:
        print('  Skipping height tracking (missing methods)')
        return

    def _t(o):
        t = o['t'].ravel()
        return t - t[0]

    def _err_z(o):
        x = o['pos_error'][:, 2]
        w = max(1, int(window_s * len(x) / _t(o)[-1]))
        return np.convolve(x, np.ones(w) / w, mode='same')

    def _rmse3d(o):
        sq = np.sum(o['pos_error'] ** 2, axis=1)   # ex^2 + ey^2 + ez^2
        w = max(1, int(window_s * len(sq) / _t(o)[-1]))
        return np.sqrt(np.convolve(sq, np.ones(w) / w, mode='same'))

    gz = E['Xpo'][:, 2] - E['pos_error'][:, 2]      # ground-truth height
    tg = _t(E)
    # GT ink / ESKF gray / adapter red / DRiFt violet (FIG_REGIMES_UWB palette)
    BK, GR, GN, OR = '#0b0b0b', '#898781', '#e34948', '#4a3aa7'

    handles = [Line2D([0], [0], color=BK, lw=2.6, label='Ground truth'),
               Line2D([0], [0], color=GR, lw=2.4, label='ESKF'),
               Line2D([0], [0], color=GN, lw=2.4, label='Adapter-only'),
               Line2D([0], [0], color=OR, lw=2.4, label='DRiFt')]
    hmax = max(float(gz.max()), float(E['Xpo'][:, 2].max()),
               float(A['Xpo'][:, 2].max()), float(F['Xpo'][:, 2].max()))

    if upper_only:
        # Single panel: estimated height vs GT only (DRiFt = top overlay).
        fig, ax1 = plt.subplots(figsize=(8.2, 3.7))
        ax1.plot(tg, gz, color=BK, lw=2.6, zorder=1)
        ax1.plot(_t(E), E['Xpo'][:, 2], color=GR, lw=1.4, alpha=0.9, zorder=2)
        ax1.plot(_t(A), A['Xpo'][:, 2], color=GN, lw=1.4, alpha=0.9, zorder=3)
        ax1.plot(_t(F), F['Xpo'][:, 2], color=OR, lw=1.6, alpha=0.95, zorder=4)
        ax1.set_ylabel('Height [m]', fontsize=14)
        ax1.set_xlabel('Time [s]', fontsize=14)
        ax1.tick_params(labelsize=12)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, tg[-1])
        ax1.set_ylim(0, hmax * 1.32)
        ax1.legend(handles=handles, loc='upper center', ncol=4, fontsize=13.5,
                   framealpha=0.92, columnspacing=1.1, handlelength=1.6)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, out_name), bbox_inches='tight')
        plt.close(fig)
        print('  Saved %s' % out_name)
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.5, 1]))
    # top: estimated height vs GT (DRiFt drawn last = top overlay)
    ax1.plot(tg, gz, color=BK, lw=2.6, zorder=1)
    ax1.plot(_t(E), E['Xpo'][:, 2], color=GR, lw=1.4, alpha=0.9, zorder=2)
    ax1.plot(_t(A), A['Xpo'][:, 2], color=GN, lw=1.4, alpha=0.9, zorder=3)
    ax1.plot(_t(F), F['Xpo'][:, 2], color=OR, lw=1.6, alpha=0.95, zorder=4)
    # bottom: vertical error (signed) or rolling 3D RMSE (magnitude)
    if bottom == 'rmse3d':
        ax2.plot(_t(E), _rmse3d(E), color=GR, lw=2.2, zorder=2)
        ax2.plot(_t(A), _rmse3d(A), color=GN, lw=2.2, zorder=3)
        ax2.plot(_t(F), _rmse3d(F), color=OR, lw=2.2, zorder=4)
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('3D RMSE [m]', fontsize=14)
    else:
        ax2.plot(_t(E), _err_z(E), color=GR, lw=2.2, zorder=2)
        ax2.plot(_t(A), _err_z(A), color=GN, lw=2.2, zorder=3)
        ax2.plot(_t(F), _err_z(F), color=OR, lw=2.2, zorder=4)
        ax2.axhline(0, color='#333', lw=0.9, ls='--', zorder=1)
        ax2.set_ylabel('Vertical error [m]', fontsize=14)

    ax1.set_ylabel('Height [m]', fontsize=14)
    ax2.set_xlabel('Time [s]', fontsize=14)
    for a in (ax1, ax2):
        a.tick_params(labelsize=12)
        a.grid(alpha=0.3)
    ax2.set_xlim(0, tg[-1])

    # headroom so the in-plot legend sits in a clean band above the curves
    ax1.set_ylim(0, hmax * 1.32)
    ax1.legend(handles=handles, loc='upper center', ncol=4, fontsize=13.5,
               framealpha=0.92, columnspacing=1.1, handlelength=1.6)
    fig.align_ylabels([ax1, ax2])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print('  Saved %s' % out_name)


# ------------------------------------------------------------------ #
#  Plot 3: Summary heatmap (methods x constellations)
# ------------------------------------------------------------------ #
def plot_summary_heatmap(summary, fig_dir):
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    n_methods = len(methods_run)
    n_cols = len(const_names) + 1  # +1 for Mean column
    grid = np.full((n_methods, n_cols), np.nan)

    for i, mkey in enumerate(methods_run):
        cr = _get_const_rms(summary, mkey)
        row_vals = []
        for j, cn in enumerate(const_names):
            v = cr.get(cn, float('inf'))
            grid[i, j] = v if v < 10 else np.nan
            if v < 10:
                row_vals.append(v)
        grid[i, -1] = np.mean(row_vals) if row_vals else np.nan

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 0.42 * n_methods + 1.2))
    ax.grid(False)

    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        plt.close(fig)
        return
    vmin, vmax = valid.min() * 0.95, valid.max() * 1.05
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)

    # Find best per column
    for j in range(n_cols):
        col = grid[:, j]
        valid_mask = ~np.isnan(col)
        if valid_mask.any():
            best_i = np.nanargmin(col)
            # Bold annotation for best
            for i in range(n_methods):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                rgba = im.cmap(im.norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = 'white' if lum < 0.45 else '#222222'
                weight = 'bold' if i == best_i else 'normal'
                ax.text(j, i, '%.3f' % val, ha='center', va='center',
                        fontsize=8.5, color=txt_color, fontweight=weight)
            # Box around best
            rect = Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                              linewidth=2.0, edgecolor='#222222', facecolor='none', zorder=5)
            ax.add_patch(rect)

    col_labels = [cn.replace('const', 'C') for cn in const_names] + ['Mean']
    row_labels = [METHOD_META.get(m, {}).get('label', m) for m in methods_run]

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Position RMSE [m] — Method vs Constellation')

    fig.colorbar(im, ax=ax, shrink=0.8, label='RMSE [m]')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'summary_heatmap.pdf'))
    plt.close(fig)
    print('  Saved summary_heatmap.pdf')


# ------------------------------------------------------------------ #
#  Plot 4: Predictor noise timeseries (mu_v, R, trace(Q), ||mu_w||)
# ------------------------------------------------------------------ #
def plot_predictor_timeseries(summary, fig_dir, representative_ds=None):
    # Find a representative dataset
    if representative_ds is None:
        # Pick the last dataset (typically const4-trial6)
        representative_ds = summary['datasets'][-1]['dataset_name']

    ds = None
    for d in summary['datasets']:
        if d['dataset_name'] == representative_ds:
            ds = d
            break
    if ds is None:
        print('  Skipping predictor timeseries (dataset %s not found)' % representative_ds)
        return

    # Collect prediction CSVs from any available predictor variant
    # (search order: TAC DR -> CDC DR -> plain ESKF)
    predictor_search = {
        'mamba':       ['mamba_dr',       'mamba_cdc',       'mamba_eskf'],
        'gru':         ['gru_dr',         'gru_cdc',         'gru_eskf'],
        'transformer': ['transformer_dr', 'transformer_cdc', 'transformer_eskf'],
        'sh':          ['sh_dr',          'sh_eskf'],
        'vb':          ['vb_dr',          'vb_eskf'],
    }
    predictor_keys = {p: lst[0] for p, lst in predictor_search.items()}
    predictor_fallback = {p: lst[-1] for p, lst in predictor_search.items()}

    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE_COL_W, 6.0), sharex=True)

    has_data = False
    for pred_name, _ in predictor_keys.items():
        mdata = None
        for cand in predictor_search[pred_name]:
            if ds['methods'].get(cand) is not None:
                mdata = ds['methods'][cand]
                break
        if mdata is None:
            continue
        csv_path = mdata.get('predictions_csv', '')
        if not csv_path or not os.path.isfile(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if 'warmup' in df.columns:
            df = df[df['warmup'] != 1]
        if len(df) == 0:
            continue

        has_data = True
        color = PREDICTOR_COLORS.get(pred_name, '#999')
        steps = df['step'].values
        label = _pred_label(pred_name)

        # Panel 1: mu_v
        if 'mu_v' in df.columns:
            axes[0].plot(steps, df['mu_v'].values, color=color, label=label,
                         linewidth=0.8, alpha=0.85)

        # Panel 2: R
        if 'R' in df.columns:
            axes[1].plot(steps, df['R'].values, color=color, label=label,
                         linewidth=0.8, alpha=0.85)

        # Panel 3: trace(Q)
        if 'Q_acc_diag' in df.columns and 'Q_gyro_diag' in df.columns:
            trace_q = []
            for _, row in df.iterrows():
                qa = _parse_csv_list(row['Q_acc_diag'])
                qg = _parse_csv_list(row['Q_gyro_diag'])
                trace_q.append(qa.sum() + qg.sum())
            axes[2].plot(steps, trace_q, color=color, label=label,
                         linewidth=0.8, alpha=0.85)

        # Panel 4: ||mu_w||
        if 'mu_w_acc' in df.columns and 'mu_w_gyro' in df.columns:
            mu_w_norm = []
            for _, row in df.iterrows():
                ma = _parse_csv_list(row['mu_w_acc'])
                mg = _parse_csv_list(row['mu_w_gyro'])
                mu_w_norm.append(np.sqrt((ma**2).sum() + (mg**2).sum()))
            axes[3].plot(steps, mu_w_norm, color=color, label=label,
                         linewidth=0.8, alpha=0.85)

    if not has_data:
        plt.close(fig)
        print('  Skipping predictor timeseries (no prediction CSVs found)')
        return

    titles = [r'Measurement bias $\mu_v$', r'Measurement variance $\hat{\Sigma}_v$',
              r'Process noise $\mathrm{tr}(\hat{\Sigma}_w)$', r'Process bias $\|\mu_w\|$']
    ylabels = [r'$\mu_v$ [m]', r'$\hat{\Sigma}_v$ [m$^2$]',
               r'$\mathrm{tr}(\hat{\Sigma}_w)$', r'$\|\mu_w\|$']
    for i, (ax, title, ylabel) in enumerate(zip(axes, titles, ylabels)):
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        if i in (1, 2):
            ax.set_yscale('log')

    # Default reference lines
    axes[1].axhline(0.05, color='#999', linestyle='--', linewidth=0.7,
                    label=r'default $\hat{\Sigma}_v$')
    default_trQ = 4.0 * 3 + 0.01 * 3  # default trace(Q)
    axes[2].axhline(default_trQ, color='#999', linestyle='--', linewidth=0.7, label='default')

    axes[-1].set_xlabel('UWB step')
    fig.suptitle('Predicted Noise Parameters — %s' % representative_ds, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'predictor_timeseries.pdf'))
    plt.close(fig)
    print('  Saved predictor_timeseries.pdf')


# ------------------------------------------------------------------ #
#  Plot 5: Noise distribution box plots
# ------------------------------------------------------------------ #
def plot_noise_distributions(summary, fig_dir):
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    # Collect R and trace(Q) distributions from prediction CSVs
    predictor_names = ['mamba', 'gru', 'transformer', 'sh', 'vb']
    pred_search = {p: [p + '_dr', p + '_cdc', p + '_eskf'] for p in predictor_names}

    data_R = {p: {cn: [] for cn in const_names} for p in predictor_names}
    data_trQ = {p: {cn: [] for cn in const_names} for p in predictor_names}

    for ds in summary['datasets']:
        cn = ds['const_name']
        for pred in predictor_names:
            mdata = None
            for cand in pred_search[pred]:
                if ds['methods'].get(cand) is not None:
                    mdata = ds['methods'][cand]
                    break
            if mdata is None:
                continue
            csv_path = mdata.get('predictions_csv', '')
            if not csv_path or not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'warmup' in df.columns:
                df = df[df['warmup'] != 1]

            if 'R' in df.columns:
                data_R[pred][cn].extend(df['R'].values.tolist())
            if 'Q_acc_diag' in df.columns and 'Q_gyro_diag' in df.columns:
                for _, row in df.iterrows():
                    qa = _parse_csv_list(row['Q_acc_diag'])
                    qg = _parse_csv_list(row['Q_gyro_diag'])
                    data_trQ[pred][cn].append(qa.sum() + qg.sum())

    # Check if we have any data
    has_R = any(data_R[p][cn] for p in predictor_names for cn in const_names)
    has_Q = any(data_trQ[p][cn] for p in predictor_names for cn in const_names)
    if not has_R and not has_Q:
        print('  Skipping noise distributions (no prediction data)')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 3.5))

    n_const = len(const_names)
    active_preds = [p for p in predictor_names
                    if any(data_R[p][cn] for cn in const_names)]
    n_pred = len(active_preds)
    if n_pred == 0:
        plt.close(fig)
        return

    # Box plot positions
    group_width = 0.7
    box_w = group_width / n_pred

    for panel_idx, (ax, data, title, ylabel) in enumerate([
        (ax1, data_R, r'Predicted $\hat{\Sigma}_v$', r'$\hat{\Sigma}_v$ [m$^2$]'),
        (ax2, data_trQ, r'Predicted $\mathrm{tr}(\hat{\Sigma}_w)$', r'$\mathrm{tr}(\hat{\Sigma}_w)$'),
    ]):
        positions = []
        box_data = []
        colors = []
        for j, cn in enumerate(const_names):
            for k, pred in enumerate(active_preds):
                pos = j + (k - (n_pred - 1) / 2) * box_w
                vals = data[pred][cn]
                if vals:
                    positions.append(pos)
                    box_data.append(vals)
                    colors.append(PREDICTOR_COLORS.get(pred, '#999'))

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=box_w * 0.8,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='#333', linewidth=1.2))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

        ax.set_xticks(range(n_const))
        ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if panel_idx == 1:
            ax.set_yscale('log')

    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PREDICTOR_COLORS.get(p, '#999'),
                            alpha=0.7, label=_pred_label(p))
                      for p in active_preds]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'noise_distributions.pdf'))
    plt.close(fig)
    print('  Saved noise_distributions.pdf')


# ------------------------------------------------------------------ #
#  Plot 6: 3D trajectory overlay
# ------------------------------------------------------------------ #
def plot_trajectory_overlay(summary, fig_dir, representative_ds=None):
    if representative_ds is None:
        representative_ds = summary['datasets'][-1]['dataset_name']

    ds = None
    for d in summary['datasets']:
        if d['dataset_name'] == representative_ds:
            ds = d
            break
    if ds is None:
        print('  Skipping trajectory overlay (dataset not found)')
        return

    # Load NPZ files for selected methods
    methods_to_plot = [
        ('eskf',         {'color': '#4878CF', 'ls': '-',           'lw': 1.5, 'label': 'ESKF'}),
        ('dr_eskf',      {'color': '#4878CF', 'ls': '--',          'lw': 1.5, 'label': 'DR-ESKF (TAC)'}),
        ('dr_eskf_cdc',  {'color': '#4878CF', 'ls': ':',           'lw': 1.5, 'label': 'DR-ESKF (CDC)'}),
        ('mamba_dr',     {'color': '#2ca02c', 'ls': '-.',          'lw': 1.2, 'label': 'Mamba+DR (TAC)'}),
        ('mamba_cdc',    {'color': '#2ca02c', 'ls': (0,(3,1,1,1)), 'lw': 1.2, 'label': 'Mamba+DR (CDC)'}),
        ('gru_dr',       {'color': '#d62728', 'ls': ':',           'lw': 1.2, 'label': 'GRU+DR (TAC)'}),
        ('gru_cdc',      {'color': '#d62728', 'ls': (0,(2,1)),     'lw': 1.2, 'label': 'GRU+DR (CDC)'}),
        ('transformer_dr',  {'color': '#ff7f0e', 'ls': (0,(5,2,1,2)), 'lw': 1.2, 'label': 'Xfmr+DR (TAC)'}),
        ('transformer_cdc', {'color': '#ff7f0e', 'ls': (0,(1,1)),     'lw': 1.2, 'label': 'Xfmr+DR (CDC)'}),
        ('sh_dr',        {'color': '#9467bd', 'ls': '--',          'lw': 1.2, 'label': 'Sage-Husa+DR (TAC)'}),
        ('vb_dr',        {'color': '#8c564b', 'ls': ':',           'lw': 1.2, 'label': 'VB-AKF+DR (TAC)'}),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL_W, 5.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))

    gt_plotted = False
    for mkey, style in methods_to_plot:
        mdata = ds['methods'].get(mkey)
        if mdata is None:
            continue
        npz_path = mdata.get('file', '')
        if not npz_path or not os.path.isfile(npz_path):
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        # Plot ground truth once
        if not gt_plotted and 'pos_vicon' in data:
            pv = data['pos_vicon']
            ax.plot(pv[:, 0], pv[:, 1], pv[:, 2],
                    color='#333', linewidth=2.0, alpha=0.8, label='Ground Truth')
            if 'anchor_position' in data:
                ap = data['anchor_position']
                ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2],
                           color='teal', s=60, alpha=0.5, edgecolors='black',
                           label='Anchors')
            gt_plotted = True

        Xpo = data['Xpo']
        ax.plot(Xpo[:, 0], Xpo[:, 1], Xpo[:, 2],
                color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'], alpha=0.7, label=style['label'])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title('3D Trajectory — %s' % representative_ds, fontsize=11)
    ax.view_init(24, -58)
    ax.set_box_aspect((1, 1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'trajectory_overlay.pdf'))
    plt.close(fig)
    print('  Saved trajectory_overlay.pdf')


# ------------------------------------------------------------------ #
#  Helper: load prediction data for all predictors across datasets
# ------------------------------------------------------------------ #
def _load_all_predictions(summary):
    '''Load prediction CSVs for all predictor methods. Returns nested dict:
    {pred_name: {const_name: [DataFrame, ...]}}.'''
    predictor_names = ['mamba', 'gru', 'transformer', 'sh', 'vb']
    pred_search = {p: [p + '_dr', p + '_cdc', p + '_eskf'] for p in predictor_names}
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    all_data = {p: {cn: [] for cn in const_names} for p in predictor_names}
    for ds in summary['datasets']:
        cn = ds['const_name']
        for pred in predictor_names:
            mdata = None
            for cand in pred_search[pred]:
                if ds['methods'].get(cand) is not None:
                    mdata = ds['methods'][cand]
                    break
            if mdata is None:
                continue
            csv_path = mdata.get('predictions_csv', '')
            if not csv_path or not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if 'warmup' in df.columns:
                    df = df[df['warmup'] != 1]
                if len(df) > 0:
                    all_data[pred][cn].append(df)
            except Exception:
                continue
    return all_data


# ------------------------------------------------------------------ #
#  Plot 7: mu_v distribution per constellation
# ------------------------------------------------------------------ #
def plot_mu_v_distributions(summary, fig_dir):
    all_data = _load_all_predictions(summary)
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))
    predictor_names = [p for p in ['mamba', 'gru', 'transformer', 'sh', 'vb']
                       if any(all_data[p][cn] for cn in const_names)]
    if not predictor_names:
        print('  Skipping mu_v distributions (no data)')
        return

    n_const = len(const_names)
    n_pred = len(predictor_names)
    group_width = 0.7
    box_w = group_width / n_pred

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.5))

    positions = []
    box_data = []
    colors = []
    for j, cn in enumerate(const_names):
        for k, pred in enumerate(predictor_names):
            pos = j + (k - (n_pred - 1) / 2) * box_w
            vals = []
            for df in all_data[pred][cn]:
                if 'mu_v' in df.columns:
                    vals.extend(df['mu_v'].values.tolist())
            if vals:
                positions.append(pos)
                box_data.append(vals)
                colors.append(PREDICTOR_COLORS.get(pred, '#999'))

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, widths=box_w * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='#333', linewidth=1.2))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

    ax.axhline(0, color='#999', linestyle='--', linewidth=0.7)
    ax.set_xticks(range(n_const))
    ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])
    ax.set_ylabel(r'$\mu_v$ [m]')
    ax.set_title(r'Predicted Measurement Bias $\mu_v$ by Constellation')

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=PREDICTOR_COLORS.get(p, '#999'),
                            alpha=0.7, label=_pred_label(p))
                      for p in predictor_names]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mu_v_distributions.pdf'))
    plt.close(fig)
    print('  Saved mu_v_distributions.pdf')


# ------------------------------------------------------------------ #
#  Plot 8: mu_w component breakdown (6 components per predictor)
# ------------------------------------------------------------------ #
def plot_mu_w_components(summary, fig_dir, representative_ds=None):
    if representative_ds is None:
        representative_ds = summary['datasets'][-1]['dataset_name']

    ds = None
    for d in summary['datasets']:
        if d['dataset_name'] == representative_ds:
            ds = d
            break
    if ds is None:
        print('  Skipping mu_w components (dataset not found)')
        return

    pred_search = {p: [p + '_dr', p + '_cdc', p + '_eskf']
                   for p in ('mamba', 'gru', 'transformer', 'sh', 'vb')}
    comp_labels = [r'$a_x$', r'$a_y$', r'$a_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_W, 4.5), sharex=True)
    axes_flat = axes.flatten()

    has_data = False
    for pred_name in ['mamba', 'gru', 'transformer', 'sh', 'vb']:
        mdata = None
        for cand in pred_search[pred_name]:
            if ds['methods'].get(cand) is not None:
                mdata = ds['methods'][cand]
                break
        if mdata is None:
            continue
        csv_path = mdata.get('predictions_csv', '')
        if not csv_path or not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if 'warmup' in df.columns:
            df = df[df['warmup'] != 1]
        if len(df) == 0 or 'mu_w_acc' not in df.columns:
            continue

        has_data = True
        color = PREDICTOR_COLORS.get(pred_name, '#999')
        steps = df['step'].values

        # Parse 6 components: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        all_comps = []
        for _, row in df.iterrows():
            ma = _parse_csv_list(row['mu_w_acc'])
            mg = _parse_csv_list(row['mu_w_gyro'])
            all_comps.append(np.concatenate([ma, mg]))
        all_comps = np.array(all_comps)  # (N, 6)

        for i in range(6):
            axes_flat[i].plot(steps, all_comps[:, i], color=color,
                              label=_pred_label(pred_name), linewidth=0.7, alpha=0.85)

    if not has_data:
        plt.close(fig)
        print('  Skipping mu_w components (no data)')
        return

    for i, ax in enumerate(axes_flat):
        ax.set_title(comp_labels[i], fontsize=10)
        ax.axhline(0, color='#999', linestyle='--', linewidth=0.5)
        if i >= 3:
            ax.set_xlabel('UWB step')
        if i % 3 == 0:
            ax.set_ylabel(r'$\mu_w$')
        ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(r'Process Bias $\mu_w$ Components — %s' % representative_ds, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mu_w_components.pdf'))
    plt.close(fig)
    print('  Saved mu_w_components.pdf')


# ------------------------------------------------------------------ #
#  Plot 9: DR benefit (delta RMSE: ESKF-only minus DR version)
# ------------------------------------------------------------------ #
def plot_dr_benefit(summary, fig_dir):
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    # Pairs: (eskf_key, dr_key, label, color, hatch)
    # We compare each {none, TAC, CDC} variant of every base method.
    pairs = [
        ('eskf',             'dr_eskf',           'Fixed (TAC)',         '#4878CF', '///'),
        ('eskf',             'dr_eskf_cdc',       'Fixed (CDC)',         '#4878CF', 'xxx'),
        ('mamba_eskf',       'mamba_dr',          'Mamba (TAC)',         '#2ca02c', '///'),
        ('mamba_eskf',       'mamba_cdc',         'Mamba (CDC)',         '#2ca02c', 'xxx'),
        ('gru_eskf',         'gru_dr',            'GRU (TAC)',           '#d62728', '///'),
        ('gru_eskf',         'gru_cdc',           'GRU (CDC)',           '#d62728', 'xxx'),
        ('transformer_eskf', 'transformer_dr',    'Transformer (TAC)',   '#ff7f0e', '///'),
        ('transformer_eskf', 'transformer_cdc',   'Transformer (CDC)',   '#ff7f0e', 'xxx'),
        ('sh_eskf',          'sh_dr',             'Sage-Husa (TAC)',     '#9467bd', '///'),
        ('vb_eskf',          'vb_dr',             'VB-AKF (TAC)',        '#8c564b', '///'),
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.8))
    n_const = len(const_names)
    active_pairs = [(ek, dk, lb, cl, ht) for ek, dk, lb, cl, ht in pairs
                    if ek in summary['methods_run'] and dk in summary['methods_run']]
    n_pairs = len(active_pairs)
    bar_w = 0.85 / max(n_pairs, 1)
    x = np.arange(n_const)

    for i, (ek, dk, label, color, hatch) in enumerate(active_pairs):
        eskf_rms = _get_const_rms(summary, ek)
        dr_rms = _get_const_rms(summary, dk)
        # Positive delta = DR helps (lowers RMSE)
        deltas = [(eskf_rms.get(cn, 0) - dr_rms.get(cn, 0)) * 1000
                  for cn in const_names]  # in mm
        offset = (i - (n_pairs - 1) / 2) * bar_w
        bars = ax.bar(x + offset, deltas, bar_w * 0.9,
                      color=color, edgecolor='white', linewidth=0.5,
                      hatch=hatch, label=label)
        for bar, d in zip(bars, deltas):
            va = 'bottom' if d >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if d >= 0 else -1),
                    '%+.0f' % d, ha='center', va=va, fontsize=5.5)

    ax.axhline(0, color='#333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])
    ax.set_ylabel(r'$\Delta$RMSE [mm] (positive = DR helps)')
    ax.set_title('Effect of DR Robustification per Predictor and Constellation '
                 '(TAC = ///, CDC = xxx)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'dr_benefit.pdf'))
    plt.close(fig)
    print('  Saved dr_benefit.pdf')


# ------------------------------------------------------------------ #
#  Plot 10: Per-dataset RMSE scatter (shows variance within constellations)
# ------------------------------------------------------------------ #
def plot_per_dataset_scatter(summary, fig_dir):
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    # Select key methods to keep the plot readable: 12 methods (4 base x 3 styles)
    key_methods = [m for m in [
        'eskf', 'dr_eskf', 'dr_eskf_cdc',
        'mamba_eskf', 'mamba_dr', 'mamba_cdc',
        'gru_eskf', 'gru_dr', 'gru_cdc',
        'transformer_eskf', 'transformer_dr', 'transformer_cdc',
        'sh_eskf', 'sh_dr', 'vb_eskf', 'vb_dr',
    ] if m in methods_run]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.0))

    n_methods = len(key_methods)
    x = np.arange(n_methods)

    # Scatter individual datasets, color-coded by constellation
    const_colors = {'const1': '#1f77b4', 'const2': '#ff7f0e',
                    'const3': '#2ca02c', 'const4': '#d62728'}
    const_markers = {'const1': 'o', 'const2': 's', 'const3': '^', 'const4': 'D'}

    for ds in summary['datasets']:
        cn = ds['const_name']
        color = const_colors.get(cn, '#999')
        marker = const_markers.get(cn, 'o')
        for j, mkey in enumerate(key_methods):
            mdata = ds['methods'].get(mkey, {})
            rms = mdata.get('rms_all', float('inf'))
            if isinstance(rms, str):
                rms = float('inf')
            if rms < 10:
                ax.scatter(j, rms, color=color, marker=marker, s=40, alpha=0.7,
                           edgecolors='white', linewidths=0.5, zorder=3)

    # Overlay mean as horizontal bars
    for j, mkey in enumerate(key_methods):
        cr = _get_const_rms(summary, mkey)
        overall_mean = np.mean(list(cr.values()))
        ax.plot([j - 0.25, j + 0.25], [overall_mean, overall_mean],
                color='#333', linewidth=2.0, zorder=4)

    method_labels = [METHOD_META.get(m, {}).get('label', m) for m in key_methods]
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('RMSE [m]')
    ax.set_title('Per-Dataset RMSE (dots) with Mean (bar)')

    # Constellation legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=const_markers.get(cn, 'o'), color='w',
                       markerfacecolor=const_colors.get(cn, '#999'), markersize=7,
                       label=cn.replace('const', 'C'))
               for cn in const_names]
    ax.legend(handles=handles, fontsize=8, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'per_dataset_scatter.pdf'))
    plt.close(fig)
    print('  Saved per_dataset_scatter.pdf')


# ------------------------------------------------------------------ #
#  Plot 11: Position error timeseries (representative dataset)
# ------------------------------------------------------------------ #
def plot_position_error_timeseries(summary, fig_dir, representative_ds=None):
    if representative_ds is None:
        representative_ds = summary['datasets'][-1]['dataset_name']

    ds = None
    for d in summary['datasets']:
        if d['dataset_name'] == representative_ds:
            ds = d
            break
    if ds is None:
        print('  Skipping position error timeseries (dataset not found)')
        return

    methods_to_plot = [
        ('eskf',             '#4878CF', '-',   1.2, 'ESKF'),
        ('dr_eskf',          '#4878CF', '--',  1.0, 'DR-ESKF (TAC)'),
        ('dr_eskf_cdc',      '#4878CF', ':',   1.0, 'DR-ESKF (CDC)'),
        ('mamba_eskf',       '#2ca02c', '-',   1.0, 'Mamba'),
        ('gru_dr',           '#d62728', '--',  1.0, 'GRU+DR (TAC)'),
        ('transformer_eskf', '#ff7f0e', '-.',  1.0, 'Transformer'),
        ('sh_eskf',          '#9467bd', '-',   1.0, 'Sage-Husa'),
        ('vb_eskf',          '#8c564b', '-',   1.0, 'VB-AKF'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL_W, 5.0), sharex=True)
    axis_labels = ['X error [m]', 'Y error [m]', 'Z error [m]']

    has_data = False
    for mkey, color, ls, lw, label in methods_to_plot:
        mdata = ds['methods'].get(mkey)
        if mdata is None:
            continue
        npz_path = mdata.get('file', '')
        if not npz_path or not os.path.isfile(npz_path):
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        pos_error = data.get('pos_error')
        t = data.get('t')
        if pos_error is None or t is None:
            continue

        has_data = True
        t_sec = t.ravel()
        # pos_error might be (K, 3) or needs slicing
        if pos_error.ndim == 2 and pos_error.shape[1] >= 3:
            for i in range(3):
                axes[i].plot(t_sec[:len(pos_error)], pos_error[:, i],
                             color=color, linestyle=ls, linewidth=lw,
                             alpha=0.7, label=label)

    if not has_data:
        plt.close(fig)
        print('  Skipping position error timeseries (no NPZ data)')
        return

    for i, ax in enumerate(axes):
        ax.set_ylabel(axis_labels[i])
        ax.axhline(0, color='#999', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Position Error — %s' % representative_ds, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'position_error_timeseries.pdf'))
    plt.close(fig)
    print('  Saved position_error_timeseries.pdf')


# ------------------------------------------------------------------ #
#  Plot 12: Per-constellation RMSE grouped (all 8 methods, single panel)
# ------------------------------------------------------------------ #
def plot_rmse_single_panel(summary, fig_dir):
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))
    n_const = len(const_names)
    n_methods = len(methods_run)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.0))
    bar_w = 0.8 / max(n_methods, 1)
    x = np.arange(n_const)

    for i, mkey in enumerate(methods_run):
        meta = METHOD_META.get(mkey, {})
        const_rms = _get_const_rms(summary, mkey)
        vals = [const_rms.get(cn, float('inf')) for cn in const_names]
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w * 0.9,
               color=meta.get('color', '#999'),
               hatch=meta.get('hatch', ''),
               edgecolor='white', linewidth=0.5,
               label=meta.get('label', mkey))

    ax.set_ylabel('RMSE [m]')
    ax.set_xticks(x)
    ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])
    ax.legend(fontsize=6.5, ncol=4, loc='upper left')
    ax.set_title('Position RMSE — All 8 Methods')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'rmse_bar_single.pdf'))
    plt.close(fig)
    print('  Saved rmse_bar_single.pdf')


# ------------------------------------------------------------------ #
#  Plot 13: Noise calibration — NIS statistics
# ------------------------------------------------------------------ #
def _load_nis(npz_path):
    '''Load NIS values from the .nis.csv alongside an NPZ file.'''
    nis_path = npz_path.replace('.npz', '.nis.csv')
    if not os.path.isfile(nis_path):
        return None
    try:
        data = np.loadtxt(nis_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 1]  # nis_value column
    except Exception:
        return None


def plot_nis_calibration(summary, fig_dir):
    '''NIS mean and 95% pass rate per method — tests measurement noise calibration.
    Well-calibrated: mean NIS ≈ 1.0, pass rate ≈ 95%.'''
    methods_run = summary['methods_run']
    chi2_95 = 3.841  # chi2(1) 95th percentile

    method_nis_mean = {}
    method_nis_pass = {}

    for mkey in methods_run:
        all_nis = []
        for ds in summary['datasets']:
            mdata = ds['methods'].get(mkey, {})
            npz_path = mdata.get('file', '')
            if not npz_path:
                continue
            nis = _load_nis(npz_path)
            if nis is not None and len(nis) > 0:
                all_nis.extend(nis.tolist())

        if all_nis:
            arr = np.array(all_nis)
            method_nis_mean[mkey] = np.mean(arr)
            method_nis_pass[mkey] = np.mean(arr < chi2_95) * 100
        else:
            method_nis_mean[mkey] = float('nan')
            method_nis_pass[mkey] = float('nan')

    active = [m for m in methods_run if not np.isnan(method_nis_mean.get(m, float('nan')))]
    if not active:
        print('  Skipping NIS calibration (no NIS data)')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 3.5))

    # Panel 1: Mean NIS (ideal = 1.0)
    x = np.arange(len(active))
    means = [method_nis_mean[m] for m in active]
    colors = [METHOD_META.get(m, {}).get('color', '#999') for m in active]
    labels = [METHOD_META.get(m, {}).get('label', m) for m in active]
    hatches = [METHOD_META.get(m, {}).get('hatch', '') for m in active]

    bars1 = ax1.bar(x, means, color=colors, edgecolor='white', linewidth=0.5)
    for bar, h in zip(bars1, hatches):
        bar.set_hatch(h)
    ax1.axhline(1.0, color='#333', linestyle='--', linewidth=1.0, label='Ideal (1.0)')
    ax1.axhspan(0.8, 1.2, alpha=0.1, color='green', label='Good range')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax1.set_ylabel('Mean NIS')
    ax1.set_title('Mean NIS (ideal = 1.0)')
    ax1.legend(fontsize=7)
    for i, v in enumerate(means):
        ax1.text(i, v + 0.02, '%.2f' % v, ha='center', va='bottom', fontsize=7)

    # Panel 2: NIS 95% pass rate (ideal = 95%)
    pass_rates = [method_nis_pass[m] for m in active]
    bars2 = ax2.bar(x, pass_rates, color=colors, edgecolor='white', linewidth=0.5)
    for bar, h in zip(bars2, hatches):
        bar.set_hatch(h)
    ax2.axhline(95, color='#333', linestyle='--', linewidth=1.0, label='Ideal (95%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax2.set_ylabel('Pass rate [%]')
    ax2.set_title(r'NIS Pass Rate ($\chi^2_{1,0.95}=3.84$)')
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 105)
    for i, v in enumerate(pass_rates):
        ax2.text(i, v + 0.5, '%.1f%%' % v, ha='center', va='bottom', fontsize=7)

    fig.suptitle('Measurement Noise Calibration (NIS)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nis_calibration.pdf'))
    plt.close(fig)
    print('  Saved nis_calibration.pdf')


# ------------------------------------------------------------------ #
#  Plot 14: NIS per constellation (grouped bars)
# ------------------------------------------------------------------ #
def plot_nis_per_constellation(summary, fig_dir):
    '''Mean NIS per method per constellation — shows which environments
    are well-calibrated for each predictor.'''
    methods_run = summary['methods_run']
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))
    chi2_95 = 3.841

    # Compute mean NIS per method per constellation
    nis_data = {}  # {mkey: {cn: [nis_values]}}
    for mkey in methods_run:
        nis_data[mkey] = {cn: [] for cn in const_names}
        for ds in summary['datasets']:
            cn = ds['const_name']
            mdata = ds['methods'].get(mkey, {})
            npz_path = mdata.get('file', '')
            if not npz_path:
                continue
            nis = _load_nis(npz_path)
            if nis is not None:
                nis_data[mkey][cn].extend(nis.tolist())

    # Select key methods: ESKF baseline + each predictor's TAC and CDC variants
    key_methods = [m for m in ['eskf',
                               'dr_eskf', 'dr_eskf_cdc',
                               'mamba_dr', 'mamba_cdc',
                               'gru_dr', 'gru_cdc',
                               'transformer_dr', 'transformer_cdc',
                               'sh_dr', 'vb_dr']
                   if m in methods_run]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 4.0))
    n_const = len(const_names)
    n_methods = len(key_methods)
    bar_w = 0.7 / max(n_methods, 1)
    x = np.arange(n_const)

    for i, mkey in enumerate(key_methods):
        meta = METHOD_META.get(mkey, {})
        vals = []
        for cn in const_names:
            arr = np.array(nis_data[mkey][cn])
            vals.append(np.mean(arr) if len(arr) > 0 else float('nan'))
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9,
                      color=meta.get('color', '#999'),
                      hatch=meta.get('hatch', ''),
                      edgecolor='white', linewidth=0.5,
                      label=meta.get('label', mkey))

    ax.axhline(1.0, color='#333', linestyle='--', linewidth=1.0, zorder=0)
    ax.axhspan(0.5, 1.5, alpha=0.08, color='green', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([cn.replace('const', 'C') for cn in const_names])
    ax.set_ylabel('Mean NIS')
    ax.set_title('Mean NIS per Constellation (ideal = 1.0, green = good range)')
    ax.legend(fontsize=7, ncol=3, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nis_per_constellation.pdf'))
    plt.close(fig)
    print('  Saved nis_per_constellation.pdf')


# ------------------------------------------------------------------ #
#  Plot 15: NEES — filter consistency (uses pos_error and Ppo)
# ------------------------------------------------------------------ #
def plot_nees(summary, fig_dir):
    '''NEES tests overall filter consistency (both Q and R).
    Well-calibrated: mean NEES ≈ DoF (3 for position).'''
    methods_run = summary['methods_run']
    dof = 3  # position states

    method_nees = {}
    for mkey in methods_run:
        all_nees = []
        for ds in summary['datasets']:
            mdata = ds['methods'].get(mkey, {})
            npz_path = mdata.get('file', '')
            if not npz_path or not os.path.isfile(npz_path):
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                pos_error = data['pos_error']  # (K, 3)
                Ppo = data['Ppo']              # (K, 9, 9)
                # Position block: first 3 states
                for k in range(100, len(pos_error), 10):  # subsample, skip transient
                    e = pos_error[k]
                    P_pos = Ppo[k, :3, :3]
                    try:
                        P_inv = np.linalg.inv(P_pos + 1e-10 * np.eye(3))
                        nees = float(e @ P_inv @ e)
                        if 0 < nees < 1000:  # filter out numerical garbage
                            all_nees.append(nees)
                    except np.linalg.LinAlgError:
                        pass
            except Exception:
                continue
        method_nees[mkey] = np.mean(all_nees) if all_nees else float('nan')

    active = [m for m in methods_run if not np.isnan(method_nees.get(m, float('nan')))]
    if not active:
        print('  Skipping NEES (no data)')
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.5))
    x = np.arange(len(active))
    vals = [method_nees[m] for m in active]
    colors = [METHOD_META.get(m, {}).get('color', '#999') for m in active]
    labels = [METHOD_META.get(m, {}).get('label', m) for m in active]
    hatches = [METHOD_META.get(m, {}).get('hatch', '') for m in active]

    bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.axhline(dof, color='#333', linestyle='--', linewidth=1.0, label='Ideal (DoF=%d)' % dof)
    ax.axhspan(dof * 0.5, dof * 2.0, alpha=0.08, color='green', label='Good range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Mean NEES')
    ax.set_title(r'Position NEES (ideal = %d, tests $\hat{\Sigma}_w + \hat{\Sigma}_v$ calibration)' % dof)
    ax.legend(fontsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, '%.1f' % v, ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'nees_calibration.pdf'))
    plt.close(fig)
    print('  Saved nees_calibration.pdf')


# ------------------------------------------------------------------ #
#  Plot 16: R calibration — predicted R vs empirical residual variance
# ------------------------------------------------------------------ #
def plot_R_calibration(summary, fig_dir):
    '''For each predictor, bin predicted R values and compute empirical
    variance of NIS in each bin. If well-calibrated, mean NIS ≈ 1 in all bins
    (i.e., R matches actual squared innovation).'''
    predictor_names = ['mamba', 'gru', 'transformer', 'sh', 'vb']
    pred_search = {p: [p + '_dr', p + '_cdc', p + '_eskf'] for p in predictor_names}

    fig, axes = plt.subplots(1, len(predictor_names),
                             figsize=(DOUBLE_COL_W, 3.0))

    has_data = False
    for idx, pred in enumerate(predictor_names):
        ax = axes[idx]
        # Collect (R, NIS) pairs from prediction CSV + NIS CSV
        R_vals = []
        nis_vals = []

        for ds in summary['datasets']:
            mdata = None
            for cand in pred_search[pred]:
                if ds['methods'].get(cand) is not None:
                    mdata = ds['methods'][cand]
                    break
            if mdata is None:
                continue
            csv_path = mdata.get('predictions_csv', '')
            npz_path = mdata.get('file', '')
            if not csv_path or not os.path.isfile(csv_path):
                continue
            if not npz_path:
                continue
            nis = _load_nis(npz_path)
            if nis is None:
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'warmup' in df.columns:
                df = df[df['warmup'] != 1]
            if 'R' not in df.columns or len(df) == 0:
                continue

            # Align: prediction CSV has one entry per muse_interval step,
            # NIS has one per UWB measurement. Use all R values broadcast.
            r_arr = df['R'].values
            # Match lengths (take min)
            n = min(len(r_arr), len(nis))
            R_vals.extend(r_arr[:n].tolist())
            nis_vals.extend(nis[:n].tolist())

        if len(R_vals) < 50:
            ax.set_title(_pred_label(pred), fontsize=10)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        has_data = True
        R_arr = np.array(R_vals)
        nis_arr = np.array(nis_vals)

        # Bin by predicted R
        n_bins = 8
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(R_arr, percentiles)
        bin_edges[-1] += 1e-10  # include max

        bin_centers = []
        bin_mean_nis = []
        bin_counts = []

        for b in range(n_bins):
            mask = (R_arr >= bin_edges[b]) & (R_arr < bin_edges[b + 1])
            if mask.sum() < 10:
                continue
            bin_centers.append(np.mean(R_arr[mask]))
            bin_mean_nis.append(np.mean(nis_arr[mask]))
            bin_counts.append(mask.sum())

        color = PREDICTOR_COLORS.get(pred, '#999')
        ax.scatter(bin_centers, bin_mean_nis, color=color, s=40, zorder=3,
                   edgecolors='white', linewidths=0.5)
        ax.plot(bin_centers, bin_mean_nis, color=color, linewidth=1.0, alpha=0.7)
        ax.axhline(1.0, color='#333', linestyle='--', linewidth=0.8)
        ax.axhspan(0.5, 1.5, alpha=0.08, color='green')
        ax.set_xlabel(r'Predicted $\hat{\Sigma}_v$ [m²]')
        if idx == 0:
            ax.set_ylabel('Mean NIS in bin')
        ax.set_title(_pred_label(pred), fontsize=10)
        ax.set_ylim(0, max(3.0, max(bin_mean_nis) * 1.2) if bin_mean_nis else 3.0)

    if not has_data:
        plt.close(fig)
        print('  Skipping R calibration (no paired R/NIS data)')
        return

    fig.suptitle(r'$\hat{\Sigma}_v$ Calibration: Mean NIS vs Predicted $\hat{\Sigma}_v$ (ideal = 1.0 everywhere)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'R_calibration.pdf'))
    plt.close(fig)
    print('  Saved R_calibration.pdf')


# ------------------------------------------------------------------ #
#  Plot 17: Innovation histogram — normalized innovations should be N(0,1)
# ------------------------------------------------------------------ #
def plot_innovation_histogram(summary, fig_dir):
    '''Normalized innovations = innovation / sqrt(S). If well-calibrated,
    should follow N(0,1). Computed as sign(innov) * sqrt(NIS).'''
    methods_to_plot = [
        ('eskf', 'ESKF'),
        ('dr_eskf', 'DR-ESKF (TAC)'),
        ('dr_eskf_cdc', 'DR-ESKF (CDC)'),
        ('mamba_eskf', 'Mamba'),
        ('gru_dr', 'GRU+DR (TAC)'),
        ('transformer_eskf', 'Transformer'),
        ('sh_eskf', 'Sage-Husa'),
        ('vb_eskf', 'VB-AKF'),
    ]
    methods_to_plot = [(m, l) for m, l in methods_to_plot if m in summary['methods_run']]

    if not methods_to_plot:
        print('  Skipping innovation histogram (no methods)')
        return

    fig, axes = plt.subplots(1, len(methods_to_plot),
                             figsize=(DOUBLE_COL_W, 3.0), sharey=True)
    if len(methods_to_plot) == 1:
        axes = [axes]

    # Reference N(0,1)
    z = np.linspace(-4, 4, 200)
    from scipy.stats import norm as norm_dist
    pdf_ref = norm_dist.pdf(z)

    for idx, (mkey, label) in enumerate(methods_to_plot):
        ax = axes[idx]
        all_nis = []
        for ds in summary['datasets']:
            mdata = ds['methods'].get(mkey, {})
            npz_path = mdata.get('file', '')
            if not npz_path:
                continue
            nis = _load_nis(npz_path)
            if nis is not None:
                all_nis.extend(nis.tolist())

        if not all_nis:
            ax.set_title(label)
            continue

        # Convert NIS to normalized innovations (sqrt(NIS), sign unknown so use absolute)
        nis_arr = np.array(all_nis)
        # For histogram: use sqrt(NIS) which should be |N(0,1)| = half-normal
        # Better: show NIS directly as chi2(1)
        ax.hist(nis_arr, bins=50, range=(0, 8), density=True,
                color=METHOD_META.get(mkey, {}).get('color', '#999'),
                alpha=0.7, edgecolor='white', linewidth=0.3)

        # Reference chi2(1) PDF
        from scipy.stats import chi2 as chi2_dist
        x_chi2 = np.linspace(0.01, 8, 200)
        ax.plot(x_chi2, chi2_dist.pdf(x_chi2, df=1),
                color='#333', linewidth=1.5, linestyle='--', label=r'$\chi^2(1)$')

        ax.axvline(3.841, color='red', linewidth=0.8, linestyle=':', alpha=0.7,
                   label='95% threshold')
        ax.set_xlabel('NIS')
        ax.set_title(label, fontsize=10)
        if idx == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=6)
        ax.set_xlim(0, 8)

    fig.suptitle(r'NIS Distribution (should match $\chi^2(1)$ if well-calibrated)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'innovation_histogram.pdf'))
    plt.close(fig)
    print('  Saved innovation_histogram.pdf')


# ------------------------------------------------------------------ #
#  Plot 14: DR theta heatmaps (per method, per constellation + combined)
# ------------------------------------------------------------------ #
def _fmt_theta(v):
    if v >= 1:
        return '%.0f' % v
    elif v >= 0.01:
        return '%.2f' % v
    else:
        return '%.0e' % v


def _draw_heatmap(grid, t1_list, t2_list, t1_label, t2_label, title, out_path):
    '''Render one (n_t1 x n_t2) RMSE heatmap with best-cell highlight.'''
    n_t1 = len(t1_list)
    n_t2 = len(t2_list)
    fig, ax = plt.subplots(figsize=(
        max(3.5, 0.7 * n_t2 + 1.2),
        max(2.5, 0.6 * n_t1 + 1.0)))
    ax.grid(False)
    valid = grid[~np.isnan(grid)]
    vmin, vmax = valid.min() * 0.95, valid.max() * 1.05
    im = ax.imshow(grid, cmap='YlOrRd', aspect='equal',
                   interpolation='nearest', origin='lower',
                   vmin=vmin, vmax=vmax)
    best_flat = np.nanargmin(grid)
    best_i, best_j = divmod(best_flat, n_t2)
    for i in range(n_t1):
        for j in range(n_t2):
            val = grid[i, j]
            if np.isnan(val):
                continue
            rgba = im.cmap(im.norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = 'white' if lum < 0.45 else '#222222'
            weight = 'bold' if (i == best_i and j == best_j) else 'normal'
            ax.text(j, i, '%.3f' % val, ha='center', va='center',
                    fontsize=7.5, color=txt_color, fontweight=weight)
    rect = Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                      linewidth=2.0, edgecolor='#222222',
                      facecolor='none', zorder=5)
    ax.add_patch(rect)
    ax.set_xticks(range(n_t2))
    ax.set_xticklabels([_fmt_theta(v) for v in t2_list])
    ax.set_yticks(range(n_t1))
    ax.set_yticklabels([_fmt_theta(v) for v in t1_list])
    ax.set_xlabel(t2_label)
    ax.set_ylabel(t1_label)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label='RMSE [m]')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dr_heatmaps(summary, fig_dir):
    '''For each DR method, plot the appropriate theta sweep heatmap
    (TAC: theta_w x theta_v; CDC: theta_x x theta_v) per dataset, per
    constellation, and combined (mean across constellations).'''
    const_names = sorted(set(ds['const_name'] for ds in summary['datasets']))

    # DR methods that have grid results, regardless of style
    dr_method_keys = [k for k, v in METHOD_META.items()
                      if v.get('group') in ('tac', 'cdc')]
    dr_method_keys = [k for k in dr_method_keys if k in summary['methods_run']]

    heatmap_dir = os.path.join(fig_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    for mkey in dr_method_keys:
        meta = METHOD_META.get(mkey, {})
        label = meta.get('label', mkey)
        dr_style = meta.get('dr_style', 'tac')

        t1_list, t2_list, _, _, t1_label, t2_label = _theta_axes(summary, dr_style)
        if not t1_list or not t2_list:
            print('  Skipping %s heatmap (no %s grid)' % (mkey, dr_style))
            continue
        n_t1 = len(t1_list)
        n_t2 = len(t2_list)

        # ---- Per-trajectory heatmaps ----
        for ds in summary['datasets']:
            ds_name = ds['dataset_name']
            mdata = ds['methods'].get(mkey, {})
            grid_data = mdata.get('grid', {})
            if not grid_data:
                continue

            grid = np.full((n_t1, n_t2), np.nan)
            for i, t1 in enumerate(t1_list):
                for j, t2 in enumerate(t2_list):
                    gkey = _grid_key(dr_style, t1, t2)
                    entry = grid_data.get(gkey, {})
                    rms = entry.get('rms_all', float('inf'))
                    if isinstance(rms, str):
                        rms = float('inf')
                    if rms < 10:
                        grid[i, j] = rms

            valid = grid[~np.isnan(grid)]
            if len(valid) == 0:
                continue

            _draw_heatmap(grid, t1_list, t2_list, t1_label, t2_label,
                          title='%s — %s' % (label, ds_name.replace('const', 'C')
                                                          .replace('trial', 'T')
                                                          .replace('tdoa', 'D')),
                          out_path=os.path.join(heatmap_dir,
                                                'heatmap_%s_%s.pdf' % (mkey, ds_name)))

        # ---- Per-constellation heatmaps (averaged over trajectories) ----
        const_grids = {}
        for cn in const_names:
            grid = np.full((n_t1, n_t2), np.nan)
            counts = np.zeros((n_t1, n_t2))

            for ds in summary['datasets']:
                if ds['const_name'] != cn:
                    continue
                mdata = ds['methods'].get(mkey, {})
                grid_data = mdata.get('grid', {})
                if not grid_data:
                    continue

                for i, t1 in enumerate(t1_list):
                    for j, t2 in enumerate(t2_list):
                        gkey = _grid_key(dr_style, t1, t2)
                        entry = grid_data.get(gkey, {})
                        rms = entry.get('rms_all', float('inf'))
                        if isinstance(rms, str):
                            rms = float('inf')
                        if rms < 10:
                            if np.isnan(grid[i, j]):
                                grid[i, j] = 0.0
                            grid[i, j] += rms
                            counts[i, j] += 1

            mask = counts > 0
            grid[mask] /= counts[mask]
            const_grids[cn] = grid

            valid = grid[~np.isnan(grid)]
            if len(valid) == 0:
                continue
            _draw_heatmap(grid, t1_list, t2_list, t1_label, t2_label,
                          title='%s — %s' % (label, cn.replace('const', 'C')),
                          out_path=os.path.join(heatmap_dir,
                                                'heatmap_%s_%s.pdf' % (mkey, cn)))

        # ---- Combined heatmap (mean across constellations) ----
        combined = np.full((n_t1, n_t2), 0.0)
        combined_count = np.zeros((n_t1, n_t2))
        for cn, g in const_grids.items():
            mask = ~np.isnan(g)
            combined[mask] += g[mask]
            combined_count[mask] += 1
        mask = combined_count > 0
        combined[mask] /= combined_count[mask]
        combined[~mask] = np.nan

        valid = combined[~np.isnan(combined)]
        if len(valid) == 0:
            continue
        _draw_heatmap(combined, t1_list, t2_list, t1_label, t2_label,
                      title='%s — Combined (mean over constellations)' % label,
                      out_path=os.path.join(heatmap_dir,
                                            'heatmap_%s_combined.pdf' % mkey))

    # ---- Summary: all DR methods side-by-side (combined only) ----
    active_dr = [k for k in dr_method_keys
                 if any(ds['methods'].get(k, {}).get('grid')
                        for ds in summary['datasets'])]
    if active_dr:
        # Each panel is sized for its own grid (TAC vs CDC may differ)
        # so we just use the maximum per-axis count for a uniform width.
        max_t2 = 0
        max_t1 = 0
        for mkey in active_dr:
            style = METHOD_META.get(mkey, {}).get('dr_style', 'tac')
            t1l, t2l, *_ = _theta_axes(summary, style)
            max_t1 = max(max_t1, len(t1l))
            max_t2 = max(max_t2, len(t2l))
        n_methods = len(active_dr)
        fig, axes = plt.subplots(1, n_methods,
                                 figsize=(n_methods * (0.7 * max_t2 + 1.0) + 0.8,
                                          0.6 * max_t1 + 1.5))
        if n_methods == 1:
            axes = [axes]

        for idx, mkey in enumerate(active_dr):
            ax = axes[idx]
            ax.grid(False)
            meta = METHOD_META.get(mkey, {})
            label = meta.get('label', mkey)
            dr_style = meta.get('dr_style', 'tac')
            t1_list, t2_list, _, _, t1_label, t2_label = _theta_axes(summary, dr_style)
            n_t1 = len(t1_list)
            n_t2 = len(t2_list)
            if n_t1 == 0 or n_t2 == 0:
                ax.set_visible(False)
                continue

            # Rebuild combined grid for this method's own theta axes
            combined = np.full((n_t1, n_t2), 0.0)
            combined_count = np.zeros((n_t1, n_t2))
            for ds in summary['datasets']:
                mdata = ds['methods'].get(mkey, {})
                grid_data = mdata.get('grid', {})
                if not grid_data:
                    continue
                for i, t1 in enumerate(t1_list):
                    for j, t2 in enumerate(t2_list):
                        gkey = _grid_key(dr_style, t1, t2)
                        entry = grid_data.get(gkey, {})
                        rms = entry.get('rms_all', float('inf'))
                        if isinstance(rms, str):
                            rms = float('inf')
                        if rms < 10:
                            combined[i, j] += rms
                            combined_count[i, j] += 1
            mask = combined_count > 0
            combined[mask] /= combined_count[mask]
            combined[~mask] = np.nan

            valid = combined[~np.isnan(combined)]
            if len(valid) == 0:
                ax.set_visible(False)
                continue

            vmin_all = valid.min() * 0.95
            vmax_all = valid.max() * 1.05
            im = ax.imshow(combined, cmap='YlOrRd', aspect='equal',
                           interpolation='nearest', origin='lower',
                           vmin=vmin_all, vmax=vmax_all)

            best_flat = np.nanargmin(combined)
            best_i, best_j = divmod(best_flat, n_t2)
            for i in range(n_t1):
                for j in range(n_t2):
                    val = combined[i, j]
                    if np.isnan(val):
                        continue
                    rgba = im.cmap(im.norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    txt_color = 'white' if lum < 0.45 else '#222222'
                    weight = 'bold' if (i == best_i and j == best_j) else 'normal'
                    ax.text(j, i, '%.3f' % val, ha='center', va='center',
                            fontsize=6.5, color=txt_color, fontweight=weight)

            rect = Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                              linewidth=2.0, edgecolor='#222222',
                              facecolor='none', zorder=5)
            ax.add_patch(rect)

            ax.set_xticks(range(n_t2))
            ax.set_xticklabels([_fmt_theta(v) for v in t2_list], fontsize=7)
            ax.set_yticks(range(n_t1))
            ax.set_yticklabels([_fmt_theta(v) for v in t1_list], fontsize=7)
            ax.set_xlabel(t2_label)
            ax.set_ylabel(t1_label)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(label, fontsize=9)

        fig.suptitle(r'DR $\theta$ Sweep — Mean RMSE [m] (all constellations; '
                     r'TAC: $\theta_w\!\times\!\theta_v$, CDC: $\theta_x\!\times\!\theta_v$)',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'dr_heatmaps_combined.pdf'))
        plt.close(fig)

    print('  Saved DR heatmaps (%d methods x %d constellations + combined)' %
          (len(active_dr), len(const_names)))


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description='Generate 8-way comparison plots')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory with eval_8way_summary.json')
    parser.add_argument('--save_only', action='store_true',
                        help='Save PDFs without displaying')
    parser.add_argument('--representative', type=str, default=None,
                        help='Dataset name for timeseries/trajectory plots')
    args = parser.parse_args()

    if args.save_only:
        matplotlib.use('Agg')

    summary = load_summary(args.results_dir)
    fig_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print('Generating 8-way comparison plots...')
    print('  Methods: %s' % ', '.join(summary['methods_run']))
    print('  Datasets: %d' % len(summary['datasets']))

    plot_rmse_bar(summary, fig_dir)
    plot_rmse_single_panel(summary, fig_dir)
    plot_improvement(summary, fig_dir)
    plot_headline_improvement(summary, fig_dir)
    plot_height_tracking(summary, fig_dir)
    plot_height_tracking(summary, fig_dir, bottom='rmse3d',
                         out_name='rmse3d_tracking_C4_traj3.pdf')
    write_constellation_table(summary, fig_dir)
    plot_summary_heatmap(summary, fig_dir)
    plot_dr_benefit(summary, fig_dir)
    plot_per_dataset_scatter(summary, fig_dir)
    plot_predictor_timeseries(summary, fig_dir, args.representative)
    plot_mu_v_distributions(summary, fig_dir)
    plot_noise_distributions(summary, fig_dir)
    plot_mu_w_components(summary, fig_dir, args.representative)
    plot_position_error_timeseries(summary, fig_dir, args.representative)
    plot_trajectory_overlay(summary, fig_dir, args.representative)
    plot_nis_calibration(summary, fig_dir)
    plot_nis_per_constellation(summary, fig_dir)
    plot_nees(summary, fig_dir)
    plot_R_calibration(summary, fig_dir)
    plot_innovation_histogram(summary, fig_dir)
    plot_dr_heatmaps(summary, fig_dir)

    print('\nAll plots saved to %s' % fig_dir)


if __name__ == '__main__':
    main()
