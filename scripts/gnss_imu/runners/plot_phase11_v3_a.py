'''Per-variant plots from the 64-grid DR sweep (plot title controlled by
--variant_label, auto-detected from data_root's last component).

Reads:
  results_gnss/i2nav/phase11/v3_a/dr_sweep_baseline/<seq>_tw<W>_tv<V>.json
  results_gnss/i2nav/phase11/v3_a/dr_sweep_adapter_muv/<seq>_tw<W>_tv<V>.json
  results_gnss/i2nav/phase11/v3_a/traj/<seq>_<cond>.npz   (for trajectory plots)

Produces 6 plots, each answering a distinct question:

  1. dr_heatmap_aggregate.{pdf,png}
       Q: Across the full theta_w × theta_v grid, what's the median RMSE
          surface for baseline-DR and adapter+DR? Where's the optimum?
       Form: 1×2 panel — left = baseline DR sweep heatmap, right = adapter+DR
              sweep heatmap. Best cell highlighted in each.

  2. dr_heatmap_per_seq.{pdf,png}
       Q: Does the optimal theta vary across sequences? Are there
          regimes where the adapter changes the heatmap shape vs baseline?
       Form: 2 rows (baseline, adapter) × 7 cols (seqs, parking00 excluded).

  3. headline_per_seq.{pdf,png}
       Q: At the chosen best-theta, how do the 4 methods compare per seq?
       Form: 4-bar group per sequence (baseline / adapter / DR / adapter+DR).
              Adapter+DR bars annotated with %-vs-baseline.

  4. improvement_summary.{pdf,png}
       Q: What's the median improvement of each method? Is the DR-marginal
          on top of the adapter positive?
       Form: 3-bar improvement (with IQR whiskers) plus an isolated
              DR-marginal bar (the headline metric of this phase).

  5. nis_calibration.{pdf,png}
       Q: Did lambda_cal=0.05 actually pull the post-adapter NIS toward 1?
       Form: per-sequence mean NIS for each method, with NIS=1 reference
              and shaded "well-calibrated" band.

  6. trajectory_overlay.{pdf,png}
       Q: Visually, does adapter+DR track better than baseline on hard seqs?
       Form: 1×3 N-E trajectory overlays (street02 / building00 / street00).

The "best theta" used for plots 3-5 is the (theta_w, theta_v) that minimizes
the median horizontal RMSE across the 7 well-behaved sequences (parking00
excluded). It's chosen *separately* for the baseline-DR and adapter+DR sweeps
because the optimal theta differs between the two.
'''
import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# ---- Constants matching sweep_dr_theta.py ----
THETA_W_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
THETA_V_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

NORMAL_SEQS = ['building00', 'building01', 'building02',
               'playground00', 'street00', 'street01', 'street02']
ALL_SEQS = NORMAL_SEQS + ['parking00']

# Method colors / labels for non-heatmap plots
COND_COLORS = {
    'baseline':    '#888888',
    'adapter':     '#2ca02c',
    'dr':          '#4878CF',
    'adapter_dr':  '#d62728',
}
COND_LABELS = {
    'baseline':    'baseline',
    'adapter':     'adapter alone',
    'dr':          'DR alone',
    'adapter_dr':  'adapter + DR',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
})


def _short(seq):
    return (seq.replace('building', 'b')
               .replace('playground', 'plg')
               .replace('street', 's')
               .replace('parking', 'park'))


def _save(fig, path):
    fig.savefig(path)
    fig.savefig(path[:-4] + '.png', dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def _fmt_t(v):
    if v == 0:
        return '0'
    if v >= 1:
        return f'{v:g}'
    return f'{v:.2g}'


# ------------------------------------------------------------------ #
#  Load grid sweeps
# ------------------------------------------------------------------ #
def load_sweep(root):
    '''Return a 4-D dict: data[seq][(tw, tv)] = {'rmse_h', 'nis_mean', ...}.'''
    data = {s: {} for s in ALL_SEQS}
    if not os.path.isdir(root):
        print(f'  [warn] sweep dir missing: {root}')
        return data
    for fn in os.listdir(root):
        if not fn.endswith('.json'):
            continue
        # Parse pattern <seq>_tw<W>_tv<V>.json
        try:
            base = fn[:-5]   # strip .json
            seq, tw_part, tv_part = base.rsplit('_', 2)
            assert tw_part.startswith('tw') and tv_part.startswith('tv')
            tw = float(tw_part[2:]); tv = float(tv_part[2:])
        except (ValueError, AssertionError):
            continue
        if seq not in data:
            continue
        try:
            d = json.load(open(os.path.join(root, fn)))
            d = d[0] if isinstance(d, list) else d
            data[seq][(tw, tv)] = {
                'rmse_h':    float(d['rmse']['rmse_h']),
                'rmse_3d':   float(d['rmse']['rmse_3d']),
                'nis_mean':  float(d.get('nis', {}).get('nis_mean', float('nan'))),
                'nis_pass95':float(d.get('nis', {}).get('nis_pass_95', float('nan'))),
            }
        except Exception as e:
            print(f'    skip {fn}: {e}')
    return data


def grid_matrix(sweep_data, seqs, metric='rmse_h', agg='median'):
    '''Build (n_tw, n_tv) matrix of aggregated metric across given seqs.'''
    n_tw = len(THETA_W_GRID)
    n_tv = len(THETA_V_GRID)
    mat = np.full((n_tw, n_tv), np.nan)
    for i, tw in enumerate(THETA_W_GRID):
        for j, tv in enumerate(THETA_V_GRID):
            vals = []
            for s in seqs:
                cell = sweep_data.get(s, {}).get((tw, tv))
                if cell is None:
                    continue
                v = cell.get(metric, float('nan'))
                if not math.isnan(v):
                    vals.append(v)
            if vals:
                if agg == 'median':
                    mat[i, j] = float(np.median(vals))
                elif agg == 'mean':
                    mat[i, j] = float(np.mean(vals))
                else:
                    raise ValueError(agg)
    return mat


def find_best(mat, exclude_zero=False):
    '''Return (i, j, value) of the minimum cell. exclude_zero=True skips
    the (0,0) corner which corresponds to "no DR".'''
    work = mat.copy()
    if exclude_zero:
        work[0, 0] = np.inf
    if np.all(np.isnan(work)):
        return None, None, None
    flat = np.nanargmin(work)
    i, j = divmod(flat, work.shape[1])
    return i, j, float(mat[i, j])


# ------------------------------------------------------------------ #
#  Plot 1 — aggregate heatmaps (baseline DR sweep + adapter+DR sweep)
# ------------------------------------------------------------------ #
def plot_dr_heatmap_aggregate_for(base_sweep, adp_sweep, seqs, out_path, subset_label):
    base_mat = grid_matrix(base_sweep, seqs, agg='mean')
    adp_mat  = grid_matrix(adp_sweep,  seqs, agg='mean')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    titles = ['baseline DR sweep (no adapter)', 'adapter + DR sweep (μv-only)']
    for ax, mat, title in zip(axes, [base_mat, adp_mat], titles):
        valid = mat[~np.isnan(mat)]
        if len(valid) == 0:
            ax.set_visible(False); continue
        vmin, vmax = float(valid.min()) * 0.96, float(valid.max()) * 1.04
        im = ax.imshow(mat, cmap='YlOrRd', aspect='equal',
                       interpolation='nearest', origin='lower',
                       vmin=vmin, vmax=vmax)
        # Annotate
        bi, bj, bv = find_best(mat, exclude_zero=False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if math.isnan(v): continue
                rgba = im.cmap(im.norm(v))
                lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                col = 'white' if lum < 0.45 else '#222'
                weight = 'bold' if (i == bi and j == bj) else 'normal'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=col, fontsize=7.5, fontweight=weight)
        # Box on best
        if bi is not None:
            rect = Rectangle((bj-0.5, bi-0.5), 1, 1, linewidth=2,
                             edgecolor='#222', facecolor='none', zorder=5)
            ax.add_patch(rect)
        ax.set_xticks(range(len(THETA_V_GRID)))
        ax.set_xticklabels([_fmt_t(v) for v in THETA_V_GRID], fontsize=7)
        ax.set_yticks(range(len(THETA_W_GRID)))
        ax.set_yticklabels([_fmt_t(v) for v in THETA_W_GRID], fontsize=7)
        ax.set_xlabel(r'$\theta_v$ (measurement-noise BW radius)')
        ax.set_ylabel(r'$\theta_w$ (process-noise BW radius)')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        bv_str = f' best (θ_w={_fmt_t(THETA_W_GRID[bi])}, θ_v={_fmt_t(THETA_V_GRID[bj])}) → {bv:.2f} m' if bi is not None else ''
        ax.set_title(f'{title}\nmean h-RMSE [m]; subset = {subset_label}{bv_str}',
                     fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label='mean h-RMSE [m]')

    fig.suptitle(f'i2Nav {VARIANT_LABEL} — DR θ-grid sweep (8×8=64 configs/panel; {subset_label})',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 2 — per-sequence heatmap grid
# ------------------------------------------------------------------ #
def plot_dr_heatmap_per_seq(base_sweep, adp_sweep, out_path):
    seqs = NORMAL_SEQS
    n = len(seqs)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n + 1.4, 4.6),
                             sharex=True, sharey=True)

    for col, seq in enumerate(seqs):
        for row, (sweep, name) in enumerate([(base_sweep, 'baseline'),
                                              (adp_sweep,  'adapter')]):
            ax = axes[row, col]
            mat = grid_matrix({seq: sweep.get(seq, {})}, [seq])
            valid = mat[~np.isnan(mat)]
            if len(valid) == 0:
                ax.set_visible(False); continue
            vmin, vmax = float(valid.min()) * 0.95, float(valid.max()) * 1.05
            im = ax.imshow(mat, cmap='YlOrRd', aspect='equal',
                           origin='lower', vmin=vmin, vmax=vmax)
            bi, bj, _ = find_best(mat)
            if bi is not None:
                rect = Rectangle((bj-0.5, bi-0.5), 1, 1, linewidth=1.4,
                                 edgecolor='#222', facecolor='none', zorder=5)
                ax.add_patch(rect)
            ax.set_xticks(range(0, len(THETA_V_GRID), 2))
            ax.set_xticklabels([_fmt_t(v) for v in THETA_V_GRID[::2]],
                               fontsize=6, rotation=0)
            ax.set_yticks(range(0, len(THETA_W_GRID), 2))
            ax.set_yticklabels([_fmt_t(v) for v in THETA_W_GRID[::2]],
                               fontsize=6)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(_short(seq), fontsize=8)
            if col == 0:
                ax.set_ylabel(name + r'$\,$θ_w', fontsize=8)
            if row == 1:
                ax.set_xlabel(r'θ_v', fontsize=8)

    fig.suptitle(f'i2Nav {VARIANT_LABEL} — per-sequence DR heatmaps '
                 '(top: no adapter; bottom: with adapter; lower = better)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Helpers for non-heatmap plots: extract the 4 conditions at best theta
# ------------------------------------------------------------------ #
def select_best_theta(sweep_data, seqs):
    '''Return ((tw_idx, tv_idx), (tw_value, tv_value)).
    Uses MEAN aggregation across seqs (matches eval_8way_v4 convention,
    so a single divergent seq is correctly penalised).'''
    mat = grid_matrix(sweep_data, seqs, agg='mean')
    bi, bj, _ = find_best(mat, exclude_zero=True)
    return (bi, bj), (THETA_W_GRID[bi], THETA_V_GRID[bj])


def build_condition_table(base_sweep, adp_sweep, base_theta, adp_theta):
    '''For each seq, extract 4 conditions:
      baseline  = base_sweep[seq][(0,0)]
      adapter   = adp_sweep[seq][(0,0)]
      dr        = base_sweep[seq][best baseline theta]
      adapter_dr = adp_sweep[seq][best adapter theta]'''
    out = {}
    for seq in ALL_SEQS:
        b00 = base_sweep.get(seq, {}).get((0.0, 0.0))
        a00 = adp_sweep.get(seq, {}).get((0.0, 0.0))
        bdr = base_sweep.get(seq, {}).get(base_theta)
        adr = adp_sweep.get(seq, {}).get(adp_theta)
        if b00 is None or a00 is None or bdr is None or adr is None:
            continue
        out[seq] = {
            'baseline':   b00,
            'adapter':    a00,
            'dr':         bdr,
            'adapter_dr': adr,
        }
    return out


# ------------------------------------------------------------------ #
#  Plot 3 — per-sequence headline at best theta
# ------------------------------------------------------------------ #
def plot_headline_per_seq(cond_table, subset_seqs, out_path, base_theta, adp_theta, subset_label):
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs:
        print('  skip headline_per_seq — no data'); return

    # If parking00 is *outside* the subset but we have data for it, show in side panel
    show_parking_inset = ('parking00' not in subset_seqs and 'parking00' in cond_table)
    fig = plt.figure(figsize=(9.0, 4.6))
    if show_parking_inset:
        gs = fig.add_gridspec(1, 5, wspace=0.32)
        ax = fig.add_subplot(gs[0, :4])
        ax_park = fig.add_subplot(gs[0, 4])
    else:
        ax = fig.add_subplot(111)
        ax_park = None

    conds = ['baseline', 'dr', 'adapter', 'adapter_dr']
    bw = 0.8 / len(conds)
    x = np.arange(len(seqs))
    for i, c in enumerate(conds):
        vals = [cond_table[s][c]['rmse_h'] for s in seqs]
        offset = (i - (len(conds) - 1) / 2) * bw
        ax.bar(x + offset, vals, bw * 0.92,
               color=COND_COLORS[c], edgecolor='white', linewidth=0.4,
               label=COND_LABELS[c])
    ax.set_xticks(x)
    ax.set_xticklabels([_short(s) for s in seqs], fontsize=9)
    ax.set_ylabel('horizontal RMSE [m]')
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    # Annotate adapter+DR with % vs baseline
    for j, s in enumerate(seqs):
        base = cond_table[s]['baseline']['rmse_h']
        apdr = cond_table[s]['adapter_dr']['rmse_h']
        if base > 0:
            pct = 100 * (base - apdr) / base
            offset = ((conds.index('adapter_dr') - (len(conds) - 1) / 2) * bw)
            ax.text(j + offset, apdr + 0.08, f'{pct:+.0f}%',
                    ha='center', va='bottom', fontsize=6.5,
                    color=COND_COLORS['adapter_dr'])

    if base_theta is None or adp_theta is None:
        theta_line = 'DR θ chosen per-seq from 8x8 grid'
    else:
        theta_line = (f'(DR alone θ_w={_fmt_t(base_theta[0])}, θ_v={_fmt_t(base_theta[1])};  '
                      f'adapter+DR θ_w={_fmt_t(adp_theta[0])}, θ_v={_fmt_t(adp_theta[1])})')
    ax.set_title(f'i2Nav {VARIANT_LABEL} — per-sequence performance at best θ '
                 f'({subset_label})\n{theta_line}',
                 fontsize=10)

    if ax_park is not None and 'parking00' in cond_table:
        s = 'parking00'
        for i, c in enumerate(conds):
            v = cond_table[s][c]['rmse_h']
            offset = (i - (len(conds) - 1) / 2) * 0.8
            ax_park.bar(offset, v, 0.7, color=COND_COLORS[c],
                        edgecolor='white', linewidth=0.4)
        ax_park.set_xticks([])
        ax_park.set_ylabel('h-RMSE [m]', fontsize=9)
        ax_park.set_title('parking00\n(out-of-scale)', fontsize=9)
        ax_park.grid(True, axis='y', alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 4 — improvement summary
# ------------------------------------------------------------------ #
def plot_improvement_summary(cond_table, subset_seqs, out_path, subset_label):
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs:
        return

    def mean_rms(c):
        return float(np.mean([cond_table[s][c]['rmse_h'] for s in seqs]))
    def std_rms(c):
        return float(np.std([cond_table[s][c]['rmse_h'] for s in seqs]))

    base = mean_rms('baseline')
    keys = ['dr', 'adapter', 'adapter_dr']
    pcts = [100 * (base - mean_rms(k)) / base for k in keys]
    # ±1σ in % of baseline as whiskers (mean-aggregate analog of IQR).
    err_lo = [100 * (base - (mean_rms(k) + std_rms(k))) / base for k in keys]
    err_hi = [100 * (base - (mean_rms(k) - std_rms(k))) / base for k in keys]

    # DR marginal on adapter
    apd = mean_rms('adapter_dr')
    adp = mean_rms('adapter')
    marg = 100 * (adp - apd) / adp

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8),
                             gridspec_kw={'width_ratios': [1.6, 1.0]})

    ax = axes[0]
    x = np.arange(len(keys))
    bars = ax.bar(x, pcts,
                  color=[COND_COLORS[k] for k in keys],
                  edgecolor='white', linewidth=0.5)
    for i in range(len(keys)):
        ax.plot([i, i], [err_lo[i], err_hi[i]], color='#222', linewidth=1.2)
        ax.plot([i-0.10, i+0.10], [err_lo[i], err_lo[i]], color='#222', linewidth=1.2)
        ax.plot([i-0.10, i+0.10], [err_hi[i], err_hi[i]], color='#222', linewidth=1.2)
    for i, p in enumerate(pcts):
        ax.text(i, p + 0.6, f'{p:+.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(0, color='#333', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel(f'Δ horizontal RMSE vs baseline (%)\n(mean across {len(seqs)} seqs; whiskers = ±1σ)')
    ax.set_title(f'Aggregate improvement ({subset_label})', fontsize=11)

    # DR marginal panel
    ax2 = axes[1]
    color = '#2ca02c' if marg > 0 else '#d62728'
    ax2.bar(0, marg, 0.6, color=color, edgecolor='white', linewidth=0.5)
    ax2.text(0, marg + (1 if marg >= 0 else -2),
             f'{marg:+.1f}%', ha='center',
             va='bottom' if marg >= 0 else 'top',
             fontsize=12, fontweight='bold')
    ax2.axhline(0, color='#333', linewidth=0.7)
    ax2.set_xticks([])
    ax2.set_ylabel('Δ DR marginal on adapter (%)\n(positive = DR adds value)')
    ax2.set_title('Did DR earn its keep?', fontsize=11)
    pad = 5
    ax2.set_ylim(min(-pad, marg - pad), max(15, marg + pad))

    fig.suptitle(f'i2Nav {VARIANT_LABEL} adapter',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 5 — NIS calibration
# ------------------------------------------------------------------ #
def plot_nis_calibration(cond_table, subset_seqs, out_path, subset_label):
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.axhspan(0.7, 1.5, alpha=0.10, color='#2ca02c', zorder=0,
               label='well-calibrated band')
    ax.axhline(1.0, color='#222', linewidth=0.8, linestyle='--', zorder=1,
               label='ideal NIS = 1')

    conds = ['baseline', 'dr', 'adapter', 'adapter_dr']
    bw = 0.8 / len(conds)
    x = np.arange(len(seqs))
    for i, c in enumerate(conds):
        vals = [cond_table[s][c].get('nis_mean', float('nan')) for s in seqs]
        offset = (i - (len(conds) - 1) / 2) * bw
        ax.bar(x + offset, vals, bw * 0.92,
               color=COND_COLORS[c], edgecolor='white', linewidth=0.4,
               label=COND_LABELS[c])
    ax.set_xticks(x)
    ax.set_xticklabels([_short(s) for s in seqs])
    ax.set_ylabel('mean NIS on test trajectory')
    ax.set_title(f'i2Nav {VARIANT_LABEL} — NIS calibration per method, per sequence ({subset_label})\n'
                 'λ_cal pulls adapter NIS toward 1 vs the underconfident baseline',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 6 — trajectory overlays
# ------------------------------------------------------------------ #
def plot_trajectory(traj_root, out_path, seqs_to_show=None):
    if seqs_to_show is None:
        seqs_to_show = ['street02', 'building00', 'street00']
    fig, axes = plt.subplots(1, len(seqs_to_show),
                             figsize=(4.0 * len(seqs_to_show), 4.5))
    if len(seqs_to_show) == 1:
        axes = [axes]
    has_data = False
    for ax, seq in zip(axes, seqs_to_show):
        gt_plotted = False
        for cond_file, cond_label in [
            ('baseline',    'baseline'),
            ('adapter_muv', 'adapter alone'),
            ('adapter_dr',  'adapter + DR'),
        ]:
            npz_path = os.path.join(traj_root, f'{seq}_{cond_file}.npz')
            if not os.path.isfile(npz_path):
                continue
            try:
                d = np.load(npz_path, allow_pickle=True)
                est = np.asarray(d.get('est_ned'))
                gt = np.asarray(d.get('gt_ned'))
                if est is None or gt is None:
                    continue
            except Exception:
                continue
            has_data = True
            if not gt_plotted:
                ax.plot(gt[:, 1], gt[:, 0], color='#222', linewidth=2.0,
                        alpha=0.85, label='ground truth', zorder=2)
                gt_plotted = True
            color_key = 'baseline' if cond_file == 'baseline' \
                else 'adapter' if cond_file == 'adapter_muv' \
                else 'adapter_dr'
            ls = '-' if cond_file == 'adapter_dr' else ('--' if cond_file == 'adapter_muv' else ':')
            lw = 1.4 if cond_file == 'adapter_dr' else 1.0
            ax.plot(est[:, 1], est[:, 0],
                    color=COND_COLORS[color_key],
                    linewidth=lw, linestyle=ls,
                    alpha=0.85, label=cond_label, zorder=3)
        ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(seq, fontsize=10)
        ax.legend(fontsize=7, loc='best')
    if not has_data:
        plt.close(fig)
        print('  skip trajectory_overlay — no .npz sidecars in traj/')
        return
    fig.suptitle(f'i2Nav {VARIANT_LABEL} — N-E trajectory overlay (representative seqs)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  eval_8way_v4-style headline trio: improvement averaged + per-seq +
#  rmse_bar grouped. Designed to match the look of
#    results/eval_8way_v4/figures/{improvement_8way,rmse_bar_8way}.pdf.
# ------------------------------------------------------------------ #
METHOD_META_8WAY = {
    'baseline':   {'color': '#4878CF', 'hatch': '',    'label': 'ESKF baseline'},
    'adapter':    {'color': '#2ca02c', 'hatch': '',    'label': 'adapter (μv)'},
    'dr':         {'color': '#bcbd22', 'hatch': '///', 'label': 'Baseline + DR'},
    'adapter_dr': {'color': '#d62728', 'hatch': '///', 'label': 'adapter + DR'},
}


def plot_improvement_averaged(cond_table, subset_seqs, out_path, subset_label,
                              font_scale=1.0, show_title=True,
                              xlabel=r'Improvement over ESKF baseline [%]'):
    """Aggregate Δ% over seqs using MEAN of RMSE (per-seq divergences
    propagate, no median-style outlier robustness).  Matches the eval_8way_v4
    convention (mean across constellations).  Honest about the fact that
    a single bad seq can sink the recipe."""
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs: return
    base_mn = float(np.mean([cond_table[s]['baseline']['rmse_h'] for s in seqs]))
    if base_mn <= 0: return
    keys = ['dr', 'adapter', 'adapter_dr']   # fixed display order
    pcts = []
    for k in keys:
        m = float(np.mean([cond_table[s][k]['rmse_h'] for s in seqs]))
        pcts.append((k, 100*(base_mn - m)/base_mn))
    # Display order is fixed (baseline+DR → adapter → adapter+DR), not sorted by Δ%.
    fig, ax = plt.subplots(figsize=(7.16, 0.55*len(pcts) + 1.2))
    y = np.arange(len(pcts))
    labels = [METHOD_META_8WAY[k]['label'] for k, _ in pcts]
    colors = [METHOD_META_8WAY[k]['color'] for k, _ in pcts]
    vals   = [v for _, v in pcts]
    bars = ax.barh(y, vals, color=colors, edgecolor='white', height=0.6)
    for bar, (k, _) in zip(bars, pcts):
        if METHOD_META_8WAY[k]['hatch']:
            bar.set_hatch(METHOD_META_8WAY[k]['hatch'])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9 * font_scale)
    ax.set_xlabel(xlabel, fontsize=11 * font_scale)
    ax.tick_params(axis='x', labelsize=9 * font_scale)
    ax.axvline(0, color='#333', lw=0.8, zorder=0)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.4 if v >= 0 else v - 0.4, i,
                f'{v:+.1f}%', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9 * font_scale)
    if show_title:
        ax.set_title(f'i2Nav {VARIANT_LABEL} — RMSE Improvement Over ESKF Baseline\n'
                     f'(mean across {len(seqs)} seqs, {subset_label})')
    fig.tight_layout()
    _save(fig, out_path)


def plot_improvement_per_seq(cond_table, subset_seqs, out_path, subset_label):
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs: return
    keys = ['dr', 'adapter', 'adapter_dr']
    n = len(keys)
    bw = 0.8 / n
    x = np.arange(len(seqs))
    fig, ax = plt.subplots(figsize=(max(7.16, 0.9*len(seqs)+2.0), 4.0))
    for i, k in enumerate(keys):
        meta = METHOD_META_8WAY[k]
        vals = []
        for s in seqs:
            b = cond_table[s]['baseline']['rmse_h']
            v = cond_table[s][k]['rmse_h']
            vals.append(100*(b-v)/b if b > 0 else 0)
        offset = (i - (n-1)/2) * bw
        bars = ax.bar(x + offset, vals, bw*0.92, color=meta['color'],
                      hatch=meta['hatch'], edgecolor='white', linewidth=0.5,
                      label=meta['label'])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + (0.6 if v >= 0 else -0.6),
                    f'{v:+.0f}%', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=7)
    ax.axhline(0, color='#333', lw=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(s) for s in seqs], fontsize=9)
    ax.set_ylabel(r'Improvement vs ESKF baseline [%]')
    ax.set_title(f'i2Nav {VARIANT_LABEL} — per-sequence improvement over baseline ({subset_label})')
    ax.legend(loc='best', fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3, lw=0.4)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Per-seq-best-θ aggregation (matches eval_8way_v4 UWB convention).
#  Each seq picks its OWN optimal θ from the 8×8 grid, then we report
#  the mean across seqs. This is the right comparator: DR θ is a
#  deployment-time hyperparameter that can be conditioned per-environment.
# ------------------------------------------------------------------ #
def _per_seq_best_psb(sweep_data, seqs, exclude_zero=False):
    """For each seq, return ((tw, tv), best_rmse_h) — own-optimal θ."""
    out = {}
    for s in seqs:
        cells = sweep_data.get(s, {})
        if not cells:
            continue
        best = (None, None, float('inf'))
        for (tw, tv), c in cells.items():
            if exclude_zero and tw == 0 and tv == 0:
                continue
            v = c.get('rmse_h', float('nan'))
            if math.isnan(v):
                continue
            if v < best[2]:
                best = (tw, tv, v)
        if best[0] is not None:
            out[s] = best
    return out


def build_psb_cond_table(base_sweep, adp_sweep, seqs):
    """Build cond_table where 'dr' and 'adapter_dr' use per-seq own-best θ
    (each seq picks its own optimal). 'baseline' and 'adapter' remain at (0,0)
    since adapter is independent of θ."""
    base_psb = _per_seq_best_psb(base_sweep, seqs, exclude_zero=False)
    adp_psb  = _per_seq_best_psb(adp_sweep,  seqs, exclude_zero=False)
    out = {}
    for s in seqs:
        b00 = base_sweep.get(s, {}).get((0.0, 0.0))
        a00 = adp_sweep.get(s, {}).get((0.0, 0.0))
        bdr = base_psb.get(s)
        adr = adp_psb.get(s)
        if any(x is None for x in (b00, a00, bdr, adr)):
            continue
        out[s] = {
            'baseline':   b00,
            'adapter':    a00,
            'dr':         {'rmse_h': bdr[2],
                           'rmse_3d': base_sweep[s][(bdr[0], bdr[1])].get('rmse_3d', float('nan')),
                           'nis_mean': base_sweep[s][(bdr[0], bdr[1])].get('nis_mean', float('nan')),
                           'theta': (bdr[0], bdr[1])},
            'adapter_dr': {'rmse_h': adr[2],
                           'rmse_3d': adp_sweep[s][(adr[0], adr[1])].get('rmse_3d', float('nan')),
                           'nis_mean': adp_sweep[s][(adr[0], adr[1])].get('nis_mean', float('nan')),
                           'theta': (adr[0], adr[1])},
        }
    return out


def plot_rmse_bar_8way_style(cond_table, subset_seqs, out_path, subset_label):
    """Single-panel grouped bar chart: 4 methods (baseline / adapter / DR /
    adapter+DR) shown together per sequence. Same x-axis as eval_8way_v4."""
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs: return
    methods = ['baseline', 'dr', 'adapter', 'adapter_dr']
    fig, ax = plt.subplots(figsize=(max(8.0, 1.0*len(seqs)+2.0), 4.0))
    n_m = len(methods)
    bw = 0.8 / n_m
    x = np.arange(len(seqs))
    all_rms = []
    for i, k in enumerate(methods):
        meta = METHOD_META_8WAY[k]
        vals = [cond_table[s][k]['rmse_h'] for s in seqs]
        offset = (i - (n_m-1)/2) * bw
        bars = ax.bar(x + offset, vals, bw*0.9,
                      color=meta['color'], hatch=meta['hatch'],
                      edgecolor='white', linewidth=0.5, label=meta['label'])
        for bar, v in zip(bars, vals):
            if v < 1000:
                ax.text(bar.get_x()+bar.get_width()/2, v*1.02,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6.5,
                        rotation=0)
                all_rms.append(v)
    ax.set_ylabel('horizontal RMSE [m]')
    ax.set_xticks(x)
    ax.set_xticklabels([_short(s) for s in seqs])
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3, lw=0.4)
    if all_rms:
        ax.set_ylim(0, max(all_rms) * 1.18)
    ax.set_title(f'i2Nav {VARIANT_LABEL} — horizontal RMSE per sequence ({subset_label})',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 7 — per-sequence trajectory (full overlay of 4 conditions)
# ------------------------------------------------------------------ #
def plot_trajectory_per_seq(seq, traj_root, out_path):
    """Full per-seq N-E overlay with all 4 conditions if NPZs are available."""
    files = {
        'baseline':    os.path.join(traj_root, f'{seq}_baseline.npz'),
        'adapter':     os.path.join(traj_root, f'{seq}_adapter_muv.npz'),
        'dr':          os.path.join(traj_root, f'{seq}_dr.npz'),
        'adapter_dr':  os.path.join(traj_root, f'{seq}_adapter_dr.npz'),
    }
    if not any(os.path.isfile(p) for p in files.values()):
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    gt_plotted = False
    for cond in ('baseline', 'adapter', 'dr', 'adapter_dr'):
        p = files[cond]
        if not os.path.isfile(p): continue
        try:
            d = np.load(p, allow_pickle=True)
            est = np.asarray(d.get('est_ned')); gt = np.asarray(d.get('gt_ned'))
        except Exception:
            continue
        if not gt_plotted and gt is not None:
            ax.plot(gt[:, 1], gt[:, 0], 'k--', lw=1.6, alpha=0.9,
                    label='Ground truth', zorder=10)
            ax.scatter([gt[0, 1]], [gt[0, 0]], c='limegreen', marker='o', s=70,
                       zorder=20, edgecolors='black', linewidths=0.6, label='Start')
            ax.scatter([gt[-1, 1]], [gt[-1, 0]], c='red', marker='s', s=70,
                       zorder=20, edgecolors='black', linewidths=0.6, label='End')
            gt_plotted = True
        ax.plot(est[:, 1], est[:, 0],
                color=COND_COLORS[cond], lw=0.9, alpha=0.85,
                label=COND_LABELS[cond])
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(f'{seq} — N-E trajectory', fontsize=10)
    ax.legend(loc='best', fontsize=7, framealpha=0.92)
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 8 — per-sequence error-over-time (symlog horizontal error)
# ------------------------------------------------------------------ #
def plot_error_over_time(seq, traj_root, out_path):
    files = {
        'baseline':    os.path.join(traj_root, f'{seq}_baseline.npz'),
        'adapter':     os.path.join(traj_root, f'{seq}_adapter_muv.npz'),
        'dr':          os.path.join(traj_root, f'{seq}_dr.npz'),
        'adapter_dr':  os.path.join(traj_root, f'{seq}_adapter_dr.npz'),
    }
    if not any(os.path.isfile(p) for p in files.values()):
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    has_data = False
    for cond in ('baseline', 'adapter', 'dr', 'adapter_dr'):
        p = files[cond]
        if not os.path.isfile(p): continue
        try:
            d = np.load(p, allow_pickle=True)
            est = np.asarray(d.get('est_ned')); gt = np.asarray(d.get('gt_ned'))
            t = np.asarray(d.get('t'))
        except Exception:
            continue
        if est is None or gt is None or t is None: continue
        err = est - gt
        err_h = np.sqrt(err[:, 0]**2 + err[:, 1]**2)
        ax.plot(t - t[0], err_h, color=COND_COLORS[cond], lw=0.7, alpha=0.85,
                label=COND_LABELS[cond])
        has_data = True
    if not has_data:
        plt.close(fig); return
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Horizontal error (m)')
    ax.set_title(f'{seq} — horizontal error over time', fontsize=10)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.92, ncol=2)
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Plot 9 — numeric summary table image
# ------------------------------------------------------------------ #
def plot_summary_table(cond_table, subset_seqs, out_path, subset_label):
    seqs = [s for s in subset_seqs if s in cond_table]
    if not seqs:
        return
    rows = [_short(s) for s in seqs] + ['MEAN']
    cols = ['Baseline', 'Base+DR', 'Adapter', 'Adp+DR',
            'Δdr (%)', 'Δadp (%)', 'Δapdr (%)', 'Base NIS', 'Adp+DR NIS']
    data = []
    bh, ah, dh, adh = [], [], [], []
    for s in seqs:
        b = cond_table[s]['baseline']['rmse_h']
        d = cond_table[s]['dr']['rmse_h']
        a = cond_table[s]['adapter']['rmse_h']
        ad = cond_table[s]['adapter_dr']['rmse_h']
        bn = cond_table[s]['baseline'].get('nis_mean', float('nan'))
        an = cond_table[s]['adapter_dr'].get('nis_mean', float('nan'))
        bh.append(b); ah.append(a); dh.append(d); adh.append(ad)
        p_d  = 100*(b-d)/b if b > 0 else 0
        p_a  = 100*(b-a)/b if b > 0 else 0
        p_ad = 100*(b-ad)/b if b > 0 else 0
        data.append([f'{b:.2f}', f'{d:.2f}', f'{a:.2f}', f'{ad:.2f}',
                     f'{p_d:+.1f}', f'{p_a:+.1f}', f'{p_ad:+.1f}',
                     f'{bn:.2f}' if not math.isnan(bn) else '—',
                     f'{an:.2f}' if not math.isnan(an) else '—'])
    bm = float(np.mean(bh)); am = float(np.mean(ah))
    dm = float(np.mean(dh)); apm = float(np.mean(adh))
    p_d_m  = 100*(bm-dm)/bm if bm > 0 else 0
    p_a_m  = 100*(bm-am)/bm if bm > 0 else 0
    p_ad_m = 100*(bm-apm)/bm if bm > 0 else 0
    data.append([f'{bm:.2f}', f'{dm:.2f}', f'{am:.2f}', f'{apm:.2f}',
                 f'{p_d_m:+.1f}', f'{p_a_m:+.1f}', f'{p_ad_m:+.1f}', '', ''])
    fig_h = max(2.6, 0.32 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis('off')
    cell_colors = [['white']*len(cols) for _ in range(len(rows))]
    for i in range(len(seqs)):
        for col_idx in (4, 5, 6):  # Δdr, Δadp, Δapdr columns
            t = data[i][col_idx]
            if t.startswith('+'):
                cell_colors[i][col_idx] = '#d4edda'
            elif t.startswith('-'):
                cell_colors[i][col_idx] = '#f8d7da'
    table = ax.table(cellText=data, rowLabels=rows, colLabels=cols,
                     cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.05, 1.5)
    for j in range(len(cols)):
        table[(0, j)].set_text_props(fontweight='bold')
        table[(len(rows), j)].set_text_props(fontweight='bold')
    ax.set_title(f'Per-sequence numeric summary — {subset_label}', pad=10)
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def run_one_subset(base_sweep, adp_sweep, traj_root, out_dir, seqs, subset_label):
    '''Generate the full plot+JSON suite for one sequence subset
    (either the 7 well-behaved seqs or all 8 including parking00).
    Best θ is chosen using the *given subset* so the picked theta is
    appropriate for the chosen aggregate.'''
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n=== {subset_label}: subset = {seqs} ===')

    # Heatmaps depend on which seqs are aggregated, so they go in the subset dir
    plot_dr_heatmap_aggregate_for(base_sweep, adp_sweep, seqs,
                                   os.path.join(out_dir, 'dr_heatmap_aggregate.pdf'),
                                   subset_label)
    plot_dr_heatmap_per_seq(base_sweep, adp_sweep,
                             os.path.join(out_dir, 'dr_heatmap_per_seq.pdf'))

    # All headline tables / per-seq plots use per-seq best-θ. Each seq
    # picks its own optimal θ from the 8x8 grid (matches eval_8way_v4 UWB
    # convention). Single-θ-for-all is intentionally not used.
    cond_table = build_psb_cond_table(base_sweep, adp_sweep, seqs)
    base_theta = None   # per-seq θ, no single value
    adp_theta  = None

    plot_headline_per_seq(cond_table, seqs,
                          os.path.join(out_dir, 'headline_per_seq.pdf'),
                          base_theta, adp_theta, subset_label + ' [per-seq best θ]')
    # `improvement_summary` is intentionally omitted — its 2-panel layout
    # (mean Δ% with ±1σ whiskers + a DR-marginal callout) was hard to read
    # and duplicated `improvement_averaged.pdf` + `improvement_per_seq.pdf`.
    plot_nis_calibration(cond_table, seqs,
                          os.path.join(out_dir, 'nis_calibration.pdf'),
                          subset_label)
    plot_trajectory(traj_root,
                    os.path.join(out_dir, 'trajectory_overlay.pdf'))

    # Per-seq trajectory + error-over-time for any seq that has NPZs.
    # Only the ALL_SEQS list — graceful skip if NPZ missing for that seq.
    for seq in ALL_SEQS:
        plot_trajectory_per_seq(seq, traj_root,
                                os.path.join(out_dir, f'trajectory_{seq}.pdf'))
        plot_error_over_time(seq, traj_root,
                             os.path.join(out_dir, f'error_over_time_{seq}.pdf'))

    plot_summary_table(cond_table, seqs,
                       os.path.join(out_dir, 'summary_table.pdf'),
                       subset_label)

    # All headline plots use per-seq best-θ aggregation (each seq picks its
    # own optimal θ from the 8×8 grid — matches eval_8way_v4 UWB convention).
    # Single-θ-for-all is intentionally NOT plotted: it forces every seq to
    # use one operating point, which masks DR's per-environment value and
    # creates misleading "DR is broken" pictures.
    cond_table_psb = build_psb_cond_table(base_sweep, adp_sweep, seqs)
    if cond_table_psb:
        plot_improvement_averaged(cond_table_psb, seqs,
                                  os.path.join(out_dir, 'improvement_averaged.pdf'),
                                  subset_label)
        plot_improvement_per_seq(cond_table_psb, seqs,
                                 os.path.join(out_dir, 'improvement_per_seq.pdf'),
                                 subset_label)
        plot_rmse_bar_8way_style(cond_table_psb, seqs,
                                 os.path.join(out_dir, 'rmse_bar.pdf'),
                                 subset_label)
        plot_summary_table(cond_table_psb, seqs,
                           os.path.join(out_dir, 'summary_table.pdf'),
                           subset_label)

    # JSON + console table aggregated *over the chosen subset*
    summary = {
        'subset_label': subset_label,
        'subset_seqs': list(seqs),
        'best_theta_baseline': base_theta,
        'best_theta_adapter_dr': adp_theta,
        'cond_table_at_best_theta': {
            s: {c: cond_table[s][c]
                for c in ('baseline', 'adapter', 'dr', 'adapter_dr')}
            for s in cond_table
        },
    }
    # Aggregate means (mean-of-RMSE), matching eval_8way_v4 convention.
    # Reports delta_pct using mean — sensitive to per-seq divergence.
    def mn(c):
        vals = [cond_table[s][c]['rmse_h'] for s in seqs if s in cond_table]
        return float(np.mean(vals)) if vals else None
    base_mn = mn('baseline'); adp_mn = mn('adapter')
    dr_mn = mn('dr'); apdr_mn = mn('adapter_dr')
    summary['mean_h_rmse_subset'] = {
        'baseline': base_mn, 'adapter': adp_mn,
        'dr': dr_mn, 'adapter_dr': apdr_mn,
    }
    if base_mn:
        summary['delta_pct'] = {
            'adapter':       100 * (base_mn - adp_mn) / base_mn,
            'dr':            100 * (base_mn - dr_mn) / base_mn,
            'adapter_dr':    100 * (base_mn - apdr_mn) / base_mn,
            'dr_marginal':   100 * (adp_mn - apdr_mn) / adp_mn,
        }
    out_json = os.path.join(out_dir, 'summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  wrote {out_json}')

    print()
    print(f'=== {subset_label} — at best theta (per-sequence h-RMSE [m]) ===')
    print(f'{"seq":15s} {"baseline":>9s} {"adapter":>9s} {"DR":>9s} {"adp+DR":>9s} | '
          f'{"%adp":>7s} {"%apDR":>7s} {"%marg":>7s}')
    print('-' * 92)
    for s in ALL_SEQS:
        if s not in cond_table:
            continue
        in_subset = '*' if s in seqs else ' '
        b = cond_table[s]['baseline']['rmse_h']
        a = cond_table[s]['adapter']['rmse_h']
        d = cond_table[s]['dr']['rmse_h']
        ad = cond_table[s]['adapter_dr']['rmse_h']
        p_adp = 100*(b-a)/b; p_apdr = 100*(b-ad)/b; p_marg = 100*(a-ad)/a
        print(f'{in_subset}{s:14s} {b:>9.2f} {a:>9.2f} {d:>9.2f} {ad:>9.2f} | '
              f'{p_adp:>+6.1f}% {p_apdr:>+6.1f}% {p_marg:>+6.1f}%')
    print('  (* = in aggregation subset)')

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='results_gnss/i2nav/phase11/v3_a')
    ap.add_argument('--out_dir',   default='results_gnss/i2nav/phase11/v3_a/figures')
    ap.add_argument('--variant_label', default=None,
                    help='Display name for plot titles (e.g., "v6_c", "v5_d"). '
                         'Auto-detected from data_root last component if not set.')
    args = ap.parse_args()
    # Auto-detect from data_root if not provided
    if args.variant_label is None:
        args.variant_label = os.path.basename(args.data_root.rstrip('/'))
    # Make available to all plot helpers via module global
    global VARIANT_LABEL
    VARIANT_LABEL = args.variant_label
    print(f'plot variant label: {VARIANT_LABEL}')

    base_sweep = load_sweep(os.path.join(args.data_root, 'dr_sweep_baseline'))
    adp_sweep  = load_sweep(os.path.join(args.data_root, 'dr_sweep_adapter_muv'))
    n_base = sum(len(v) for v in base_sweep.values())
    n_adp  = sum(len(v) for v in adp_sweep.values())
    print(f'baseline DR sweep: {n_base} configs across {sum(1 for v in base_sweep.values() if v)} seqs')
    print(f'adapter+DR sweep:  {n_adp} configs across {sum(1 for v in adp_sweep.values() if v)} seqs')

    traj_root = os.path.join(args.data_root, 'traj')

    # Run twice: once excluding parking00, once including all 8 seqs.
    # Each subdir contains the full plot suite + summary.json computed
    # over its own aggregation subset (best-theta picked per subset, so
    # the headline numbers reflect the chosen scope honestly).
    summaries = {}
    summaries['excl_parking00'] = run_one_subset(
        base_sweep, adp_sweep, traj_root,
        os.path.join(args.out_dir, 'excl_parking00'),
        NORMAL_SEQS,
        '7 seqs (parking00 excluded)')
    summaries['all_seqs'] = run_one_subset(
        base_sweep, adp_sweep, traj_root,
        os.path.join(args.out_dir, 'all_seqs'),
        ALL_SEQS,
        'all 8 seqs (including parking00)')

    # Top-level combined summary
    combined_path = os.path.join(args.out_dir, 'summary_both_subsets.json')
    with open(combined_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f'\nwrote combined: {combined_path}')

    # Side-by-side comparison print
    print()
    print('=' * 80)
    print('Subset-level comparison: excl_parking00  vs  all_seqs')
    print('=' * 80)
    for key in ('adapter', 'dr', 'adapter_dr', 'dr_marginal'):
        a = summaries['excl_parking00'].get('delta_pct', {}).get(key)
        b = summaries['all_seqs'].get('delta_pct', {}).get(key)
        if a is not None and b is not None:
            print(f'  Δ {key:14s}  excl_parking00 = {a:+7.2f}%   all_seqs = {b:+7.2f}%')


if __name__ == '__main__':
    main()
