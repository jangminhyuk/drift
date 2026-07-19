"""Candidate publication figures for DRiFt on i2Nav (v6_c, muv-only deploy).

Generates several distinct candidates so we can pick:
  A. aggregate RMSE decomposition (KF-GINS / DR-only / Adapter-only / DRiFt)
  B. per-sequence dumbbell (baseline -> adapter -> DRiFt), shows consistency
  C. horizontal position error over time for a showcase sequence (3 methods)

Data: results_gnss/i2nav/phase12/v6_c/{figures/summary_both_subsets.json, traj/*.npz}
"""
import os, argparse, json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Paper style (2026-07-09): DejaVu Sans + the FIG_REGIMES_UWB palette,
# shared by all paper figures (see plot_8way.py / fig_drift_guideline_v5_uwb.py)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['axes.edgecolor'] = '#c3c2b7'
rcParams['xtick.color'] = '#52514e'
rcParams['ytick.color'] = '#52514e'
rcParams['grid.color'] = '#e1e0d9'

ROOT = Path(os.environ.get('DRIFT_GNSS_DATA', str(Path(__file__).resolve().parents[3] / 'data/precomputed/gnss')))
TRAJ = ROOT / 'traj'
OUT = ROOT / 'figures' / 'pub_v1'

# semantic palette: baseline gray / DR blue / adapter red / DRiFt violet
C_BASE = '#898781'
C_DR = '#2a78d6'
C_ADAP = '#e34948'
C_DRIFT = '#4a3aa7'

SEQS = ['building00', 'building01', 'building02', 'playground00',
        'street00', 'street01', 'street02']


def load_summary():
    d = json.load(open(ROOT / 'figures' / 'summary_both_subsets.json'))
    return d['excl_parking00']


def herr(npz):
    z = np.load(npz)
    return z['t'], np.linalg.norm(z['err_ned'][:, :2], axis=1), z['gt_ned'], z['est_ned']


def roll(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


# ---------------------------------------------------------------- A
def fig_agg_bars(S):
    m = S['mean_h_rmse_subset']
    dp = S['delta_pct']
    keys = ['baseline', 'dr', 'adapter', 'adapter_dr']
    labels = ['KF-GINS', 'DR only', 'Adapter only', 'DRiFt']
    colors = [C_BASE, C_DR, C_ADAP, C_DRIFT]
    vals = [m[k] for k in keys]
    impr = [0.0, dp['dr'], dp['adapter'], dp['adapter_dr']]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x = np.arange(4)
    bars = ax.bar(x, vals, width=0.66, color=colors, edgecolor='black', lw=0.6)
    for i, (b, v, p) in enumerate(zip(bars, vals, impr)):
        lab = f'{v:.2f} m' if i == 0 else f'{v:.2f} m\n(−{p:.1f}%)'
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, lab, ha='center',
                va='bottom', fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13.5)
    ax.set_ylabel('Mean horizontal RMSE [m]', fontsize=14)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.25)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, 'A_aggregate_rmse')


# ---------------------------------------------------------------- B
def fig_dumbbell(S):
    ct = S['cond_table_at_best_theta']
    rows = []
    for s in SEQS:
        rows.append((s, ct[s]['baseline']['rmse_h'], ct[s]['adapter']['rmse_h'],
                     ct[s]['adapter_dr']['rmse_h']))
    rows.sort(key=lambda r: r[1])  # by baseline ascending
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, (s, base, adap, drift) in enumerate(rows):
        ax.plot([drift, base], [i, i], '-', color='#cccccc', lw=2.2, zorder=1)
        ax.scatter(base, i, s=95, color=C_BASE, zorder=3, edgecolor='black', lw=0.5)
        ax.scatter(adap, i, s=70, color=C_ADAP, zorder=3, edgecolor='black', lw=0.5)
        ax.scatter(drift, i, s=95, color=C_DRIFT, zorder=4, edgecolor='black', lw=0.5)
        red = 100 * (base - drift) / base
        ax.text(base * 1.06, i, f'−{red:.0f}%', va='center', ha='left',
                fontsize=10.5, color='#444444')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 16)
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(['0.5', '1', '2', '5', '10'])
    ax.set_xlabel('Horizontal RMSE [m]  (log scale)', fontsize=13.5)
    ax.tick_params(axis='x', labelsize=12)
    # legend
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='o', color='w', mfc=C_BASE, mec='k', ms=10, label='KF-GINS'),
         Line2D([0], [0], marker='o', color='w', mfc=C_ADAP, mec='k', ms=9, label='Adapter only'),
         Line2D([0], [0], marker='o', color='w', mfc=C_DRIFT, mec='k', ms=10, label='DRiFt')]
    ax.legend(handles=h, loc='lower right', fontsize=12, frameon=True,
              framealpha=0.92, edgecolor='0.7', columnspacing=1.1, handlelength=1.6)
    ax.grid(axis='x', alpha=0.22, which='both')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, 'B_per_sequence_dumbbell')


# ---------------------------------------------------------------- C
def fig_error_time(seq, smooth_s=4.0):
    tb, eb, _, _ = herr(TRAJ / f'{seq}_baseline.npz')
    ta, ea, _, _ = herr(TRAJ / f'{seq}_adapter_muv.npz')
    td, ed, _, _ = herr(TRAJ / f'{seq}_adapter_dr.npz')
    t0 = tb[0]
    fs = 1.0 / np.median(np.diff(tb))
    w = max(1, int(smooth_s * fs))
    rb, ra, rd = roll(eb, w), roll(ea, w), roll(ed, w)
    ymax = max(rb.max(), ra.max(), rd.max())
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(tb - t0, rb, color=C_BASE, lw=1.6, label='KF-GINS', zorder=2)
    ax.plot(ta - t0, ra, color=C_ADAP, lw=1.6, label='Adapter only', zorder=3)
    ax.plot(td - t0, rd, color=C_DRIFT, lw=1.9, label='DRiFt', zorder=4)
    ax.set_xlabel('Time [s]', fontsize=14.5)
    ax.set_ylabel('Horizontal position error [m]', fontsize=14.5)
    ax.set_xlim(0, (tb - t0)[-1])
    ax.set_ylim(0, ymax * 1.28)
    ax.tick_params(labelsize=12.5)
    ax.legend(loc='upper center', ncol=3, fontsize=13.5, framealpha=0.92,
              edgecolor='0.7', columnspacing=1.1, handlelength=1.6)
    ax.grid(alpha=0.22)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, f'C_error_time_{seq}')


# ---------------------------------------------------------------- D
def fig_trajectory(seq):
    """Top-down NED trajectory: GT vs KF-GINS vs DRiFt. Dramatic only when the
    baseline divergence is large vs the path extent (e.g. parking00)."""
    b = np.load(TRAJ / f'{seq}_baseline.npz')
    d = np.load(TRAJ / f'{seq}_adapter_dr.npz')
    gt, eb, ed = b['gt_ned'], b['est_ned'], d['est_ned']
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(eb[:, 1], eb[:, 0], color=C_BASE, lw=1.3, alpha=0.9,
            label='KF-GINS', zorder=2)
    ax.plot(ed[:, 1], ed[:, 0], color=C_DRIFT, lw=1.6, alpha=0.95,
            label='DRiFt', zorder=3)
    ax.plot(gt[:, 1], gt[:, 0], color='#0b0b0b', lw=2.6, label='Ground truth',
            zorder=4)
    ax.scatter([gt[0, 1]], [gt[0, 0]], s=70, marker='o', color='white',
               edgecolor='black', lw=1.3, zorder=5)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('East [m]', fontsize=14.5)
    ax.set_ylabel('North [m]', fontsize=14.5)
    ax.tick_params(labelsize=12.5)
    ax.legend(loc='upper right', fontsize=13.5, framealpha=0.92,
              edgecolor='0.7', handlelength=1.6)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    _save(fig, f'D_trajectory_{seq}')


# ---------------------------------------------------------------- G
def fig_tail_dumbbell():
    """Per-sequence 95th-percentile (worst-case) horizontal error, baseline ->
    adapter -> DRiFt. Honestly shows building01's tail REGRESSION."""
    def p95(s, m):
        z = np.load(TRAJ / f'{s}_{m}.npz')
        return float(np.quantile(np.linalg.norm(z['err_ned'][:, :2], axis=1), 0.95))
    rows = [(s, p95(s, 'baseline'), p95(s, 'adapter_muv'), p95(s, 'adapter_dr'))
            for s in SEQS]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    from matplotlib.lines import Line2D
    for i, (s, base, adap, drift) in enumerate(rows):
        worse = drift > base
        ax.plot([drift, base], [i, i], '-',
                color=('#f4cccc' if worse else '#cccccc'), lw=2.4, zorder=1)
        ax.scatter(base, i, s=95, color=C_BASE, zorder=3, edgecolor='black', lw=0.5)
        ax.scatter(adap, i, s=70, color=C_ADAP, zorder=3, edgecolor='black', lw=0.5)
        ax.scatter(drift, i, s=95, color=C_DRIFT, zorder=4, edgecolor='black', lw=0.5)
        red = 100 * (base - drift) / base
        ax.text(max(base, drift) * 1.06, i, f'{"+" if red < 0 else "−"}{abs(red):.0f}%',
                va='center', ha='left', fontsize=10.5,
                color=('#c0392b' if red < 0 else '#444444'))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(0.7, 40)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(['1', '2', '5', '10', '20'])
    ax.set_xlabel('95th-percentile horizontal error [m]  (log scale)', fontsize=13.5)
    ax.tick_params(axis='x', labelsize=12)
    h = [Line2D([0], [0], marker='o', color='w', mfc=C_BASE, mec='k', ms=10, label='KF-GINS'),
         Line2D([0], [0], marker='o', color='w', mfc=C_ADAP, mec='k', ms=9, label='Adapter only'),
         Line2D([0], [0], marker='o', color='w', mfc=C_DRIFT, mec='k', ms=10, label='DRiFt')]
    ax.legend(handles=h, loc='lower right', fontsize=12, frameon=True,
              framealpha=0.92, edgecolor='0.7', handlelength=1.6)
    ax.grid(axis='x', alpha=0.22, which='both')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, 'G_per_seq_tail_p95')


# ---------------------------------------------------------------- E
def fig_per_seq_error_time(S, smooth_s=4.0):
    """Small multiples: horizontal error over time, one panel per sequence."""
    ct = S['cond_table_at_best_theta']
    import matplotlib.gridspec as gs
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 11.5))
    axes = axes.ravel()
    for k, s in enumerate(SEQS):
        ax = axes[k]
        tb, eb, _, _ = herr(TRAJ / f'{s}_baseline.npz')
        ta, ea, _, _ = herr(TRAJ / f'{s}_adapter_muv.npz')
        td, ed, _, _ = herr(TRAJ / f'{s}_adapter_dr.npz')
        t0 = tb[0]
        fs = 1.0 / np.median(np.diff(tb))
        w = max(1, int(smooth_s * fs))
        rb, ra, rd = roll(eb, w), roll(ea, w), roll(ed, w)
        ax.plot(tb - t0, rb, color=C_BASE, lw=1.4, zorder=2)
        ax.plot(ta - t0, ra, color=C_ADAP, lw=1.4, zorder=3)
        ax.plot(td - t0, rd, color=C_DRIFT, lw=1.7, zorder=4)
        red = 100 * (ct[s]['baseline']['rmse_h'] - ct[s]['adapter_dr']['rmse_h']) \
            / ct[s]['baseline']['rmse_h']
        ax.set_title(f'{s}   (DRiFt $-${red:.0f}%)', fontsize=12.5)
        ax.set_xlim(0, (tb - t0)[-1])
        ax.set_ylim(0, max(rb.max(), ra.max(), rd.max()) * 1.1)
        ax.tick_params(labelsize=10.5)
        ax.grid(alpha=0.2)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    # legend in the 8th (unused) cell
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_BASE, lw=2.4, label='KF-GINS'),
         Line2D([0], [0], color=C_ADAP, lw=2.4, label='Adapter only'),
         Line2D([0], [0], color=C_DRIFT, lw=2.6, label='DRiFt')]
    axes[7].axis('off')
    axes[7].legend(handles=h, loc='center', fontsize=15, framealpha=0.92,
                   edgecolor='0.7', handlelength=1.8)
    fig.supxlabel('Time [s]', fontsize=14)
    fig.supylabel('Horizontal position error [m]', fontsize=14)
    fig.tight_layout()
    _save(fig, 'E_per_seq_error_time')


# ---------------------------------------------------------------- F
def fig_per_seq_improvement(S):
    """Grouped bars: per-sequence % RMSE reduction vs KF-GINS for each lever."""
    ct = S['cond_table_at_best_theta']
    seqs = sorted(SEQS, key=lambda s: 100 * (ct[s]['baseline']['rmse_h']
                  - ct[s]['adapter_dr']['rmse_h']) / ct[s]['baseline']['rmse_h'])
    def imp(s, m):
        b = ct[s]['baseline']['rmse_h']
        return 100 * (b - ct[s][m]['rmse_h']) / b
    adap = [imp(s, 'adapter') for s in seqs]
    dr = [imp(s, 'dr') for s in seqs]
    drift = [imp(s, 'adapter_dr') for s in seqs]
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    x = np.arange(len(seqs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    # light fill + solid colored edge, like FIG_REGIMES_UWB panel (b)
    for off, vals, c in [(-w, dr, C_DR), (0, adap, C_ADAP), (w, drift, C_DRIFT)]:
        ax.bar(x + off, vals, w, color=to_rgba(c, 0.35), edgecolor=c, lw=1.6)
    ax.axhline(0, color='#0b0b0b', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(seqs, rotation=20, fontsize=14, ha='right')
    ax.set_ylabel('RMSE improvement over KF-GINS [%]', fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    handles = [Patch(fc=to_rgba(C_DR, 0.35), ec=C_DR, lw=1.6, label='DR-only'),
               Patch(fc=to_rgba(C_ADAP, 0.35), ec=C_ADAP, lw=1.6, label='Adapter-only'),
               Patch(fc=to_rgba(C_DRIFT, 0.35), ec=C_DRIFT, lw=1.6, label='DRiFt')]
    ax.legend(handles=handles, loc='upper left', fontsize=16, framealpha=0.92,
              edgecolor='0.7', handlelength=1.6, ncol=3, columnspacing=1.1)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(min(min(dr), min(adap), min(drift)) - 4,
                max(max(dr), max(adap), max(drift)) * 1.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, 'F_per_seq_improvement')


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / f'{name}.pdf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--error_time_seqs', nargs='+',
                    default=['street01', 'playground00', 'building00'])
    args = ap.parse_args()
    S = load_summary()
    fig_agg_bars(S)
    fig_dumbbell(S)
    fig_per_seq_improvement(S)
    fig_per_seq_error_time(S)
    for s in args.error_time_seqs:
        fig_error_time(s)
    # dramatic stress-test (severe GNSS denial; excluded from aggregates)
    fig_trajectory('parking00')
    fig_error_time('parking00')


if __name__ == '__main__':
    main()
