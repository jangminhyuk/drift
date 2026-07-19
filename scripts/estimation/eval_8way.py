'''
Unified 12-method evaluation: runs all filter/predictor combinations on
test datasets and produces a single JSON summary for plotting.

Methods (4 base x 3 DR variants):
  1. ESKF (baseline, no DR)
  2. DR-ESKF-TAC (best fixed (theta_w, theta_v) from grid)
  3. DR-ESKF-CDC (best fixed (theta_x, theta_v) from grid)
  4-6. Mamba predictor + {ESKF, DR-TAC, DR-CDC}
  7-9. GRU predictor + {ESKF, DR-TAC, DR-CDC}
  10-12. Transformer predictor + {ESKF, DR-TAC, DR-CDC}

TAC sweeps theta_w x theta_v (BW ball on process + meas. noise).
CDC sweeps theta_x x theta_v (BW ball on prior state + meas. noise).
The two sweeps are independent and saved as separate NPZ namespaces.

Usage:
    python eval_8way.py \
        --ckpt_mamba results/muse_e2e_v43.pt \
        --ckpt_gru results/muse_e2e_gru.pt \
        --ckpt_transformer results/muse_e2e_transformer.pt \
        --consts 1 2 3 4 --trials 6 \
        --theta_w 0.001 0.01 0.1 1.0 2.0 \
        --theta_x 0.001 0.01 0.1 1.0 2.0 \
        --theta_v 0.001 0.01 0.1 1.0 2.0 \
        --device cuda --output_dir results/eval_8way_v4

Reusing prior runs (skip method if its output NPZ already exists):
    python eval_8way.py ... --output_dir results/eval_8way_v4 --skip_existing
'''
import argparse
import os
import sys
import math
import json
import time
import csv as csvmod
import numpy as np
from itertools import product

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cwd)
sys.path.insert(0, os.path.join(cwd, '..'))

from main import run_estimation
from main_muse_theta import make_muse_callback
from main_adaptive import discover_datasets, _save_result_npz, _save_nis_csv, get_const_name
from classical_adaptive import make_classical_callback

# ------------------------------------------------------------------ #
#  Method registry
#  dr_style: None (no DR), 'tac' (TAC: BW ball on Sigma_w, Sigma_v),
#            or 'cdc' (CDC: BW ball on X_prior, Sigma_v)
# ------------------------------------------------------------------ #
METHODS = [
    {'key': 'eskf',                'label': 'ESKF',           'filter': 'eskf',         'ckpt': None,          'dr_style': None},
    {'key': 'dr_eskf',             'label': 'DR-ESKF (TAC)',  'filter': 'dr_eskf_tac',  'ckpt': None,          'dr_style': 'tac'},
    {'key': 'dr_eskf_cdc',         'label': 'DR-ESKF (CDC)',  'filter': 'dr_eskf_cdc',  'ckpt': None,          'dr_style': 'cdc'},
    {'key': 'mamba_eskf',          'label': 'Mamba+ESKF',     'filter': 'eskf',         'ckpt': 'mamba',       'dr_style': None},
    {'key': 'mamba_dr',            'label': 'Mamba+DR (TAC)', 'filter': 'dr_eskf_tac',  'ckpt': 'mamba',       'dr_style': 'tac'},
    {'key': 'mamba_cdc',           'label': 'Mamba+DR (CDC)', 'filter': 'dr_eskf_cdc',  'ckpt': 'mamba',       'dr_style': 'cdc'},
    {'key': 'gru_eskf',            'label': 'GRU+ESKF',       'filter': 'eskf',         'ckpt': 'gru',         'dr_style': None},
    {'key': 'gru_dr',              'label': 'GRU+DR (TAC)',   'filter': 'dr_eskf_tac',  'ckpt': 'gru',         'dr_style': 'tac'},
    {'key': 'gru_cdc',             'label': 'GRU+DR (CDC)',   'filter': 'dr_eskf_cdc',  'ckpt': 'gru',         'dr_style': 'cdc'},
    {'key': 'transformer_eskf',    'label': 'Xfmr+ESKF',      'filter': 'eskf',         'ckpt': 'transformer', 'dr_style': None},
    {'key': 'transformer_dr',      'label': 'Xfmr+DR (TAC)',  'filter': 'dr_eskf_tac',  'ckpt': 'transformer', 'dr_style': 'tac'},
    {'key': 'transformer_cdc',     'label': 'Xfmr+DR (CDC)',  'filter': 'dr_eskf_cdc',  'ckpt': 'transformer', 'dr_style': 'cdc'},
    # Classical (statistical) adaptive baselines — activated via --classical.
    # TAC variants only (noise-centric ambiguity).
    {'key': 'sh_eskf',             'label': 'SageHusa+ESKF',     'filter': 'eskf',         'ckpt': None, 'dr_style': None,  'classical': 'sage_husa'},
    {'key': 'sh_dr',               'label': 'SageHusa+DR (TAC)', 'filter': 'dr_eskf_tac',  'ckpt': None, 'dr_style': 'tac', 'classical': 'sage_husa'},
    {'key': 'vb_eskf',             'label': 'VB-AKF+ESKF',       'filter': 'eskf',         'ckpt': None, 'dr_style': None,  'classical': 'vbakf'},
    {'key': 'vb_dr',               'label': 'VB-AKF+DR (TAC)',   'filter': 'dr_eskf_tac',  'ckpt': None, 'dr_style': 'tac', 'classical': 'vbakf'},
]

# Canonical display order for the 12 methods
METHOD_ORDER = [m['key'] for m in METHODS]


def _grid_key(dr_style, t1, t2):
    '''Build the JSON grid key. TAC uses (theta_w, theta_v); CDC uses (theta_x, theta_v).'''
    if dr_style == 'cdc':
        return 'tx%.3g_tv%.3g' % (t1, t2)
    return 'tw%.3g_tv%.3g' % (t1, t2)


def _npz_basename(mkey, dr_style, t1, t2):
    '''Build the per-config NPZ filename.
    For the no-predictor TAC method we keep the legacy "dr_eskf_tw*_tv*.npz" name
    so v2 outputs can be reused as-is. CDC always carries the explicit "_cdc"
    namespace and "tx_*" grid encoding.'''
    if dr_style == 'cdc':
        return '%s_tx%.3g_tv%.3g.npz' % (mkey, t1, t2)
    if mkey == 'dr_eskf':
        return 'dr_eskf_tw%.3g_tv%.3g.npz' % (t1, t2)
    return '%s_tw%.3g_tv%.3g.npz' % (mkey, t1, t2)


def _sanitize_json(obj):
    '''Replace inf/nan with string markers for JSON serialization.'''
    if isinstance(obj, float):
        if math.isinf(obj):
            return 'inf'
        if math.isnan(obj):
            return 'nan'
    elif isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _save_predictions_csv(path, adapter):
    '''Save predictor's prediction log to CSV.'''
    if not adapter.prediction_log:
        return
    fieldnames = ['step', 'mu_v', 'R', 'warmup']
    sample_row = adapter.prediction_log[-1]
    if 'Q_acc_diag' in sample_row:
        fieldnames = ['step', 'mu_v', 'R',
                      'Q_acc_diag', 'Q_gyro_diag',
                      'mu_w_acc', 'mu_w_gyro', 'warmup']
    with open(path, 'w', newline='') as f:
        writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in adapter.prediction_log:
            writer.writerow(row)


def _run_single(anchor_npz, csv_file, filter_type, theta_w=0.01, theta_v=0.01,
                dr_solver='fw_exact', verbose=False, theta_callback=None):
    '''Run estimation with error handling; returns result dict or None on divergence.'''
    try:
        result = run_estimation(
            anchor_npz, csv_file,
            filter_type=filter_type,
            theta_w=theta_w, theta_v=theta_v,
            dr_solver=dr_solver,
            verbose=verbose,
            theta_callback=theta_callback,
        )
        return result
    except (ValueError, FloatingPointError, RuntimeError) as e:
        print('    DIVERGED: %s' % e)
        return None


def _result_to_summary(result):
    '''Extract summary fields from a run_estimation result dict.'''
    if result is None:
        return {'rms_x': float('inf'), 'rms_y': float('inf'),
                'rms_z': float('inf'), 'rms_all': float('inf'),
                'diverged': True}
    return {
        'rms_x': result['rms_x'],
        'rms_y': result['rms_y'],
        'rms_z': result['rms_z'],
        'rms_all': result['rms_all'],
        'diverged': False,
    }


# ------------------------------------------------------------------ #
#  Main evaluation loop
# ------------------------------------------------------------------ #
def run_8way(datasets, checkpoints, theta_w_list, theta_v_list,
             ema_alpha, dr_solver, device, output_dir, verbose=False,
             diagnostics=False, theta_x_list=None, only_styles=None,
             skip_existing=False, seed_summary_path=None,
             classical_methods=None, only_classical=False,
             only_ckpts=None, mu_v_only=False, summary_only=False,
             save_best_only=False):
    '''
    Run the 12-method evaluation across datasets.

    TAC sweep: theta_w_list x theta_v_list   (dr_style='tac' methods).
    CDC sweep: theta_x_list x theta_v_list   (dr_style='cdc' methods).
    Baseline / non-DR predictor methods run a single config.

    Args:
        only_styles: optional iterable in {None, 'tac', 'cdc'}; if given,
            only methods with dr_style in this set are executed.
            Use only_styles={'cdc'} to extend an existing v2 directory with
            just the CDC variants.
        skip_existing: if True, skip a method/config when its NPZ already
            exists in output_dir. Lets you resume an interrupted sweep.
        seed_summary_path: optional path to a previous eval_8way_summary.json
            whose entries are merged into this run's summary as a starting
            point. Useful for adding CDC results on top of a v2 baseline.
    '''
    os.makedirs(output_dir, exist_ok=True)
    if theta_x_list is None:
        theta_x_list = list(theta_w_list)

    # Determine which methods to run
    classical_set = set(classical_methods or [])
    active_methods = []
    for m in METHODS:
        if only_styles is not None and m['dr_style'] not in only_styles:
            continue
        if m.get('classical'):
            if m['classical'] in classical_set:
                active_methods.append(m)
            continue
        if only_classical:
            continue
        if only_ckpts is not None and m['ckpt'] not in only_ckpts:
            # Restrict to specific predictor families (this also skips the
            # ckpt-None baseline methods — reuse those from a prior eval dir).
            continue
        if m['ckpt'] is None:
            active_methods.append(m)
        elif checkpoints.get(m['ckpt']) is not None:
            active_methods.append(m)
    active_keys = [m['key'] for m in active_methods]

    # Count total runs for progress
    n_tac = len(theta_w_list) * len(theta_v_list)
    n_cdc = len(theta_x_list) * len(theta_v_list)
    runs_per_ds = 0
    for m in active_methods:
        if m['dr_style'] == 'tac':
            runs_per_ds += n_tac
        elif m['dr_style'] == 'cdc':
            runs_per_ds += n_cdc
        else:
            runs_per_ds += 1
    total_runs = len(datasets) * runs_per_ds
    run_idx = 0

    print('=== 12-Way Evaluation ===')
    print('  Datasets:  %d' % len(datasets))
    print('  Methods:   %s' % ', '.join(active_keys))
    print('  TAC grid:  %d theta_w x %d theta_v = %d pairs' %
          (len(theta_w_list), len(theta_v_list), n_tac))
    print('  CDC grid:  %d theta_x x %d theta_v = %d pairs' %
          (len(theta_x_list), len(theta_v_list), n_cdc))
    if only_styles is not None:
        print('  Only styles: %s' % sorted([s for s in only_styles if s] + (['none'] if None in only_styles else [])))
    if skip_existing:
        print('  Resume mode: skipping configs whose NPZ already exists')
    if seed_summary_path:
        print('  Seeding summary from: %s' % seed_summary_path)
    print('  Total runs: %d' % total_runs)
    print('=========================\n')

    summary = {
        'methods_run': active_keys,
        'checkpoints': {k: v for k, v in checkpoints.items() if v},
        'classical_methods': sorted(classical_set),
        'theta_w_list': theta_w_list,
        'theta_x_list': theta_x_list,
        'theta_v_list': theta_v_list,
        'mu_v_only': bool(mu_v_only),
        'datasets': [],
    }
    if mu_v_only:
        print('  ADAPTER DEPLOYMENT: mu_v-ONLY (nominal R/Q/mu_w; '
              'GNSS-v6_c-style deployment)\n')

    # Load seed summary so we can merge prior method entries (e.g., from v2)
    seed_by_ds = {}
    if seed_summary_path and os.path.isfile(seed_summary_path):
        try:
            with open(seed_summary_path) as f:
                seed_summary = json.load(f)
            for ds in seed_summary.get('datasets', []):
                seed_by_ds[ds['dataset_name']] = ds
            # Merge methods_run union, but final methods_run is recomputed below
            # to reflect what is actually present.
            print('  Loaded seed: %d datasets, %d methods'
                  % (len(seed_by_ds), len(seed_summary.get('methods_run', []))))
        except Exception as e:
            print('  WARNING: failed to load seed summary: %s' % e)

    for anchor_npz, csv_file, ds_name in datasets:
        const_name = get_const_name(ds_name)
        ds_dir = os.path.join(output_dir, const_name, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        ds_entry = {
            'dataset_name': ds_name,
            'const_name': const_name,
            'methods': {},
        }

        # Seed any pre-existing method entries from a prior run (e.g., v2)
        seed_methods = seed_by_ds.get(ds_name, {}).get('methods', {})
        for k, v in seed_methods.items():
            if k not in active_keys:
                # Carry forward methods we are NOT re-running, so the final
                # summary contains all 12 methods even when --only_styles is set.
                ds_entry['methods'][k] = v

        for m in active_methods:
            mkey = m['key']
            mlabel = m['label']
            mfilter = m['filter']
            dr_style = m['dr_style']
            ckpt_path = checkpoints.get(m['ckpt']) if m['ckpt'] else None

            # -------- Non-DR methods: single config -------- #
            if dr_style is None:
                run_idx += 1
                print('[%d/%d] %s on %s' % (run_idx, total_runs, mlabel, ds_name))
                npz_path = os.path.join(ds_dir, '%s.npz' % mkey)
                pred_csv_path = os.path.join(ds_dir, '%s_predictions.csv' % mkey)

                if skip_existing and os.path.isfile(npz_path):
                    seed_s = seed_methods.get(mkey)
                    if seed_s and seed_s.get('file') == npz_path:
                        ds_entry['methods'][mkey] = seed_s
                        print('  SKIP (reused from seed): %s' % npz_path)
                    else:
                        # File present but no seed entry — mark as available
                        ds_entry['methods'][mkey] = {
                            'rms_x': float('nan'), 'rms_y': float('nan'),
                            'rms_z': float('nan'), 'rms_all': float('nan'),
                            'diverged': False, 'file': npz_path,
                            'predictions_csv': pred_csv_path if os.path.isfile(pred_csv_path) else '',
                        }
                        print('  SKIP: %s exists but not in seed' % npz_path)
                    continue

                cb, adapter = None, None
                if m.get('classical'):
                    cb, adapter, _ = make_classical_callback(
                        m['classical'], return_theta=False)
                elif ckpt_path is not None:
                    cb, adapter, _ = make_muse_callback(
                        ckpt_path, device=device, ema_alpha=ema_alpha,
                        diagnostics=diagnostics, return_theta=False,
                        mu_v_only=mu_v_only)

                t0 = time.time()
                result = _run_single(anchor_npz, csv_file, mfilter,
                                     verbose=verbose, theta_callback=cb)
                s = _result_to_summary(result)
                if result is not None:
                    _save_result_npz(npz_path, result)
                    _save_nis_csv(npz_path, result['nis_log'], result['t'])
                    s['file'] = npz_path
                    if adapter is not None:
                        _save_predictions_csv(pred_csv_path, adapter)
                        s['predictions_csv'] = pred_csv_path
                    else:
                        s['predictions_csv'] = ''
                else:
                    s['file'] = ''
                    s['predictions_csv'] = ''
                ds_entry['methods'][mkey] = s
                print('  RMS=%.4f (%.1fs)' % (s['rms_all'], time.time() - t0))
                continue

            # -------- DR methods: 2-D theta grid sweep -------- #
            if dr_style == 'tac':
                t1_list, t2_list = theta_w_list, theta_v_list
                t1_name, t2_name = 'theta_w', 'theta_v'
                t1_short, t2_short = 'tw', 'tv'
            elif dr_style == 'cdc':
                t1_list, t2_list = theta_x_list, theta_v_list
                t1_name, t2_name = 'theta_x', 'theta_v'
                t1_short, t2_short = 'tx', 'tv'
            else:
                raise ValueError('Unknown dr_style: %s' % dr_style)

            best_rms = float('inf')
            best_t1, best_t2 = t1_list[0], t2_list[0]
            best_s = None
            best_pred_csv = ''
            best_result = None       # save_best_only: best config's trajectory
            best_npz_path = None
            best_adapter = None
            grid_results = {}

            for t1, t2 in product(t1_list, t2_list):
                run_idx += 1
                gkey = _grid_key(dr_style, t1, t2)
                print('[%d/%d] %s %s=%.3g %s=%.3g on %s' %
                      (run_idx, total_runs, mlabel, t1_name, t1, t2_name, t2, ds_name))

                npz_name = _npz_basename(mkey, dr_style, t1, t2)
                npz_path = os.path.join(ds_dir, npz_name)
                pred_csv_path = os.path.join(
                    ds_dir, npz_name.replace('.npz', '_predictions.csv'))

                cb, adapter = None, None
                if m.get('classical'):
                    cb, adapter, _ = make_classical_callback(
                        m['classical'], theta_w_fixed=t1, theta_v_fixed=t2,
                        return_theta=True)
                elif ckpt_path is not None:
                    cb, adapter, _ = make_muse_callback(
                        ckpt_path, device=device, ema_alpha=ema_alpha,
                        theta_w_fixed=t1, theta_v_fixed=t2,
                        diagnostics=diagnostics, return_theta=True,
                        mu_v_only=mu_v_only)

                if skip_existing and os.path.isfile(npz_path):
                    seed_grid = seed_methods.get(mkey, {}).get('grid', {})
                    seed_s = seed_grid.get(gkey)
                    if seed_s and seed_s.get('file') == npz_path:
                        s = seed_s
                        print('  SKIP (reused from seed)')
                    else:
                        s = {
                            'rms_x': float('nan'), 'rms_y': float('nan'),
                            'rms_z': float('nan'), 'rms_all': float('nan'),
                            'diverged': False, 'file': npz_path,
                            t1_name: t1, t2_name: t2,
                        }
                        print('  SKIP: NPZ present but no seed entry')
                else:
                    t0 = time.time()
                    # run_estimation expects theta_w, theta_v keyword names; for CDC
                    # main.py forwards them as theta_x, theta_v internally.
                    result = _run_single(anchor_npz, csv_file, mfilter,
                                         theta_w=t1, theta_v=t2,
                                         dr_solver=dr_solver, verbose=verbose,
                                         theta_callback=cb)
                    s = _result_to_summary(result)
                    s[t1_name] = t1
                    s[t2_name] = t2
                    if result is not None:
                        s['dr_solve_count'] = result['dr_solve_count']
                        s['dr_fallback_count'] = result['dr_fallback_count']
                        if summary_only or save_best_only:
                            # Don't save per-config artifacts during the sweep.
                            # summary_only keeps only RMSE; save_best_only defers
                            # the trajectory save to the best-theta config (below).
                            s['file'] = ''
                            s['predictions_csv'] = ''
                        else:
                            extra = {t1_name: t1, t2_name: t2, 'dr_solver': dr_solver,
                                     'dr_style': dr_style}
                            _save_result_npz(npz_path, result, extra_fields=extra)
                            _save_nis_csv(npz_path, result['nis_log'], result['t'])
                            s['file'] = npz_path
                            if adapter is not None:
                                _save_predictions_csv(pred_csv_path, adapter)
                                s['predictions_csv'] = pred_csv_path
                            else:
                                s['predictions_csv'] = ''
                    else:
                        s['file'] = ''
                        s['predictions_csv'] = ''
                    print('  RMS=%.4f (%.1fs)' % (s['rms_all'], time.time() - t0))

                grid_results[gkey] = s
                rms_all = s.get('rms_all', float('inf'))
                if isinstance(rms_all, str):
                    rms_all = float('inf')
                if rms_all < best_rms:
                    best_rms = rms_all
                    best_t1, best_t2 = t1, t2
                    best_s = s
                    best_pred_csv = s.get('predictions_csv', '')
                    if save_best_only and not (skip_existing and os.path.isfile(npz_path)):
                        best_result = result
                        best_npz_path = npz_path
                        best_adapter = adapter

            # save_best_only: write ONLY the best-theta config's trajectory.
            if save_best_only and best_result is not None:
                extra = {t1_name: best_t1, t2_name: best_t2,
                         'dr_solver': dr_solver, 'dr_style': dr_style}
                _save_result_npz(best_npz_path, best_result, extra_fields=extra)
                _save_nis_csv(best_npz_path, best_result['nis_log'], best_result['t'])
                best_s['file'] = best_npz_path
                if best_adapter is not None:
                    bpc = best_npz_path.replace('.npz', '_predictions.csv')
                    _save_predictions_csv(bpc, best_adapter)
                    best_s['predictions_csv'] = bpc

            best_t1_key = 'best_' + t1_short
            best_t2_key = 'best_' + t2_short
            ds_entry['methods'][mkey] = {
                best_t1_key: best_t1,
                best_t2_key: best_t2,
                'dr_style': dr_style,
                'rms_x': best_s['rms_x'] if best_s else float('inf'),
                'rms_y': best_s['rms_y'] if best_s else float('inf'),
                'rms_z': best_s['rms_z'] if best_s else float('inf'),
                'rms_all': best_rms,
                'file': best_s['file'] if best_s else '',
                'predictions_csv': best_pred_csv,
                'grid': grid_results,
                'diverged': best_rms == float('inf'),
            }

        summary['datasets'].append(ds_entry)

    # Recompute methods_run to include seeded-only methods that we did not run
    # but want present in the final summary.
    full_methods_run = list(active_keys)
    for ds in summary['datasets']:
        for mk in ds['methods'].keys():
            if mk not in full_methods_run:
                full_methods_run.append(mk)
    # Reorder per METHOD_ORDER for stable display
    summary['methods_run'] = [m for m in METHOD_ORDER if m in full_methods_run]
    summary['methods_run'] += [m for m in full_methods_run if m not in METHOD_ORDER]

    # ------------------------------------------------------------------ #
    #  Print comparison table
    # ------------------------------------------------------------------ #
    _print_table(summary, active_keys)

    # Save JSON
    summary_file = os.path.join(output_dir, 'eval_8way_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(_sanitize_json(summary), f, indent=2)
    print('\nSummary saved to %s' % summary_file)

    return summary


def _print_table(summary, active_keys):
    '''Print a formatted comparison table: rows=methods, cols=constellations.'''
    # Group datasets by constellation
    const_datasets = {}
    for ds in summary['datasets']:
        cn = ds['const_name']
        const_datasets.setdefault(cn, []).append(ds)
    const_names = sorted(const_datasets.keys())

    # Compute per-constellation mean RMSE for each method
    method_labels = {m['key']: m['label'] for m in METHODS}

    print('\n' + '=' * 90)
    print('8-WAY COMPARISON (per-constellation mean RMSE)')
    print('=' * 90)

    header = '%-22s' % 'Method'
    for cn in const_names:
        header += '  %8s' % cn
    header += '  %8s' % 'Mean'
    print(header)
    print('-' * 90)

    for mkey in active_keys:
        label = method_labels.get(mkey, mkey)
        # Add best theta info for DR methods
        first_ds = summary['datasets'][0] if summary['datasets'] else None
        mdata = first_ds['methods'].get(mkey, {}) if first_ds else {}
        # CDC uses theta_x; TAC uses theta_w
        bt1 = mdata.get('best_tx', mdata.get('best_tw'))
        bt2 = mdata.get('best_tv')
        if isinstance(bt1, float) and isinstance(bt2, float):
            label += ' (%.2g,%.2g)' % (bt1, bt2)

        row = '%-22s' % label[:22]
        all_rms = []
        for cn in const_names:
            rms_vals = []
            for ds in const_datasets[cn]:
                mdata = ds['methods'].get(mkey, {})
                rms = mdata.get('rms_all', float('inf'))
                if isinstance(rms, str):
                    rms = float('inf')
                rms_vals.append(rms)
            mean_rms = np.mean(rms_vals) if rms_vals else float('inf')
            all_rms.append(mean_rms)
            if mean_rms < 100:
                row += '  %8.4f' % mean_rms
            else:
                row += '  %8s' % 'div'
        overall = np.mean(all_rms) if all_rms else float('inf')
        if overall < 100:
            row += '  %8.4f' % overall
        else:
            row += '  %8s' % 'div'
        print(row)

    print('=' * 90)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Unified 8-method evaluation: ESKF, DR-ESKF, '
                    '3 predictors x {ESKF, DR-ESKF}')

    parser.add_argument('--ckpt_mamba', type=str, default=None,
                        help='Mamba predictor checkpoint')
    parser.add_argument('--ckpt_gru', type=str, default=None,
                        help='GRU predictor checkpoint')
    parser.add_argument('--ckpt_transformer', type=str, default=None,
                        help='Transformer predictor checkpoint')

    parser.add_argument('--consts', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--trials', type=int, nargs='+', default=[6])
    parser.add_argument('--tdoa', type=int, default=2)

    parser.add_argument('--theta_w', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 1.0, 2.0],
                        help='TAC: BW ball radii on Sigma_w (process noise)')
    parser.add_argument('--theta_x', type=float, nargs='+',
                        default=None,
                        help='CDC: BW ball radii on X_prior (defaults to --theta_w)')
    parser.add_argument('--theta_v', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 1.0, 2.0],
                        help='Shared: BW ball radii on Sigma_v (measurement noise)')
    parser.add_argument('--ema_alpha', type=float, default=0.3)
    parser.add_argument('--dr_solver', choices=['fw', 'fw_exact'],
                        default='fw_exact')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='results/eval_8way')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--only_styles', type=str, nargs='+',
                        choices=['none', 'tac', 'cdc'], default=None,
                        help='Restrict execution to a subset of dr_style values. '
                             'E.g. --only_styles cdc to add CDC results to a v2 dir.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip method/config when its NPZ already exists.')
    parser.add_argument('--seed_summary', type=str, default=None,
                        help='Path to a prior eval_8way_summary.json (e.g. v2). '
                             'Methods present there but not re-run will be carried forward.')
    parser.add_argument('--classical', type=str, nargs='+',
                        choices=['sage_husa', 'vbakf'], default=None,
                        help='Also run classical adaptive baselines '
                             '(Sage-Husa and/or VB-AKF), TAC variants only.')
    parser.add_argument('--only_ckpts', type=str, nargs='+',
                        choices=['mamba', 'gru', 'transformer'], default=None,
                        help='Restrict to specific predictor families; also '
                             'skips the ckpt-None baselines (eskf/dr_eskf). '
                             'Use to add adapter variants on top of an '
                             'existing baseline directory.')
    parser.add_argument('--only_names', type=str, nargs='+', default=None,
                        help='Restrict to datasets whose exact dataset_name is '
                             'in this list (for per-sequence array parallelism).')
    parser.add_argument('--summary_only', action='store_true',
                        help='Do not save per-config trajectory NPZ / NIS / '
                             'prediction CSVs; keep only RMSE in the summary '
                             'JSON. ~1000x less disk for large grid sweeps.')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Run the full grid (all RMSE in the summary) but '
                             'save the trajectory NPZ only for the best-theta '
                             'config of each method (one NPZ per method/seq).')
    parser.add_argument('--mu_v_only', action='store_true',
                        help='Deploy adapters MEAN-ONLY: inject predicted '
                             'mu_v, keep nominal R/Q/mu_w. UWB analogue of '
                             'the GNSS v6_c mu_v-only deployment.')
    parser.add_argument('--only_classical', action='store_true',
                        help='Run ONLY the --classical methods (use with '
                             '--seed_summary to carry forward prior results).')

    args = parser.parse_args()

    datasets = discover_datasets(args.consts, args.trials, args.tdoa)
    if args.only_names:
        keep = set(args.only_names)
        datasets = [d for d in datasets if d[2] in keep]
    if not datasets:
        print('ERROR: No datasets found')
        sys.exit(1)

    checkpoints = {
        'mamba': args.ckpt_mamba,
        'gru': args.ckpt_gru,
        'transformer': args.ckpt_transformer,
    }
    n_ckpts = sum(1 for v in checkpoints.values() if v)
    if n_ckpts == 0:
        print('WARNING: No predictor checkpoints provided. '
              'Running baseline ESKF and fixed DR-ESKF only.')

    only_styles = None
    if args.only_styles:
        only_styles = set(None if s == 'none' else s for s in args.only_styles)

    run_8way(
        datasets=datasets,
        checkpoints=checkpoints,
        theta_w_list=args.theta_w,
        theta_x_list=args.theta_x,
        theta_v_list=args.theta_v,
        ema_alpha=args.ema_alpha,
        dr_solver=args.dr_solver,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        diagnostics=args.diagnostics,
        only_styles=only_styles,
        skip_existing=args.skip_existing,
        seed_summary_path=args.seed_summary,
        classical_methods=args.classical,
        only_classical=args.only_classical,
        only_ckpts=args.only_ckpts,
        mu_v_only=args.mu_v_only,
        summary_only=args.summary_only,
        save_best_only=args.save_best_only,
    )


if __name__ == '__main__':
    main()
