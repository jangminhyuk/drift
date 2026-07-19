'''
    Sweep + adaptive estimation runner for DR-ESKF-TAC.

    Runs two filter types across discovered datasets:
      1. Baseline ESKF
      2. DR-ESKF-TAC with fixed (theta_w, theta_v) grid

    Output directory structure (mirrors sweep.py):
        <output_dir>/
            <const_name>/
                <ds_name>/
                    baseline_eskf.npz
                    baseline_eskf.nis.csv
                    dr_eskf_tac_tw1e-01_tv1e-01.npz
                    dr_eskf_tac_tw1e-01_tv1e-01.nis.csv
                    ...
            sweep_summary.json

    Usage examples:
        # Default grids on const3
        python main_adaptive.py

        # Specific constellation and trials
        python main_adaptive.py --consts 1 2 --trials 1 2 3 --tdoa 2

        # Custom theta grids
        python main_adaptive.py --theta_w 0.01 0.1 1.0 --theta_v 0.01 0.1

        # Single dataset mode (bypass dataset discovery)
        python main_adaptive.py -i anchor.npz data.csv
'''
#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import time
import json
import math
import numpy as np
from itertools import product

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cwd)
sys.path.insert(0, os.path.join(cwd, '..'))

from main import run_estimation


# --------------------------------------------------------------------------- #
#  Paths (same as sweep.py)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(os.path.join(cwd, '..', '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'flight-dataset')
CSV_DIR = os.path.join(DATASET_DIR, 'csv-data')
SURVEY_DIR = os.path.join(DATASET_DIR, 'survey-results')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'adaptive_sweep')


# --------------------------------------------------------------------------- #
#  Dataset discovery (same logic as sweep.py)
# --------------------------------------------------------------------------- #
def get_const_name(ds_name):
    '''Extract constellation name, e.g. "const3-trial1-tdoa2" -> "const3".'''
    return ds_name.split('-')[0]


def discover_datasets(consts, trials, tdoa, csv_dir=CSV_DIR, survey_dir=SURVEY_DIR):
    '''Discover matching (anchor_npz, csv_file, dataset_name) tuples.'''
    datasets = []
    for c in consts:
        anchor_npz = os.path.join(survey_dir, 'anchor_const%d.npz' % c)
        if not os.path.isfile(anchor_npz):
            print('WARNING: anchor file not found: %s' % anchor_npz)
            continue
        const_dir = os.path.join(csv_dir, 'const%d' % c)
        if not os.path.isdir(const_dir):
            print('WARNING: const dir not found: %s' % const_dir)
            continue
        for tr in trials:
            pattern = os.path.join(const_dir, 'const%d-trial%d-tdoa%d*.csv' % (c, tr, tdoa))
            matches = sorted(glob.glob(pattern))
            for csv_file in matches:
                name = os.path.splitext(os.path.basename(csv_file))[0]
                datasets.append((anchor_npz, csv_file, name))
    return datasets


# --------------------------------------------------------------------------- #
#  NIS CSV helper (same as sweep.py)
# --------------------------------------------------------------------------- #
def _save_nis_csv(npz_file, nis_log, t):
    '''Save NIS log as csv alongside an npz file.'''
    if not nis_log:
        return
    nis_file = npz_file.replace('.npz', '.nis.csv')
    rows = []
    for idx, nv, dof in nis_log:
        rows.append([float(t[int(idx)]), float(nv), int(dof)])
    np.savetxt(nis_file, np.array(rows),
               delimiter=',', header='timestamp,nis_value,dof', comments='')


# --------------------------------------------------------------------------- #
#  Result saving helper
# --------------------------------------------------------------------------- #
def _save_result_npz(filepath, result, extra_fields=None):
    '''Save estimation result as .npz with standard fields.'''
    fields = dict(
        t=result['t'],
        Xpo=result['Xpo'],
        Ppo=result['Ppo'],
        q_list=result['q_list'],
        t_vicon=result['t_vicon'],
        pos_vicon=result['pos_vicon'],
        anchor_position=result['anchor_position'],
        pos_error=result['pos_error'],
        rms_x=result['rms_x'],
        rms_y=result['rms_y'],
        rms_z=result['rms_z'],
        rms_all=result['rms_all'],
    )
    if extra_fields:
        fields.update(extra_fields)
    np.savez(filepath, **fields)


# --------------------------------------------------------------------------- #
#  Sweep runner
# --------------------------------------------------------------------------- #
def run_adaptive_sweep(datasets, theta_w_list, theta_v_list,
                       dr_solver, output_dir, verbose=True):
    '''
    Run baseline ESKF and DR-ESKF-TAC grid
    across all datasets. Saves results to output_dir.
    '''
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'theta_w_list': theta_w_list,
        'theta_v_list': theta_v_list,
        'dr_solver': dr_solver,
        'datasets': [],
    }

    n_tac = len(theta_w_list) * len(theta_v_list)
    total_configs = 1 + n_tac  # baseline + TAC grid
    total_runs = len(datasets) * total_configs
    run_idx = 0

    for ds_idx, (anchor_npz, csv_file, ds_name) in enumerate(datasets):
        const_name = get_const_name(ds_name)
        ds_output_dir = os.path.join(output_dir, const_name, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)

        ds_summary = {
            'dataset_name': ds_name,
            'const_name': const_name,
            'anchor_npz': anchor_npz,
            'csv_file': csv_file,
            'baseline': {},
            'dr_eskf_tac': {},
        }

        # ---- 1. Baseline ESKF ----
        run_idx += 1
        print('\n[%d/%d] Baseline ESKF on %s' % (run_idx, total_runs, ds_name))
        t_start = time.time()
        result = run_estimation(
            anchor_npz, csv_file,
            filter_type='eskf', verbose=verbose
        )
        elapsed = time.time() - t_start
        print('  -> RMS=%.4f [m]  (%.1fs)' % (result['rms_all'], elapsed))

        baseline_file = os.path.join(ds_output_dir, 'baseline_eskf.npz')
        _save_result_npz(baseline_file, result)
        _save_nis_csv(baseline_file, result['nis_log'], result['t'])

        ds_summary['baseline'] = {
            'rms_x': result['rms_x'],
            'rms_y': result['rms_y'],
            'rms_z': result['rms_z'],
            'rms_all': result['rms_all'],
            'file': baseline_file,
        }

        # ---- 2. DR-ESKF-TAC grid (theta_w x theta_v) ----
        for tw, tv in product(theta_w_list, theta_v_list):
            run_idx += 1
            config_name = 'dr_eskf_tac_tw%.0e_tv%.0e' % (tw, tv)
            print('\n[%d/%d] DR-ESKF-TAC (tw=%.1e, tv=%.1e) on %s' %
                  (run_idx, total_runs, tw, tv, ds_name))
            t_start = time.time()
            result = run_estimation(
                anchor_npz, csv_file,
                filter_type='dr_eskf_tac',
                theta_w=tw, theta_v=tv,
                dr_solver=dr_solver,
                verbose=verbose
            )
            elapsed = time.time() - t_start
            print('  -> RMS=%.4f [m]  (%.1fs)  DR_solves=%d  fallbacks=%d' %
                  (result['rms_all'], elapsed,
                   result['dr_solve_count'], result['dr_fallback_count']))

            dr_file = os.path.join(ds_output_dir, config_name + '.npz')
            _save_result_npz(dr_file, result, extra_fields=dict(
                theta_w=tw, theta_v=tv, dr_solver=dr_solver,
                dr_solve_count=result['dr_solve_count'],
                dr_fallback_count=result['dr_fallback_count']))
            _save_nis_csv(dr_file, result['nis_log'], result['t'])

            key = 'tw%.0e_tv%.0e' % (tw, tv)
            ds_summary['dr_eskf_tac'][key] = {
                'theta_w': tw,
                'theta_v': tv,
                'rms_x': result['rms_x'],
                'rms_y': result['rms_y'],
                'rms_z': result['rms_z'],
                'rms_all': result['rms_all'],
                'dr_solve_count': result['dr_solve_count'],
                'dr_fallback_count': result['dr_fallback_count'],
                'file': dr_file,
            }

        summary['datasets'].append(ds_summary)

    # Save JSON summary
    summary_file = os.path.join(output_dir, 'sweep_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== Sweep complete. Summary saved to %s ===' % summary_file)

    return summary


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sweep baseline ESKF + fixed-theta DR-ESKF-TAC')

    # Dataset discovery (sweep mode)
    parser.add_argument('--consts', type=int, nargs='+', default=[3],
                        help='Anchor constellations to test (default: [3])')
    parser.add_argument('--trials', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Trial numbers to test (default: [1,2,3,4,5])')
    parser.add_argument('--tdoa', type=int, default=2, choices=[2, 3],
                        help='TDoA mode (default: 2)')

    # Single dataset mode (overrides discovery)
    parser.add_argument('-i', action='store', nargs=2, default=None,
                        help='Single dataset mode: anchor_npz csv_file')

    # Fixed-theta grid
    parser.add_argument('--theta_w', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 1.0, 5.0],
                        help='TAC: BW ball radii on Q_w for grid sweep')
    parser.add_argument('--theta_v', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 1.0, 5.0],
                        help='TAC: BW ball radii on Sigma_v for grid sweep')

    parser.add_argument('--dr_solver', choices=['fw', 'fw_exact'], default='fw_exact',
                        help='DR-MMSE solver (default: fw_exact)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results (default: results/adaptive_sweep)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-step verbose output')
    args = parser.parse_args()

    # Resolve datasets
    if args.i is not None:
        # Single dataset mode
        anchor_npz, csv_file = args.i
        ds_name = os.path.splitext(os.path.basename(csv_file))[0]
        datasets = [(anchor_npz, csv_file, ds_name)]
    else:
        # Discovery mode
        datasets = discover_datasets(args.consts, args.trials, args.tdoa)
        if not datasets:
            print('ERROR: No datasets found for consts=%s, trials=%s, tdoa=%d' %
                  (args.consts, args.trials, args.tdoa))
            sys.exit(1)

    n_tac = len(args.theta_w) * len(args.theta_v)
    n_configs = 1 + n_tac  # baseline + TAC grid

    print('=== Sweep Configuration ===')
    print('  Datasets:  %d' % len(datasets))
    for _, _, name in datasets:
        print('    - %s' % name)
    print('  TAC theta_w: %s' % args.theta_w)
    print('  TAC theta_v: %s' % args.theta_v)
    print('  DR solver:   %s' % args.dr_solver)
    print('  Output:      %s' % args.output_dir)
    print('  Configs per dataset: 1 baseline + %d TAC = %d' %
          (n_tac, n_configs))
    print('  Total runs: %d datasets x %d configs = %d' %
          (len(datasets), n_configs, len(datasets) * n_configs))
    print('====================================\n')

    run_adaptive_sweep(
        datasets=datasets,
        theta_w_list=args.theta_w,
        theta_v_list=args.theta_v,
        dr_solver=args.dr_solver,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
