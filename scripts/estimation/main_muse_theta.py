'''
    Comparison entrypoint: ESKF vs DR-ESKF-TAC (fixed theta grid) vs
    DR-ESKF-TAC with MUSE adaptive measurement noise (mu_v + R).

    Patterned after main_adaptive.py: runs all three filter types across
    discovered datasets and saves a summary JSON for comparison.

    Usage:
        # Full comparison on const4
        python main_muse_theta.py --checkpoint results/meas_noise_muse.pt \
            --consts 4 --trials 1 2 3

        # Single dataset
        python main_muse_theta.py --checkpoint results/meas_noise_muse.pt \
            -i dataset/flight-dataset/survey-results/anchor_const4.npz \
               dataset/flight-dataset/csv-data/const4/const4-trial1-tdoa2.csv
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
from main_adaptive import discover_datasets, _save_result_npz, _save_nis_csv, get_const_name
from theta_muse_adapter import MeasNoiseMUSEAdapter
from theta_sweep_worker import get_imu_chunk
from eval_calibration import (
    innovation_whiteness_test, gt_residual_vs_predicted,
    reliability_diagram, nis_per_constellation,
)


def make_muse_callback(checkpoint_path, device='cpu', ema_alpha=0.3,
                       theta_w_fixed=0.01, theta_v_fixed=0.01,
                       diagnostics=False, return_theta=True,
                       mu_v_only=False):
    '''Create a theta_callback for run_estimation using MUSE adapter.

    The callback sets eskf.mu_v, eskf.std_uwb_tdoa, and optionally
    eskf.Q_base and eskf.mu_w directly (side effects), and returns
    fixed (theta_w, theta_v) for the DR margin.

    Args:
        return_theta: if True, callback returns (theta_w, theta_v) for DR.
                      if False, returns None (for predictor + vanilla ESKF mode).
        mu_v_only: if True, inject ONLY the predicted measurement-noise mean
                   mu_v; keep the filter's nominal R / Q / mu_w untouched.
                   UWB analogue of the GNSS v6_c mu_v-only deployment
                   (sweep_dr_theta.py --no_R). Default False = legacy
                   behavior (inject mu_v + R + Q + mu_w).

    Returns (callback_fn, adapter, diag_log).
    '''
    adapter = MeasNoiseMUSEAdapter(
        checkpoint_path, device=device, ema_alpha=ema_alpha,
        theta_w_fixed=theta_w_fixed, theta_v_fixed=theta_v_fixed)
    L_imu = adapter.cfg.L_imu
    state = {'prev_uwb_t': 0.0}
    diag_log = []  # extended diagnostics log

    def callback(eskf, k, t_uwb_now, imu_raw, t_imu_raw):
        imu_chunk = get_imu_chunk(t_imu_raw, imu_raw,
                                  state['prev_uwb_t'], t_uwb_now, L_imu)
        mu_v, R, Q, mu_w = adapter.step(eskf, k, imu_chunk, t_uwb_now)
        state['prev_uwb_t'] = t_uwb_now

        # Apply measurement noise predictions
        eskf.mu_v = mu_v
        if not mu_v_only:
            eskf.std_uwb_tdoa = math.sqrt(max(R, 1e-6))

            # Apply process noise predictions (Q_base overrides default Qi)
            if hasattr(eskf, 'set_Q_base'):
                eskf.set_Q_base(Q)
            else:
                eskf.Q_base = Q

            # Apply process bias
            eskf.mu_w = mu_w

        # Record diagnostics if enabled
        if diagnostics:
            diag = eskf._last_dr_diagnostics or {}
            diag_log.append({
                'step': k,
                't': float(t_uwb_now),
                'mu_v': float(mu_v),
                'R': float(R),
                'err_uwb': float(diag.get('err_uwb', 0.0)),
                'S_star': float(diag.get('S_star', eskf.std_uwb_tdoa**2)),
            })

        if return_theta:
            return adapter.theta_w_fixed, adapter.theta_v_fixed
        return None

    return callback, adapter, diag_log


def run_comparison(datasets, theta_w_list, theta_v_list,
                   checkpoint_path, theta_w_fixed_list, theta_v_fixed_list,
                   ema_alpha, dr_solver, device, output_dir, verbose=True,
                   diagnostics=False, predictor_eskf=False):
    '''
    Run all three filter types across datasets and produce comparison.
      1. Baseline ESKF
      2. DR-ESKF-TAC with fixed theta grid
      3. DR-ESKF-TAC with MUSE adaptive mu_v + R (swept over theta grid)
    '''
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'theta_w_list': theta_w_list,
        'theta_v_list': theta_v_list,
        'muse_checkpoint': checkpoint_path,
        'muse_theta_w_fixed_list': theta_w_fixed_list,
        'muse_theta_v_fixed_list': theta_v_fixed_list,
        'muse_ema_alpha': ema_alpha,
        'dr_solver': dr_solver,
        'datasets': [],
    }

    n_tac = len(theta_w_list) * len(theta_v_list)
    n_muse = len(theta_w_fixed_list) * len(theta_v_fixed_list)
    n_pred_eskf = 1 if predictor_eskf else 0
    total_configs = 1 + n_tac + n_muse + n_pred_eskf
    total_runs = len(datasets) * total_configs
    run_idx = 0

    for ds_idx, (anchor_npz, csv_file, ds_name) in enumerate(datasets):
        const_name = get_const_name(ds_name)
        ds_output_dir = os.path.join(output_dir, const_name, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)

        ds_summary = {
            'dataset_name': ds_name,
            'const_name': const_name,
            'baseline': {},
            'dr_eskf_tac': {},
            'dr_eskf_tac_adaptive': {},
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

        # ---- 3. MUSE adaptive mu_v + R (sweep theta grid) ----
        for tw_f, tv_f in product(theta_w_fixed_list, theta_v_fixed_list):
            run_idx += 1
            config_name = 'dr_eskf_tac_muse_tw%.0e_tv%.0e' % (tw_f, tv_f)
            print('\n[%d/%d] MUSE mu_v+R (tw=%.1e, tv=%.1e) on %s' %
                  (run_idx, total_runs, tw_f, tv_f, ds_name))

            muse_cb, muse_adapter, diag_log = make_muse_callback(
                checkpoint_path, device=device, ema_alpha=ema_alpha,
                theta_w_fixed=tw_f, theta_v_fixed=tv_f,
                diagnostics=diagnostics)

            t_start = time.time()
            try:
                result = run_estimation(
                    anchor_npz, csv_file,
                    filter_type='dr_eskf_tac',
                    theta_w=tw_f, theta_v=tv_f,
                    dr_solver=dr_solver,
                    verbose=verbose,
                    theta_callback=muse_cb,
                )
                elapsed = time.time() - t_start
                diverged = False
            except (ValueError, FloatingPointError, RuntimeError) as e:
                elapsed = time.time() - t_start
                print('  -> DIVERGED (%.1fs): %s' % (elapsed, e))
                diverged = True

            mu_v_final = muse_adapter.mu_v
            R_final = muse_adapter.R
            muse_updates = muse_adapter._step_count

            if diverged:
                result = {
                    'rms_all': float('inf'), 'rms_x': float('inf'),
                    'rms_y': float('inf'), 'rms_z': float('inf'),
                    'dr_solve_count': 0, 'dr_fallback_count': 0,
                    'nis_log': [], 't': np.array([]),
                }

            print('  -> RMS=%.4f [m]  (%.1fs)  DR_solves=%d  fallbacks=%d  MUSE_updates=%d' %
                  (result['rms_all'], elapsed,
                   result['dr_solve_count'], result['dr_fallback_count'],
                   muse_updates))
            if mu_v_final is not None and not diverged:
                print('     Final mu_v=%.4f  R=%.4f  std=%.4f' %
                      (mu_v_final, R_final, math.sqrt(R_final)))

            if not diverged:
                muse_file = os.path.join(ds_output_dir, config_name + '.npz')
                _save_result_npz(muse_file, result, extra_fields=dict(
                    theta_w_fixed=tw_f, theta_v_fixed=tv_f,
                    mu_v_final=mu_v_final or 0.0,
                    R_final=R_final or 0.0,
                    dr_solver=dr_solver,
                    dr_solve_count=result['dr_solve_count'],
                    dr_fallback_count=result['dr_fallback_count'],
                    muse_updates=muse_updates))
                _save_nis_csv(muse_file, result['nis_log'], result['t'])

                # Save prediction trajectory CSV
                pred_csv = os.path.join(ds_output_dir, config_name + '_predictions.csv')
                if muse_adapter.prediction_log:
                    fieldnames = ['step', 'mu_v', 'R', 'warmup']
                    sample_row = muse_adapter.prediction_log[-1]
                    if 'Q_acc_diag' in sample_row:
                        fieldnames = ['step', 'mu_v', 'R',
                                      'Q_acc_diag', 'Q_gyro_diag',
                                      'mu_w_acc', 'mu_w_gyro', 'warmup']
                    with open(pred_csv, 'w', newline='') as f:
                        csv_writer = csvmod.DictWriter(
                            f, fieldnames=fieldnames, extrasaction='ignore')
                        csv_writer.writeheader()
                        for row in muse_adapter.prediction_log:
                            csv_writer.writerow(row)

                # Save diagnostics data if enabled
                if diagnostics and diag_log:
                    diag_dir = os.path.join(ds_output_dir, 'diagnostics')
                    os.makedirs(diag_dir, exist_ok=True)

                    err_uwb = np.array([d['err_uwb'] for d in diag_log])
                    S_star = np.array([d['S_star'] for d in diag_log])
                    mu_v_arr = np.array([d['mu_v'] for d in diag_log])
                    R_arr = np.array([d['R'] for d in diag_log])

                    # Innovation whiteness test
                    whiteness = innovation_whiteness_test(err_uwb, S_star, mu_v_arr)

                    # GT residual vs predicted (use err_uwb as GT residual proxy)
                    gt_vs_pred = gt_residual_vs_predicted(
                        err_uwb, mu_v_arr, R_arr, window=50)

                    # Reliability diagram
                    rel_diag = reliability_diagram(R_arr, err_uwb, mu_v_arr)

                    # Save all diagnostic arrays
                    np.savez(os.path.join(diag_dir, config_name + '_diag.npz'),
                             err_uwb=err_uwb, S_star=S_star,
                             mu_v=mu_v_arr, R=R_arr,
                             acf_lags=whiteness['lags'],
                             acf_vals=whiteness['acf_vals'],
                             acf_bound=whiteness['confidence_bound'],
                             ks_stat=whiteness['ks_statistic'],
                             ks_pval=whiteness['ks_pvalue'],
                             lb_Q=whiteness['ljung_box_Q'],
                             lb_pval=whiteness['ljung_box_pvalue'],
                             rel_bin_centers=rel_diag['bin_centers'],
                             rel_empirical_var=rel_diag['empirical_var'],
                             rel_bin_counts=rel_diag['bin_counts'])
            else:
                muse_file = ''
                pred_csv = ''

            key = 'muse_tw%.0e_tv%.0e' % (tw_f, tv_f)
            ds_summary['dr_eskf_tac_adaptive'][key] = {
                'theta_w_fixed': tw_f,
                'theta_v_fixed': tv_f,
                'mu_v_final': mu_v_final,
                'R_final': R_final,
                'rms_x': result['rms_x'],
                'rms_y': result['rms_y'],
                'rms_z': result['rms_z'],
                'rms_all': result['rms_all'],
                'dr_solve_count': result['dr_solve_count'],
                'dr_fallback_count': result['dr_fallback_count'],
                'muse_updates': muse_updates,
                'diverged': diverged,
                'file': muse_file,
                'predictions_csv': pred_csv,
            }

        # ---- 4. Predictor + vanilla ESKF (no DR) ----
        if predictor_eskf:
            if 'predictor_eskf' not in ds_summary:
                ds_summary['predictor_eskf'] = {}
            run_idx += 1
            print('\n[%d/%d] Predictor + ESKF (no DR) on %s' %
                  (run_idx, total_runs, ds_name))

            eskf_cb, eskf_adapter, eskf_diag_log = make_muse_callback(
                checkpoint_path, device=device, ema_alpha=ema_alpha,
                theta_w_fixed=0.0, theta_v_fixed=0.0,
                diagnostics=diagnostics, return_theta=False)

            t_start = time.time()
            try:
                result = run_estimation(
                    anchor_npz, csv_file,
                    filter_type='eskf',
                    verbose=verbose,
                    theta_callback=eskf_cb,
                )
                elapsed = time.time() - t_start
                diverged = False
            except (ValueError, FloatingPointError, RuntimeError) as e:
                elapsed = time.time() - t_start
                print('  -> DIVERGED (%.1fs): %s' % (elapsed, e))
                diverged = True

            if diverged:
                result = {
                    'rms_all': float('inf'), 'rms_x': float('inf'),
                    'rms_y': float('inf'), 'rms_z': float('inf'),
                    'dr_solve_count': 0, 'dr_fallback_count': 0,
                    'nis_log': [], 't': np.array([]),
                }

            mu_v_final = eskf_adapter.mu_v
            R_final = eskf_adapter.R
            muse_updates = eskf_adapter._step_count

            print('  -> RMS=%.4f [m]  (%.1fs)  MUSE_updates=%d' %
                  (result['rms_all'], elapsed, muse_updates))
            if mu_v_final is not None and not diverged:
                print('     Final mu_v=%.4f  R=%.4f  std=%.4f' %
                      (mu_v_final, R_final, math.sqrt(R_final)))

            if not diverged:
                eskf_file = os.path.join(ds_output_dir, 'predictor_eskf.npz')
                _save_result_npz(eskf_file, result, extra_fields=dict(
                    mu_v_final=mu_v_final or 0.0,
                    R_final=R_final or 0.0,
                    muse_updates=muse_updates))
                _save_nis_csv(eskf_file, result['nis_log'], result['t'])

                # Save prediction trajectory CSV
                pred_csv = os.path.join(ds_output_dir, 'predictor_eskf_predictions.csv')
                if eskf_adapter.prediction_log:
                    fieldnames = ['step', 'mu_v', 'R', 'warmup']
                    sample_row = eskf_adapter.prediction_log[-1]
                    if 'Q_acc_diag' in sample_row:
                        fieldnames = ['step', 'mu_v', 'R',
                                      'Q_acc_diag', 'Q_gyro_diag',
                                      'mu_w_acc', 'mu_w_gyro', 'warmup']
                    with open(pred_csv, 'w', newline='') as f:
                        csv_writer = csvmod.DictWriter(
                            f, fieldnames=fieldnames, extrasaction='ignore')
                        csv_writer.writeheader()
                        for row in eskf_adapter.prediction_log:
                            csv_writer.writerow(row)
            else:
                eskf_file = ''
                pred_csv = ''

            ds_summary['predictor_eskf'] = {
                'mu_v_final': mu_v_final,
                'R_final': R_final,
                'rms_x': result['rms_x'],
                'rms_y': result['rms_y'],
                'rms_z': result['rms_z'],
                'rms_all': result['rms_all'],
                'muse_updates': muse_updates,
                'diverged': diverged,
                'file': eskf_file,
                'predictions_csv': pred_csv,
            }

        summary['datasets'].append(ds_summary)

    # ---- Print comparison table ----
    print('\n' + '='*80)
    print('COMPARISON SUMMARY')
    print('='*80)
    print('%-40s  %8s  %8s  %8s' % ('Config', 'RMS_all', 'RMS_xy', 'RMS_z'))
    print('-'*80)

    for ds in summary['datasets']:
        name = ds['dataset_name']
        print('\n%s:' % name)

        # Baseline
        b = ds['baseline']
        rms_xy = math.sqrt(b['rms_x']**2 + b['rms_y']**2)
        print('  %-38s  %8.4f  %8.4f  %8.4f' %
              ('ESKF (baseline)', b['rms_all'], rms_xy, b['rms_z']))

        # Best fixed theta
        best_key = None
        best_rms = float('inf')
        for key, val in ds['dr_eskf_tac'].items():
            if val['rms_all'] < best_rms:
                best_rms = val['rms_all']
                best_key = key

        # All fixed thetas
        for key in sorted(ds['dr_eskf_tac'].keys()):
            val = ds['dr_eskf_tac'][key]
            rms_xy = math.sqrt(val['rms_x']**2 + val['rms_y']**2)
            marker = ' *' if key == best_key else ''
            print('  %-38s  %8.4f  %8.4f  %8.4f%s' %
                  ('DR-ESKF tw=%s tv=%s' % (val['theta_w'], val['theta_v']),
                   val['rms_all'], rms_xy, val['rms_z'], marker))

        # Best MUSE theta
        best_muse_key = None
        best_muse_rms = float('inf')
        for key, val in ds['dr_eskf_tac_adaptive'].items():
            if val['rms_all'] < best_muse_rms:
                best_muse_rms = val['rms_all']
                best_muse_key = key

        # All MUSE configs
        for key in sorted(ds['dr_eskf_tac_adaptive'].keys()):
            m = ds['dr_eskf_tac_adaptive'][key]
            rms_xy = math.sqrt(m['rms_x']**2 + m['rms_y']**2)
            marker = ' *' if key == best_muse_key else ''
            print('  %-38s  %8.4f  %8.4f  %8.4f%s' %
                  ('MUSE+DR tw=%s tv=%s' % (m['theta_w_fixed'], m['theta_v_fixed']),
                   m['rms_all'], rms_xy, m['rms_z'], marker))

        # Predictor + ESKF (no DR)
        if 'predictor_eskf' in ds and ds['predictor_eskf']:
            pe = ds['predictor_eskf']
            rms_xy = math.sqrt(pe['rms_x']**2 + pe['rms_y']**2)
            print('  %-38s  %8.4f  %8.4f  %8.4f' %
                  ('Predictor + ESKF (no DR)', pe['rms_all'], rms_xy, pe['rms_z']))

    # ---- Aggregate stats ----
    print('\n' + '='*80)
    print('AGGREGATE (mean over %d datasets)' % len(summary['datasets']))
    print('='*80)

    baseline_rms = [d['baseline']['rms_all'] for d in summary['datasets']]

    best_fixed_rms = []
    for d in summary['datasets']:
        best = min(v['rms_all'] for v in d['dr_eskf_tac'].values())
        best_fixed_rms.append(best)

    best_muse_rms = []
    for d in summary['datasets']:
        best = min(v['rms_all'] for v in d['dr_eskf_tac_adaptive'].values())
        best_muse_rms.append(best)

    print('  ESKF baseline:          mean=%.4f  std=%.4f' %
          (np.mean(baseline_rms), np.std(baseline_rms)))
    print('  DR-ESKF best fixed:     mean=%.4f  std=%.4f' %
          (np.mean(best_fixed_rms), np.std(best_fixed_rms)))
    print('  DR-ESKF + MUSE best:    mean=%.4f  std=%.4f' %
          (np.mean(best_muse_rms), np.std(best_muse_rms)))

    # Predictor + ESKF aggregate
    pred_eskf_rms = [d['predictor_eskf']['rms_all']
                     for d in summary['datasets']
                     if 'predictor_eskf' in d and d['predictor_eskf']]
    if pred_eskf_rms:
        print('  Predictor + ESKF:       mean=%.4f  std=%.4f' %
              (np.mean(pred_eskf_rms), np.std(pred_eskf_rms)))

    # Save JSON summary (replace inf/nan with string markers for JSON compat)
    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj):
                return 'inf'
            if math.isnan(obj):
                return 'nan'
        elif isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    summary_file = os.path.join(output_dir, 'sweep_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(_sanitize(summary), f, indent=2)
    print('\nSummary saved to %s' % summary_file)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Compare ESKF vs DR-ESKF-TAC (fixed) vs DR-ESKF-TAC + MUSE mu_v+R')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained MUSE model checkpoint')
    parser.add_argument('-i', action='store', nargs=2, default=None,
                        help='Single dataset: anchor_npz csv_file')
    parser.add_argument('--consts', type=int, nargs='+', default=[4])
    parser.add_argument('--trials', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--tdoa', type=int, default=2)

    # Fixed theta grid for DR-ESKF comparison
    parser.add_argument('--theta_w', type=float, nargs='+',
                        default=[0.001, 0.1, 1.0, 5.0])
    parser.add_argument('--theta_v', type=float, nargs='+',
                        default=[0.001, 0.1, 1.0, 5.0])

    # MUSE parameters (sweep same grid as fixed DR-ESKF by default)
    parser.add_argument('--theta_w_fixed', type=float, nargs='+',
                        default=None,
                        help='theta_w grid for MUSE (default: same as --theta_w)')
    parser.add_argument('--theta_v_fixed', type=float, nargs='+',
                        default=None,
                        help='theta_v grid for MUSE (default: same as --theta_v)')
    parser.add_argument('--ema_alpha', type=float, default=0.3)
    parser.add_argument('--dr_solver', choices=['fw', 'fw_exact'],
                        default='fw_exact')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str,
                        default='results/muse_comparison')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Enable calibration diagnostics (innovation whiteness, '
                             'NIS calibration, reliability diagram)')
    parser.add_argument('--predictor_eskf', action='store_true',
                        help='Also run predictor + vanilla ESKF (no DR robustification)')
    args = parser.parse_args()

    # Resolve datasets
    if args.i is not None:
        anchor_npz, csv_file = args.i
        ds_name = os.path.splitext(os.path.basename(csv_file))[0]
        datasets = [(anchor_npz, csv_file, ds_name)]
    else:
        datasets = discover_datasets(args.consts, args.trials, args.tdoa)
        if not datasets:
            print('ERROR: No datasets found')
            sys.exit(1)

    # Default MUSE theta grid = same as fixed DR-ESKF grid
    tw_fixed_list = args.theta_w_fixed if args.theta_w_fixed else args.theta_w
    tv_fixed_list = args.theta_v_fixed if args.theta_v_fixed else args.theta_v

    n_tac = len(args.theta_w) * len(args.theta_v)
    n_muse = len(tw_fixed_list) * len(tv_fixed_list)
    n_pred_eskf = 1 if args.predictor_eskf else 0
    n_configs = 1 + n_tac + n_muse + n_pred_eskf
    print('=== MUSE mu_v+R Comparison ===')
    print('  Datasets:    %d' % len(datasets))
    print('  Fixed grid:  %d x %d = %d pairs' %
          (len(args.theta_w), len(args.theta_v), n_tac))
    print('  MUSE grid:   %d x %d = %d pairs' %
          (len(tw_fixed_list), len(tv_fixed_list), n_muse))
    if args.predictor_eskf:
        print('  Pred+ESKF:   1')
    print('  Configs/ds:  1 baseline + %d fixed + %d MUSE + %d pred_eskf = %d' %
          (n_tac, n_muse, n_pred_eskf, n_configs))
    print('  Total runs:  %d' % (len(datasets) * n_configs))
    print('  Checkpoint:  %s' % args.checkpoint)
    print('========================\n')

    run_comparison(
        datasets=datasets,
        theta_w_list=args.theta_w,
        theta_v_list=args.theta_v,
        checkpoint_path=args.checkpoint,
        theta_w_fixed_list=tw_fixed_list,
        theta_v_fixed_list=tv_fixed_list,
        ema_alpha=args.ema_alpha,
        dr_solver=args.dr_solver,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        diagnostics=args.diagnostics,
        predictor_eskf=args.predictor_eskf,
    )


if __name__ == '__main__':
    main()
