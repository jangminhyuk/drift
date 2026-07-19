'''
    Single-task sweep worker for theta MUSE training data generation.

    Runs exactly ONE task from a JSONL file (selected by --task_id).

    Reference tasks (is_ref=True):
      - Run instrumented DR-ESKF-TAC filter at reference theta
      - Save per-attempt feature streams (UWB, IMU, odom) + GT positions

    Sweep tasks (is_ref=False):
      - Run DR-ESKF-TAC with given (theta_w, theta_v)
      - Save position estimates at each attempt timestamp

    Usage:
        python theta_sweep_worker.py --tasks_file cache/sweep_tasks.jsonl \
            --task_id 0 --cache_dir cache
'''
import argparse
import json
import os
import sys
import math
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy import interpolate

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cwd)
sys.path.insert(0, os.path.join(cwd, '..'))

from main import isin, downsamp
from utility.praser import (extract_gt, extract_tdoa, extract_acc,
                            extract_gyro, interp_meas, extract_tdoa_meas)
from dr_eskf_class_tac import DR_ESKF_TAC


def load_and_prepare_data(anchor_npz, csv_file):
    '''Load and preprocess dataset (same logic as main.py).

    Returns dict with: anchor_position, t, t_imu, imu, t_uwb, uwb,
                        t_vicon, pos_vicon, K, acc_raw, gyr_raw, t_imu_raw
    '''
    anchor_survey = np.load(anchor_npz)
    anchor_position = anchor_survey['an_pos']

    df = pd.read_csv(csv_file)

    gt_pose = extract_gt(df)
    tdoa = extract_tdoa(df)
    acc = extract_acc(df)
    gyr = extract_gyro(df)

    t_vicon = gt_pose[:, 0]
    pos_vicon = gt_pose[:, 1:4]
    quat_vicon = gt_pose[:, 4:8]   # [qx, qy, qz, qw]

    t_imu_raw = acc[:, 0].copy()
    t_imu = acc[:, 0]
    gyr_x_syn = interp_meas(gyr[:, 0], gyr[:, 1], t_imu).reshape(-1, 1)
    gyr_y_syn = interp_meas(gyr[:, 0], gyr[:, 2], t_imu).reshape(-1, 1)
    gyr_z_syn = interp_meas(gyr[:, 0], gyr[:, 3], t_imu).reshape(-1, 1)

    imu = np.concatenate((acc[:, 1:], gyr_x_syn, gyr_y_syn, gyr_z_syn), axis=1)

    min_t = min(tdoa[0, 0], t_imu[0], t_vicon[0])
    t_vicon = np.array(t_vicon)
    idx = np.argwhere(t_vicon > min_t)
    t_vicon = t_vicon[idx]
    pos_vicon = np.squeeze(np.array(pos_vicon)[idx, :])
    quat_vicon = np.squeeze(np.array(quat_vicon)[idx, :])

    t_vicon = (t_vicon - min_t).reshape(-1, 1)
    t_imu = (t_imu - min_t).reshape(-1, 1)
    t_imu_raw = (t_imu_raw - min_t)
    tdoa[:, 0] = tdoa[:, 0] - min_t

    # Keep raw before downsample for IMU chunk extraction
    imu_raw = imu.copy()
    t_imu_raw_ds = t_imu_raw.copy()  # pre-downsample raw timestamps

    t_imu = downsamp(t_imu)
    imu = downsamp(imu)

    if '-tdoa3' in str(csv_file):
        # TDoA3 (decentralized): keep ALL anchor pairs, just downsample.
        tdoa_c = downsamp(tdoa)
    else:
        # TDoA2 (centralized): keep the 8 sequential anchor pairs.
        tdoa_70, tdoa_01, tdoa_12, tdoa_23, tdoa_34, tdoa_45, tdoa_56, tdoa_67 = \
            extract_tdoa_meas(tdoa[:, 0], tdoa[:, 1:4])
        tdoa_70_ds = downsamp(tdoa_70);  tdoa_01_ds = downsamp(tdoa_01)
        tdoa_12_ds = downsamp(tdoa_12);  tdoa_23_ds = downsamp(tdoa_23)
        tdoa_34_ds = downsamp(tdoa_34);  tdoa_45_ds = downsamp(tdoa_45)
        tdoa_56_ds = downsamp(tdoa_56);  tdoa_67_ds = downsamp(tdoa_67)
        tdoa_c = np.concatenate((tdoa_70_ds, tdoa_01_ds, tdoa_12_ds, tdoa_23_ds,
                                 tdoa_34_ds, tdoa_45_ds, tdoa_56_ds, tdoa_67_ds),
                                axis=0)

    sort_id = np.argsort(tdoa_c[:, 0])
    t_uwb = tdoa_c[sort_id, 0].reshape(-1, 1)
    uwb = tdoa_c[sort_id, 1:4]

    t_all = np.sort(np.concatenate((t_imu, t_uwb)))
    t = np.unique(t_all)
    K = t.shape[0]

    return {
        'anchor_position': anchor_position,
        't': t,
        't_imu': t_imu,
        'imu': imu,
        't_uwb': t_uwb,
        'uwb': uwb,
        't_vicon': t_vicon,
        'pos_vicon': pos_vicon,
        'quat_vicon': quat_vicon,
        'K': K,
        'imu_raw': imu_raw,
        't_imu_raw': t_imu_raw_ds,
    }


def init_eskf(K, theta_w, theta_v):
    '''Create and return initialized DR_ESKF_TAC.'''
    X0 = np.zeros((6, 1))
    X0[0] = 1.25;  X0[1] = 0.0;  X0[2] = 0.07
    q0 = Quaternion([1, 0, 0, 0])
    std_xy0 = 0.1;  std_z0 = 0.1;  std_vel0 = 0.1
    std_rp0 = 0.1;  std_yaw0 = 0.1
    P0 = np.diag([std_xy0**2, std_xy0**2, std_z0**2,
                  std_vel0**2, std_vel0**2, std_vel0**2,
                  std_rp0**2, std_rp0**2, std_yaw0**2])
    return DR_ESKF_TAC(X0, q0, P0, K, theta_w=theta_w, theta_v=theta_v,
                       solver='fw_exact')


def get_imu_chunk(t_imu_raw, imu_raw, t_prev, t_curr, L=50):
    '''Extract IMU samples in (t_prev, t_curr] and resample to L samples.

    Returns array of shape (L, 6).
    '''
    mask = (t_imu_raw > t_prev) & (t_imu_raw <= t_curr)
    idx = np.where(mask)[0]

    if len(idx) < 2:
        return np.zeros((L, 6))

    t_chunk = t_imu_raw[idx]
    imu_chunk = imu_raw[idx]  # (n, 6)

    # Resample to L evenly-spaced points
    t_resamp = np.linspace(t_chunk[0], t_chunk[-1], L)
    out = np.zeros((L, 6))
    for ch in range(6):
        out[:, ch] = np.interp(t_resamp, t_chunk, imu_chunk[:, ch])
    return out


def interpolate_gt_at(t_vicon, pos_vicon, t_query):
    '''Interpolate GT position at query times using splines.'''
    tv = np.squeeze(t_vicon)
    pv = pos_vicon
    if pv.ndim == 1:
        pv = pv.reshape(1, -1)

    f_x = interpolate.splrep(tv, pv[:, 0], s=0.5)
    f_y = interpolate.splrep(tv, pv[:, 1], s=0.5)
    f_z = interpolate.splrep(tv, pv[:, 2], s=0.5)

    px = interpolate.splev(t_query, f_x, der=0)
    py = interpolate.splev(t_query, f_y, der=0)
    pz = interpolate.splev(t_query, f_z, der=0)
    return np.stack([px, py, pz], axis=-1)


def compute_gt_residuals(data, t_uwb_arr, uwb_data):
    '''Compute GT TDOA residuals for each UWB measurement.

    For each attempt k:
        p_uwb_gt = R_gt @ t_uv + p_gt
        tdoa_gt = ||anchor_B - p_uwb_gt|| - ||anchor_A - p_uwb_gt||
        r_k = z_k - tdoa_gt

    Args:
        data: dict from load_and_prepare_data() (includes quat_vicon)
        t_uwb_arr: (Nu,) UWB attempt timestamps
        uwb_data: (Nu, 3) UWB data [idA, idB, tdoa_meas] per attempt

    Returns:
        gt_residuals: (Nu,) scalar residual per attempt
        tdoa_gt_arr: (Nu,) GT TDOA per attempt (for diagnostics)
    '''
    anchor_pos = data['anchor_position']  # (8, 3)
    t_uv = np.array([-0.01245, 0.00127, 0.0908])  # lever arm

    # Interpolate GT position to UWB times
    pgt = interpolate_gt_at(data['t_vicon'], data['pos_vicon'], t_uwb_arr)

    # Interpolate GT quaternion to UWB times (nearest-neighbor)
    tv = np.squeeze(data['t_vicon'])
    nearest_idx = np.searchsorted(tv, t_uwb_arr, side='left')
    nearest_idx = np.clip(nearest_idx, 0, len(tv) - 1)
    quat_gt = data['quat_vicon'][nearest_idx]  # (Nu, 4) [qx, qy, qz, qw]

    Nu = len(t_uwb_arr)
    gt_residuals = np.zeros(Nu)
    tdoa_gt_arr = np.zeros(Nu)

    for u in range(Nu):
        idA = int(uwb_data[u, 0])
        idB = int(uwb_data[u, 1])
        z_meas = uwb_data[u, 2]

        # GT rotation matrix (pyquaternion convention: [w, x, y, z])
        qx, qy, qz, qw = quat_gt[u]
        R_gt = Quaternion([qw, qx, qy, qz]).rotation_matrix

        # GT UWB tag position (accounting for lever arm)
        p_uwb_gt = R_gt @ t_uv + pgt[u]

        d_A = np.linalg.norm(anchor_pos[idA] - p_uwb_gt)
        d_B = np.linalg.norm(anchor_pos[idB] - p_uwb_gt)
        tdoa_gt = d_B - d_A

        gt_residuals[u] = z_meas - tdoa_gt
        tdoa_gt_arr[u] = tdoa_gt

    return gt_residuals, tdoa_gt_arr


def run_reference_task(data, theta_w_ref, theta_v_ref, L_imu=50,
                       segment_len=0, gt_reset_interval=0):
    '''Run instrumented filter and extract per-attempt feature streams.

    If gt_reset_interval > 0, resets state/covariance to GT every N UWB
    attempts to prevent accumulated drift from corrupting training features.

    Returns dict with arrays: t_uwb_attempts, Xuwb, Ximu, Xodom, pgt,
                              phat, attempt_k, segment_id, uwb_data
    '''
    eskf = init_eskf(data['K'], theta_w_ref, theta_v_ref)
    t = data['t']
    K = data['K']
    t_imu_arr = data['t_imu']
    imu_arr = data['imu']
    t_uwb_arr = data['t_uwb']
    uwb_arr = data['uwb']
    anchor_position = data['anchor_position']

    # Run filter, collecting attempt timestamps
    uwb_attempt_times = []
    uwb_attempt_k = []
    uwb_count = 0

    prev_uwb_t = 0.0
    for k in range(1, K):
        imu_k, imu_check = isin(t_imu_arr, t[k-1])
        uwb_k, uwb_check = isin(t_uwb_arr, t[k-1])
        dt = t[k] - t[k-1]

        eskf.predict(imu_arr[imu_k, :], dt, imu_check, k)

        if uwb_check:
            t_uwb_now = float(t[k-1])
            eskf.UWB_correct(uwb_arr[uwb_k, :], anchor_position, k,
                             t_uwb=t_uwb_now)
            uwb_attempt_times.append(t_uwb_now)
            uwb_attempt_k.append(k)
            uwb_count += 1

            # GT reset for cleaner training features
            if gt_reset_interval > 0 and uwb_count > 0 and \
                    uwb_count % gt_reset_interval == 0:
                pos_gt = interpolate_gt_at(
                    data['t_vicon'], data['pos_vicon'],
                    np.array([t_uwb_now])).ravel()
                vel_gt = _gt_velocity_at(
                    data['t_vicon'], data['pos_vicon'], t_uwb_now)
                _reset_eskf_to_gt(eskf, k, pos_gt, vel_gt)

    Nu = len(uwb_attempt_times)
    t_attempts = np.array(uwb_attempt_times)

    # Extract features from attempt_log
    D_uwb = 12
    D_odom = 21
    Xuwb = np.zeros((Nu, D_uwb))
    Ximu = np.zeros((Nu, L_imu, 6))
    Xodom = np.zeros((Nu, D_odom))
    phat = np.zeros((Nu, 3))
    uwb_data = np.zeros((Nu, 3))  # [anchor_A_id, anchor_B_id, meas_raw]

    for u, attempt in enumerate(eskf.attempt_log):
        k = attempt['k']

        # Per-attempt UWB measurement data for GT residual computation
        uwb_data[u] = [
            float(attempt['anchor_A_id']),
            float(attempt['anchor_B_id']),
            attempt['meas_raw'],
        ]

        # UWB token (D_uwb=12)
        S_val = attempt.get('S_star', eskf.std_uwb_tdoa**2)
        z_norm = attempt.get('z_norm', 0.0)
        logS = math.log(S_val + 1e-12)
        dt_uwb = attempt.get('dt_uwb', 0.0)
        accepted = 1.0 if attempt.get('accepted', False) else 0.0
        gating_metric = attempt.get('gating_metric', 0.0)
        G_norm_val = attempt.get('G_norm', 0.0)
        trace_Ppr = attempt.get('trace_Ppr', 0.0)
        log_trace_Ppr_S = math.log(trace_Ppr / (S_val + 1e-12) + 1e-12)

        Xuwb[u] = [
            z_norm, logS, attempt['meas_raw'], dt_uwb, accepted,
            float(attempt['anchor_A_id']), float(attempt['anchor_B_id']),
            attempt['predicted_meas'], gating_metric,
            G_norm_val, log_trace_Ppr_S, attempt['err_uwb'],
        ]

        # IMU chunk
        t_prev = t_attempts[u - 1] if u > 0 else 0.0
        t_curr = t_attempts[u]
        Ximu[u] = get_imu_chunk(data['t_imu_raw'], data['imu_raw'],
                                t_prev, t_curr, L_imu)

        # Odom token (D_odom=21)
        pos = eskf.Xpo[k, 0:3]
        vel = eskf.Xpo[k, 3:6]
        q = eskf.q_list[k]
        # Euler angles from quaternion
        qobj = Quaternion(q)
        # yaw, pitch, roll from rotation matrix
        R = qobj.rotation_matrix
        pitch = math.asin(max(-1, min(1, -R[2, 0])))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
        euler = [roll, pitch, yaw]

        log_trace_Ppr_val = math.log(trace_Ppr + 1e-12)
        gain_ratio = attempt.get('gain_ratio', 0.0)
        corr_norm = attempt.get('corr_norm', 0.0)
        speed = float(np.linalg.norm(vel))

        # IMU means over chunk
        imu_chunk = Ximu[u]  # (L, 6)
        acc_mean = imu_chunk[:, :3].mean(axis=0)  # 3
        gyro_mean = imu_chunk[:, 3:].mean(axis=0)  # 3

        log_tw = math.log(eskf.theta_w + 1e-12)
        log_tv = math.log(eskf.theta_v + 1e-12)

        Xodom[u] = np.concatenate([
            pos, vel, euler,
            [log_trace_Ppr_val, gain_ratio, corr_norm, speed],
            acc_mean, gyro_mean,
            [log_tw, log_tv],
        ])

        # Position estimate at this attempt
        phat[u] = eskf.Xpo[k, 0:3]

    # GT positions at attempt timestamps
    pgt = interpolate_gt_at(data['t_vicon'], data['pos_vicon'], t_attempts)

    # Segment IDs for GT reset tracking
    segment_id = np.zeros(Nu, dtype=np.int32)
    if gt_reset_interval > 0:
        for u in range(Nu):
            segment_id[u] = u // gt_reset_interval

    return {
        't_uwb': t_attempts,
        'Xuwb': Xuwb,
        'Ximu': Ximu,
        'Xodom': Xodom,
        'pgt': pgt,
        'phat': phat,
        'attempt_k': np.array(uwb_attempt_k),
        'segment_id': segment_id,
        'uwb_data': uwb_data,
    }


def run_delayed_oracle_task(data, oracle_tw, oracle_tv,
                            theta_w_ref=0.1, theta_v_ref=0.1,
                            L_imu=50, segment_len=100):
    '''Run instrumented filter with delayed-oracle theta switching.

    Like run_reference_task, but at each segment boundary the filter
    switches theta to the previous window's oracle-optimal value.
    This matches online inference where the model predicts theta for
    the NEXT window based on the CURRENT window's features.

    Protocol:
      - Window 0: theta = (theta_w_ref, theta_v_ref)
      - Window k >= 1: theta = oracle theta from window k-1
      - GT reset at every segment boundary (same as run_reference_task)

    Returns same dict structure as run_reference_task.
    '''
    eskf = init_eskf(data['K'], theta_w_ref, theta_v_ref)
    t = data['t']
    K = data['K']
    t_imu_arr = data['t_imu']
    imu_arr = data['imu']
    t_uwb_arr = data['t_uwb']
    uwb_arr = data['uwb']
    anchor_position = data['anchor_position']

    uwb_attempt_times = []
    uwb_attempt_k = []
    uwb_count = 0

    prev_uwb_t = 0.0
    for k in range(1, K):
        imu_k, imu_check = isin(t_imu_arr, t[k-1])
        uwb_k, uwb_check = isin(t_uwb_arr, t[k-1])
        dt = t[k] - t[k-1]

        eskf.predict(imu_arr[imu_k, :], dt, imu_check, k)

        if uwb_check:
            t_uwb_now = float(t[k-1])
            eskf.UWB_correct(uwb_arr[uwb_k, :], anchor_position, k,
                             t_uwb=t_uwb_now)
            uwb_attempt_times.append(t_uwb_now)
            uwb_attempt_k.append(k)
            uwb_count += 1

            # At segment boundaries: GT reset + theta switch
            if segment_len > 0 and uwb_count % segment_len == 0:
                pos_gt = interpolate_gt_at(
                    data['t_vicon'], data['pos_vicon'],
                    np.array([t_uwb_now])).ravel()
                vel_gt = _gt_velocity_at(
                    data['t_vicon'], data['pos_vicon'], t_uwb_now)
                _reset_eskf_to_gt(eskf, k, pos_gt, vel_gt)

                # Switch theta to previous window's oracle
                next_seg = uwb_count // segment_len   # = 1, 2, 3, ...
                prev_seg = next_seg - 1               # = 0, 1, 2, ...
                if prev_seg < len(oracle_tw):
                    eskf.theta_w = float(oracle_tw[prev_seg])
                    eskf.theta_v = float(oracle_tv[prev_seg])

    Nu = len(uwb_attempt_times)
    t_attempts = np.array(uwb_attempt_times)

    # Extract features from attempt_log (identical to run_reference_task)
    D_uwb = 12
    D_odom = 21
    Xuwb = np.zeros((Nu, D_uwb))
    Ximu = np.zeros((Nu, L_imu, 6))
    Xodom = np.zeros((Nu, D_odom))
    phat = np.zeros((Nu, 3))

    for u, attempt in enumerate(eskf.attempt_log):
        k = attempt['k']

        # UWB token (D_uwb=12)
        S_val = attempt.get('S_star', eskf.std_uwb_tdoa**2)
        z_norm = attempt.get('z_norm', 0.0)
        logS = math.log(S_val + 1e-12)
        dt_uwb = attempt.get('dt_uwb', 0.0)
        accepted = 1.0 if attempt.get('accepted', False) else 0.0
        gating_metric = attempt.get('gating_metric', 0.0)
        G_norm_val = attempt.get('G_norm', 0.0)
        trace_Ppr = attempt.get('trace_Ppr', 0.0)
        log_trace_Ppr_S = math.log(trace_Ppr / (S_val + 1e-12) + 1e-12)

        Xuwb[u] = [
            z_norm, logS, attempt['meas_raw'], dt_uwb, accepted,
            float(attempt['anchor_A_id']), float(attempt['anchor_B_id']),
            attempt['predicted_meas'], gating_metric,
            G_norm_val, log_trace_Ppr_S, attempt['err_uwb'],
        ]

        # IMU chunk
        t_prev = t_attempts[u - 1] if u > 0 else 0.0
        t_curr = t_attempts[u]
        Ximu[u] = get_imu_chunk(data['t_imu_raw'], data['imu_raw'],
                                t_prev, t_curr, L_imu)

        # Odom token (D_odom=21)
        pos = eskf.Xpo[k, 0:3]
        vel = eskf.Xpo[k, 3:6]
        q = eskf.q_list[k]
        qobj = Quaternion(q)
        R = qobj.rotation_matrix
        pitch = math.asin(max(-1, min(1, -R[2, 0])))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
        euler = [roll, pitch, yaw]

        log_trace_Ppr_val = math.log(trace_Ppr + 1e-12)
        gain_ratio = attempt.get('gain_ratio', 0.0)
        corr_norm = attempt.get('corr_norm', 0.0)
        speed = float(np.linalg.norm(vel))

        imu_chunk = Ximu[u]
        acc_mean = imu_chunk[:, :3].mean(axis=0)
        gyro_mean = imu_chunk[:, 3:].mean(axis=0)

        log_tw = math.log(eskf.theta_w + 1e-12)
        log_tv = math.log(eskf.theta_v + 1e-12)

        Xodom[u] = np.concatenate([
            pos, vel, euler,
            [log_trace_Ppr_val, gain_ratio, corr_norm, speed],
            acc_mean, gyro_mean,
            [log_tw, log_tv],
        ])

        phat[u] = eskf.Xpo[k, 0:3]

    # GT positions at attempt timestamps
    pgt = interpolate_gt_at(data['t_vicon'], data['pos_vicon'], t_attempts)

    # Per-attempt segment ID
    segment_id = np.zeros(Nu, dtype=np.int32)
    for u in range(Nu):
        segment_id[u] = u // segment_len if segment_len > 0 else 0

    return {
        't_uwb': t_attempts,
        'Xuwb': Xuwb,
        'Ximu': Ximu,
        'Xodom': Xodom,
        'pgt': pgt,
        'phat': phat,
        'attempt_k': np.array(uwb_attempt_k),
        'segment_id': segment_id,
    }


def _gt_velocity_at(t_vicon, pos_vicon, t_query, h=0.02):
    '''Estimate GT velocity at t_query via central finite difference on
    the Vicon spline.'''
    pgt_plus = interpolate_gt_at(t_vicon, pos_vicon,
                                 np.array([t_query + h]))
    pgt_minus = interpolate_gt_at(t_vicon, pos_vicon,
                                  np.array([t_query - h]))
    return ((pgt_plus - pgt_minus) / (2 * h)).ravel()


def _reset_eskf_to_gt(eskf, k, pos_gt, vel_gt):
    '''Hard-reset the filter state at timestep k to ground truth.

    Resets position and velocity to GT, keeps current quaternion (attitude
    usually tracks well), resets covariance to a nominal value, and clears
    all DR-specific internal tracking so the solver starts fresh.
    '''
    eskf.Xpo[k, 0:3] = pos_gt
    eskf.Xpo[k, 3:6] = vel_gt
    eskf.Xpr[k, 0:3] = pos_gt
    eskf.Xpr[k, 3:6] = vel_gt

    P_reset = np.diag([0.1**2]*3 + [0.1**2]*3 + [0.1**2]*3)
    eskf.Ppo[k] = P_reset
    eskf.Ppr[k] = P_reset

    # Reset APA decomposition tracking
    eskf._Phi = np.eye(9)
    eskf._G_list = []
    eskf._P_post_last_meas = P_reset.copy()

    # Clear warm-start so the solver doesn't carry stale iterates
    eskf._warm_Q_w = None
    eskf._warm_Sigma_v = None


def compute_oracle_labels(data, cache_dir, ds_key, theta_w_grid, theta_v_grid,
                          window_size=100, beta=0.5):
    '''Compute per-attempt oracle theta using sliding-window RMSE.

    Pure numpy — no filter run needed, just loads sweep .npz files.
    Each sweep file is a continuous trajectory (no GT resets).

    Soft targets use rank-based softmax: softmax(-beta * rank) where
    rank is the RMSE ordering (0=best).  This is scale-invariant and
    always produces peaked distributions.  beta=0.5 gives top-1 prob
    ~0.39 for 42 theta pairs.

    For each attempt u >= window_size - 1, computes a causal
    sliding-window RMSE over [u - W + 1, u] for each theta pair,
    then derives soft/hard labels via softmax(-beta * rmse).

    Uses np.cumsum for O(1) per-window RMSE computation.

    Returns dict with per-attempt oracle labels (n_labeled = Nu - W + 1):
        oracle_tw, oracle_tv          - hard argmin theta per attempt
        oracle_log_tw, oracle_log_tv  - log of hard argmin
        soft_log_tw, soft_log_tv      - softmax-weighted log theta
        soft_dist_tw, soft_dist_tv    - marginalized soft distributions
        hard_cls_tw, hard_cls_tv      - hard class indices
        confidence                    - RMSE gap between best and 2nd best
        best_rmse                     - RMSE of best theta per attempt
        n_labeled                     - number of labeled attempts
        window_size                   - sliding window size
        label_offset                  - = window_size - 1
    '''
    from itertools import product as iterproduct

    W = window_size
    theta_pairs = list(iterproduct(theta_w_grid, theta_v_grid))
    n_theta = len(theta_pairs)

    # Get GT positions at UWB attempt times from any sweep file
    sample_tw_str = '%.6g' % theta_w_grid[0]
    sample_tv_str = '%.6g' % theta_v_grid[0]
    sample_path = os.path.join(cache_dir, 'sweeps', ds_key,
                               'tw%s_tv%s.npz' % (sample_tw_str, sample_tv_str))
    sample = np.load(sample_path)
    t_uwb_attempts = sample['t_uwb']
    Nu = len(t_uwb_attempts)

    # Interpolate GT at attempt times
    pgt = interpolate_gt_at(data['t_vicon'], data['pos_vicon'], t_uwb_attempts)

    # Load all sweep phat and compute per-attempt squared error
    sq_err = np.zeros((n_theta, Nu))
    for j, (tw, tv) in enumerate(theta_pairs):
        tw_str = '%.6g' % tw
        tv_str = '%.6g' % tv
        sweep_path = os.path.join(cache_dir, 'sweeps', ds_key,
                                  'tw%s_tv%s.npz' % (tw_str, tv_str))
        sweep = np.load(sweep_path)
        phat_j = sweep['phat']
        n = min(Nu, phat_j.shape[0])
        diff = phat_j[:n] - pgt[:n]
        sq_err[j, :n] = np.sum(diff**2, axis=1)

    # Sliding-window mean squared error via cumsum (O(1) per window)
    # cumsum[j, u] = sum of sq_err[j, 0:u+1]
    cumsum = np.cumsum(sq_err, axis=1)  # (n_theta, Nu)

    n_labeled = Nu - W + 1
    if n_labeled <= 0:
        raise ValueError('Nu=%d < window_size=%d, cannot compute labels' %
                         (Nu, W))

    # window_mse[j, i] = mean(sq_err[j, i:i+W]) for i in [0, n_labeled)
    # = (cumsum[j, i+W-1] - cumsum[j, i-1]) / W  (with cumsum[-1]=0)
    end_idx = np.arange(W - 1, Nu)        # length n_labeled
    start_idx = np.arange(0, n_labeled)    # length n_labeled
    # cumsum at end of window
    cs_end = cumsum[:, end_idx]            # (n_theta, n_labeled)
    # cumsum just before start of window (0 for start_idx==0)
    cs_start = np.zeros((n_theta, n_labeled))
    if W < Nu:
        cs_start[:, 1:] = cumsum[:, start_idx[1:] - 1]
    window_mse = (cs_end - cs_start) / W   # (n_theta, n_labeled)
    window_rmse = np.sqrt(window_mse)       # (n_theta, n_labeled)

    # Pre-compute log-theta arrays
    log_tw_arr = np.array([math.log(tw + 1e-12) for tw, _ in theta_pairs])
    log_tv_arr = np.array([math.log(tv + 1e-12) for _, tv in theta_pairs])

    # Grid arrays and index maps for marginalization
    tw_grid = np.array(theta_w_grid)
    tv_grid = np.array(theta_v_grid)
    n_tw = len(theta_w_grid)
    n_tv = len(theta_v_grid)

    # Build mapping: theta_pairs[j] -> (tw_idx, tv_idx)
    tw_idx_map = {v: i for i, v in enumerate(theta_w_grid)}
    tv_idx_map = {v: i for i, v in enumerate(theta_v_grid)}
    pair_tw_idx = np.array([tw_idx_map[tw] for tw, _ in theta_pairs])
    pair_tv_idx = np.array([tv_idx_map[tv] for _, tv in theta_pairs])

    # Per-attempt labels
    oracle_tw = np.zeros(n_labeled)
    oracle_tv = np.zeros(n_labeled)
    oracle_log_tw = np.zeros(n_labeled)
    oracle_log_tv = np.zeros(n_labeled)
    soft_log_tw = np.zeros(n_labeled)
    soft_log_tv = np.zeros(n_labeled)
    confidence = np.zeros(n_labeled)
    best_rmse_arr = np.zeros(n_labeled)

    # Classification soft target distributions
    soft_dist_tw = np.zeros((n_labeled, n_tw))
    soft_dist_tv = np.zeros((n_labeled, n_tv))
    # Hard class indices
    hard_cls_tw = np.zeros(n_labeled, dtype=np.int32)
    hard_cls_tv = np.zeros(n_labeled, dtype=np.int32)

    for i in range(n_labeled):
        rmse_i = window_rmse[:, i]  # (n_theta,)

        # Hard label: argmin
        best_j = int(np.argmin(rmse_i))
        best_tw_val, best_tv_val = theta_pairs[best_j]
        oracle_tw[i] = best_tw_val
        oracle_tv[i] = best_tv_val
        oracle_log_tw[i] = math.log(best_tw_val + 1e-12)
        oracle_log_tv[i] = math.log(best_tv_val + 1e-12)

        # Hard class indices
        hard_cls_tw[i] = tw_idx_map[best_tw_val]
        hard_cls_tv[i] = tv_idx_map[best_tv_val]

        # Softmax weights over all (tw, tv) pairs using rank-based targets.
        # Ranks are scale-invariant: best theta gets rank 0, worst gets
        # n_theta-1.  softmax(-beta * rank) always produces peaked
        # distributions regardless of absolute RMSE differences.
        ranks = np.argsort(np.argsort(rmse_i)).astype(np.float64)
        w = np.exp(-beta * ranks)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n_theta) / n_theta

        # Soft regression labels (backward compat)
        soft_log_tw[i] = float(np.dot(w, log_tw_arr))
        soft_log_tv[i] = float(np.dot(w, log_tv_arr))

        # Marginalize joint distribution to get per-axis distributions
        for j in range(n_theta):
            soft_dist_tw[i, pair_tw_idx[j]] += w[j]
            soft_dist_tv[i, pair_tv_idx[j]] += w[j]

        # Confidence: gap between best and 2nd best
        sorted_rmse = np.sort(rmse_i)
        confidence[i] = float(sorted_rmse[1] - sorted_rmse[0]) \
            if n_theta > 1 else 1.0
        best_rmse_arr[i] = float(rmse_i[best_j])

    return {
        'oracle_tw': oracle_tw,
        'oracle_tv': oracle_tv,
        'oracle_log_tw': oracle_log_tw,
        'oracle_log_tv': oracle_log_tv,
        'soft_log_tw': soft_log_tw,
        'soft_log_tv': soft_log_tv,
        'confidence': confidence,
        'best_rmse': best_rmse_arr,
        'n_labeled': n_labeled,
        'window_size': W,
        'label_offset': W - 1,
        # Classification targets
        'soft_dist_tw': soft_dist_tw,
        'soft_dist_tv': soft_dist_tv,
        'hard_cls_tw': hard_cls_tw,
        'hard_cls_tv': hard_cls_tv,
        'theta_w_grid': tw_grid,
        'theta_v_grid': tv_grid,
    }


def run_sweep_task(data, theta_w, theta_v, segment_len=0):
    '''Run filter with given thetas as a single continuous trajectory.

    No GT resets — errors accumulate naturally so that theta-sensitivity
    is captured in the position estimates.

    Returns dict with: t_uwb_attempts, phat, segment_id
    '''
    eskf = init_eskf(data['K'], theta_w, theta_v)
    t = data['t']
    K = data['K']

    uwb_attempt_times = []
    uwb_attempt_k = []
    uwb_count = 0

    for k in range(1, K):
        imu_k, imu_check = isin(data['t_imu'], t[k-1])
        uwb_k, uwb_check = isin(data['t_uwb'], t[k-1])
        dt = t[k] - t[k-1]

        eskf.predict(data['imu'][imu_k, :], dt, imu_check, k)

        if uwb_check:
            eskf.UWB_correct(data['uwb'][uwb_k, :], data['anchor_position'], k)
            uwb_attempt_times.append(float(t[k-1]))
            uwb_attempt_k.append(k)
            uwb_count += 1

    Nu = len(uwb_attempt_times)
    phat = np.zeros((Nu, 3))
    segment_id = np.zeros(Nu, dtype=np.int32)
    for u, k_val in enumerate(uwb_attempt_k):
        phat[u] = eskf.Xpo[k_val, 0:3]

    return {
        't_uwb': np.array(uwb_attempt_times),
        'phat': phat,
        'segment_id': segment_id,
    }


def main():
    parser = argparse.ArgumentParser(description='Single sweep worker')
    parser.add_argument('--tasks_file', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--L_imu', type=int, default=50)
    parser.add_argument('--segment_len', type=int, default=100,
                        help='GT-reset interval in UWB attempts (0=no reset)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Load task
    with open(args.tasks_file, 'r') as f:
        for i, line in enumerate(f):
            if i == args.task_id:
                task = json.loads(line)
                break
        else:
            print('ERROR: task_id %d not found in %s' %
                  (args.task_id, args.tasks_file))
            sys.exit(1)

    ds_key = task['dataset_key']
    is_ref = task['is_ref']

    if is_ref:
        out_path = os.path.join(args.cache_dir, 'ref_inputs', ds_key + '.npz')
    else:
        tw_str = '%.6g' % task['theta_w']
        tv_str = '%.6g' % task['theta_v']
        out_path = os.path.join(args.cache_dir, 'sweeps', ds_key,
                                'tw%s_tv%s.npz' % (tw_str, tv_str))

    if os.path.exists(out_path) and not args.overwrite:
        print('SKIP (exists): %s' % out_path)
        return

    print('Task %d: %s [%s] tw=%.4g tv=%.4g ref=%s' %
          (args.task_id, ds_key, 'REF' if is_ref else 'SWEEP',
           task['theta_w'], task['theta_v'], is_ref))

    # Load data
    data = load_and_prepare_data(task['anchor_npz'], task['csv_file'])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if is_ref:
        result = run_reference_task(data, task['theta_w'], task['theta_v'],
                                    L_imu=args.L_imu,
                                    segment_len=args.segment_len)
        np.savez(out_path,
                 t_uwb=result['t_uwb'],
                 Xuwb=result['Xuwb'],
                 Ximu=result['Ximu'],
                 Xodom=result['Xodom'],
                 pgt=result['pgt'],
                 phat=result['phat'],
                 attempt_k=result['attempt_k'],
                 segment_id=result['segment_id'])
        print('  Saved ref inputs: %s  (Nu=%d, segments=%d)' %
              (out_path, len(result['t_uwb']),
               result['segment_id'].max() + 1))
    else:
        result = run_sweep_task(data, task['theta_w'], task['theta_v'],
                               segment_len=args.segment_len)
        np.savez(out_path,
                 t_uwb=result['t_uwb'],
                 phat=result['phat'],
                 segment_id=result['segment_id'])
        print('  Saved sweep: %s  (Nu=%d, segments=%d, seg_len=%d)' %
              (out_path, len(result['t_uwb']),
               result['segment_id'].max() + 1, args.segment_len))


if __name__ == '__main__':
    main()
