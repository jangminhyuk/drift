'''
    The main file for eskf estimation
'''
#!/usr/bin/env python3
import argparse
import os, sys
import math
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy import interpolate
from sklearn.metrics import mean_squared_error

from eskf_class import ESKF
import matplotlib.pyplot as plt
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, './../'))
from utility.praser import extract_gt, extract_tdoa, extract_acc, extract_gyro, interp_meas, extract_tdoa_meas
from plot_util import plot_pos, plot_pos_err, plot_traj


def isin(t_np,t_k):
    '''
        help function for timestamp
    '''
    # check if t_k is in the numpy array t_np.
    # If t_k is in t_np, return the index and bool = Ture.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        b = True
        return res[0][0], b
    b = False
    return 0, b

def downsamp(data):
    '''
        down-sample uwb data
    '''
    data_ds = data[0::2,:]              # downsample by half

    data_ds = data_ds[0::2,:]           # downsample by half

    data_ds = data_ds[0::2,:]           # downsample by half

    # data_ds = data_ds[0::2,:]           # downsample by half

    return data_ds


def run_estimation(anchor_npz, csv_file, filter_type='eskf',
                   theta_w=0.01, theta_v=0.01, dr_solver='fw_exact',
                   verbose=True, theta_callback=None,
                   # Segment-init options (default off = backwards compatible)
                   init_pos=None, init_vel=None, init_quat=None,
                   t_start_sec=None, L_sec=None):
    '''
    Run ESKF or DR-ESKF estimation on a single dataset.

    Args:
        theta_callback: optional callable(eskf, k, t_uwb, imu_raw, t_imu_raw)
            Called at each UWB correction. Should return (theta_w, theta_v)
            to update the filter, or None to keep current values.
        init_pos, init_vel: (3,) arrays overriding the default X0 init.
            If None, the hardcoded launch pose is used.
        init_quat: 4-vector (w, x, y, z) overriding the default identity quat.
        t_start_sec, L_sec: if both given, slice all sensor data to
            [t_start_sec, t_start_sec + L_sec] AFTER the min_t time-base reset.
            Use for segment-init analysis.

    Returns a dict with all results:
        t, Xpo, Ppo, t_vicon, pos_vicon, anchor_position,
        pos_error, rms_x, rms_y, rms_z, rms_all,
        filter_type, theta_w, theta_v, dr_solver,
        dr_solve_count, dr_fallback_count
    '''
    # access the survey results
    anchor_survey = np.load(anchor_npz)
    anchor_position = anchor_survey['an_pos']
    if verbose:
        anchor_file = os.path.split(anchor_npz)[1]
        print("\nselecting anchor constellation " + str(anchor_file))

    # access csv
    df = pd.read_csv(csv_file)
    if verbose:
        csv_name = os.path.split(csv_file)[1]
        print("ESKF estimation with: " + str(csv_name))

    # tdoa2 = centralized (8 sequential anchor pairs); tdoa3 = decentralized
    # (all anchor pairs). Infer from the csv filename so both modes parse right.
    TDOA2 = ('-tdoa3' not in str(csv_file))

    # --------------- extract csv file --------------- #
    gt_pose = extract_gt(df)
    tdoa    = extract_tdoa(df)
    acc     = extract_acc(df)
    gyr     = extract_gyro(df)

    t_vicon = gt_pose[:,0];    pos_vicon = gt_pose[:,1:4]

    t_imu = acc[:,0]
    gyr_x_syn = interp_meas(gyr[:,0], gyr[:,1], t_imu).reshape(-1,1)
    gyr_y_syn = interp_meas(gyr[:,0], gyr[:,2], t_imu).reshape(-1,1)
    gyr_z_syn = interp_meas(gyr[:,0], gyr[:,3], t_imu).reshape(-1,1)

    imu = np.concatenate((acc[:,1:], gyr_x_syn, gyr_y_syn, gyr_z_syn), axis = 1)

    min_t = min(tdoa[0,0], t_imu[0], t_vicon[0])
    # get the vicon information from min_t
    t_vicon = np.array(t_vicon);
    idx = np.argwhere(t_vicon > min_t);
    t_vicon = t_vicon[idx];
    pos_vicon = np.squeeze(np.array(pos_vicon)[idx,:])

    # reset time base
    t_vicon = (t_vicon - min_t).reshape(-1,1)
    t_imu = (t_imu - min_t).reshape(-1,1)
    tdoa[:,0] = tdoa[:,0] - min_t

    # ---- Segment-init: slice all sensor arrays to [t_start_sec, t_start_sec+L_sec]
    if t_start_sec is not None and L_sec is not None:
        t_end_sec = t_start_sec + L_sec
        if verbose:
            print(f"  segment-init slice: [{t_start_sec:.2f}, {t_end_sec:.2f}] s")
        m_vicon = ((t_vicon.ravel() >= t_start_sec)
                    & (t_vicon.ravel() <= t_end_sec))
        t_vicon = t_vicon[m_vicon]
        pos_vicon = pos_vicon[m_vicon]
        m_imu = ((t_imu.ravel() >= t_start_sec) & (t_imu.ravel() <= t_end_sec))
        t_imu = t_imu[m_imu]
        imu = imu[m_imu]
        m_tdoa = (tdoa[:, 0] >= t_start_sec) & (tdoa[:, 0] <= t_end_sec)
        tdoa = tdoa[m_tdoa]

    # Keep raw (pre-downsample) IMU for theta_callback
    if theta_callback is not None:
        t_imu_raw = t_imu.ravel().copy()
        imu_raw = imu.copy()
    else:
        t_imu_raw = None
        imu_raw = None

    # ------ downsample the raw data
    t_imu = downsamp(t_imu)
    imu   = downsamp(imu)

    if TDOA2:
        # extract tdoa meas.
        tdoa_70, tdoa_01, tdoa_12, tdoa_23, tdoa_34, tdoa_45, tdoa_56, tdoa_67 = extract_tdoa_meas(tdoa[:,0], tdoa[:,1:4])
        # downsample uwb tdoa data
        tdoa_70_ds = downsamp(tdoa_70);    tdoa_01_ds = downsamp(tdoa_01)
        tdoa_12_ds = downsamp(tdoa_12);    tdoa_23_ds = downsamp(tdoa_23)
        tdoa_34_ds = downsamp(tdoa_34);    tdoa_45_ds = downsamp(tdoa_45)
        tdoa_56_ds = downsamp(tdoa_56);    tdoa_67_ds = downsamp(tdoa_67)
        # convert back to tdoa
        tdoa_c = np.concatenate((tdoa_70_ds, tdoa_01_ds, tdoa_12_ds, tdoa_23_ds,\
                                tdoa_34_ds, tdoa_45_ds, tdoa_56_ds, tdoa_67_ds), axis = 0)
    else:
        tdoa_c = downsamp(tdoa)

    sort_id=np.argsort(tdoa_c[:,0])
    t_uwb = tdoa_c[sort_id, 0].reshape(-1,1)
    uwb   = tdoa_c[sort_id, 1:4]

    # ----------------------- INITIALIZATION OF EKF -------------------------#
    # Create a compound vector t with a sorted merge of all the sensor time bases
    time = np.sort(np.concatenate((t_imu, t_uwb)))
    t = np.unique(time)
    K = t.shape[0]
    # Initial estimate for the state vector
    X0 = np.zeros((6,1))
    if init_pos is not None:
        X0[0, 0] = float(init_pos[0])
        X0[1, 0] = float(init_pos[1])
        X0[2, 0] = float(init_pos[2])
    else:
        X0[0] = 1.25;  X0[1] = 0.0;  X0[2] = 0.07
    if init_vel is not None:
        X0[3, 0] = float(init_vel[0])
        X0[4, 0] = float(init_vel[1])
        X0[5, 0] = float(init_vel[2])

    if init_quat is not None:
        # Quaternion order (w, x, y, z)
        q0 = Quaternion([float(init_quat[0]), float(init_quat[1]),
                          float(init_quat[2]), float(init_quat[3])])
    else:
        q0 = Quaternion([1,0,0,0])  # initial quaternion
    std_xy0 = 0.1;       std_z0 = 0.1;      std_vel0 = 0.1
    std_rp0 = 0.1;       std_yaw0 = 0.1
    # Initial posterior covariance
    P0 = np.diag([std_xy0**2,  std_xy0**2,  std_z0**2,\
                std_vel0**2, std_vel0**2, std_vel0**2,\
                std_rp0**2,  std_rp0**2,  std_yaw0**2 ])

    # create the filter object
    dr_solve_count = 0
    dr_fallback_count = 0
    if filter_type in ('dr_eskf', 'dr_eskf_tac'):
        from dr_eskf_class_tac import DR_ESKF_TAC
        eskf = DR_ESKF_TAC(X0, q0, P0, K, theta_w=theta_w, theta_v=theta_v,
                           solver=dr_solver)
        if verbose:
            print("DR-ESKF-TAC estimation with theta_w=%.4f, theta_v=%.4f, solver=%s" %
                  (theta_w, theta_v, dr_solver))
    elif filter_type == 'dr_eskf_cdc':
        from dr_eskf_class_cdc import DR_ESKF_CDC
        eskf = DR_ESKF_CDC(X0, q0, P0, K, theta_x=theta_w, theta_v=theta_v,
                           solver=dr_solver)
        if verbose:
            print("DR-ESKF-CDC estimation with theta_x=%.4f, theta_v=%.4f, solver=%s" %
                  (theta_w, theta_v, dr_solver))
    else:
        eskf = ESKF(X0, q0, P0, K)

    if verbose:
        print('timestep: %d' % K)
        print('Start state estimation')
    for k in range(1,K):               # k = 1 ~ K-1
        # Find what measurements are available at the current time (help function: isin() )
        imu_k,  imu_check  = isin(t_imu,   t[k-1])
        uwb_k,  uwb_check  = isin(t_uwb,   t[k-1])
        dt = t[k]-t[k-1]

        # ESKF Prediction
        eskf.predict(imu[imu_k,:], dt, imu_check, k)

        # ESKF Correction
        if uwb_check:            # if we have UWB measurement
            eskf.UWB_correct(uwb[uwb_k,:], anchor_position, k)

            # Optional theta update via callback
            if theta_callback is not None:
                cb_result = theta_callback(eskf, k, float(t[k-1]),
                                           imu_raw, t_imu_raw)
                if cb_result is not None:
                    # CDC stores its prior-state radius as theta_x; TAC uses theta_w.
                    if filter_type == 'dr_eskf_cdc':
                        eskf.theta_x, eskf.theta_v = cb_result
                    else:
                        eskf.theta_w, eskf.theta_v = cb_result

    if verbose:
        print('Finish the state estimation')

    # DR diagnostics
    if filter_type in ('dr_eskf', 'dr_eskf_tac', 'dr_eskf_cdc'):
        dr_solve_count = eskf.dr_solve_count
        dr_fallback_count = eskf.dr_fallback_count
        if verbose:
            print('DR-ESKF diagnostics:')
            print('  DR solves: %d' % dr_solve_count)
            print('  Fallbacks to vanilla: %d' % dr_fallback_count)

    ## compute the error
    # interpolate Vicon measurements
    f_x = interpolate.splrep(t_vicon, pos_vicon[:,0], s = 0.5)
    f_y = interpolate.splrep(t_vicon, pos_vicon[:,1], s = 0.5)
    f_z = interpolate.splrep(t_vicon, pos_vicon[:,2], s = 0.5)
    x_interp = interpolate.splev(t, f_x, der = 0)
    y_interp = interpolate.splev(t, f_y, der = 0)
    z_interp = interpolate.splev(t, f_z, der = 0)

    x_error = eskf.Xpo[:,0] - x_interp
    y_error = eskf.Xpo[:,1] - y_interp
    z_error = eskf.Xpo[:,2] - z_interp

    pos_error = np.concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)

    rms_x = math.sqrt(mean_squared_error(x_interp, eskf.Xpo[:,0]))
    rms_y = math.sqrt(mean_squared_error(y_interp, eskf.Xpo[:,1]))
    rms_z = math.sqrt(mean_squared_error(z_interp, eskf.Xpo[:,2]))
    rms_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)

    if verbose:
        print('RMS error: x=%.4f  y=%.4f  z=%.4f  all=%.4f [m]' % (rms_x, rms_y, rms_z, rms_all))

    return {
        't': t,
        'Xpo': eskf.Xpo,
        'Ppo': eskf.Ppo,
        'q_list': eskf.q_list,
        'nis_log': eskf.nis_log,
        't_vicon': np.squeeze(t_vicon),
        'pos_vicon': pos_vicon,
        'anchor_position': anchor_position,
        'pos_error': pos_error,
        'rms_x': rms_x,
        'rms_y': rms_y,
        'rms_z': rms_z,
        'rms_all': rms_all,
        'filter_type': filter_type,
        'theta_w': theta_w,
        'theta_v': theta_v,
        'dr_solver': dr_solver,
        'dr_solve_count': dr_solve_count,
        'dr_fallback_count': dr_fallback_count,
    }


if __name__ == "__main__":
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', nargs=2)
    parser.add_argument('--filter', choices=['eskf', 'dr_eskf', 'dr_eskf_tac', 'dr_eskf_cdc'],
                        default='eskf',
                        help='Filter type: eskf (vanilla), dr_eskf/dr_eskf_tac (TAC), or dr_eskf_cdc (CDC)')
    parser.add_argument('--theta_w', type=float, default=0.01,
                        help='DR process noise ambiguity radius (BW ball)')
    parser.add_argument('--theta_v', type=float, default=0.01,
                        help='DR measurement noise ambiguity radius (BW ball)')
    parser.add_argument('--dr_solver', choices=['fw', 'fw_exact'], default='fw_exact',
                        help='DR-MMSE solver: fw (bisection oracle) or fw_exact (exact oracle)')
    args = parser.parse_args()

    result = run_estimation(
        anchor_npz=args.i[0],
        csv_file=args.i[1],
        filter_type=args.filter,
        theta_w=args.theta_w,
        theta_v=args.theta_v,
        dr_solver=args.dr_solver,
        verbose=True
    )

    print('\nThe RMS error for position x is %f [m]' % result['rms_x'])
    print('The RMS error for position y is %f [m]' % result['rms_y'])
    print('The RMS error for position z is %f [m]' % result['rms_z'])
    print('The overall RMS error of position estimation is %f [m]\n' % result['rms_all'])

    # visualization
    plot_pos(result['t'], result['Xpo'], result['t_vicon'], result['pos_vicon'])
    plot_pos_err(result['t'], result['pos_error'], result['Ppo'])
    plot_traj(result['pos_vicon'], result['Xpo'], result['anchor_position'])
    plt.show()