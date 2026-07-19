"""KF-GINS Python runner with optional adapter + DR support.

Drives the actual KF-GINS C++ filter step-by-step from Python via pybind11.
At each GNSS update, optionally queries the learned adapter for (mu_v, R)
and injects them into KF-GINS's gnssUpdate via the override hooks.

Usage:
    python scripts/gnss_imu/runners/main_kfgins.py \
        --seq building00 --gnss_source F9P \
        [--checkpoint path/to/ckpt.pt]  # enables adapter
        [--output results.json]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'adaptive')))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'data')))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', 'cpp_kfgins')))

RAW_DIR = os.environ.get('KFGINS_RAW_DIR',
                         os.path.join(os.environ.get('DRIFT_DATA_ROOT','data'), 'i2nav_robot/raw'))

# ADIS16465 IMU noise (i2Nav-Robot default; matches KF-GINS YAML config)
_IMU_NOISE_ADIS16465 = {
    'arw':  0.24,    # deg/sqrt(hr)
    'vrw':  0.24,    # m/s/sqrt(hr)
    'gbstd': 50.0,   # deg/hr
    'abstd': 250.0,  # mGal
    'gsstd': 1000.0, # ppm
    'asstd': 1000.0, # ppm
    'corrtime': 1.0, # hr
}

# Xsens MTi 10 IMU noise (UrbanNav).
#
# UrbanNav publishes gyr_n=1.027e-2 (units 'rad/s'). If that is per-sample
# discrete std at 400Hz, ARW = gyr_n / sqrt(fs) × (180/π × √3600) ≈ 3 deg/√h.
# If it is continuous spectral density (rad/s/√Hz), ARW = gyr_n × (180/π × √3600)
# ≈ 35 deg/√h. We tested 35 deg/√h → catastrophic over-inflation (NIS=0.20,
# RMSE_h=356m); the YAML field is the per-sample discrete std interpretation.
# Final spec uses ~5× ADIS-16465 (a middle-grade tactical IMU profile) which
# matches Xsens MTi 10 datasheet specs reasonably while keeping NIS calibrated.
_IMU_NOISE_XSENS_MTI10 = {
    'arw':  1.2,     # deg/sqrt(hr)  ~5× ADIS
    'vrw':  0.5,     # m/s/sqrt(hr)  ~2× ADIS
    'gbstd': 100.0,  # deg/hr
    'abstd': 500.0,  # mGal
    'gsstd': 2000.0, # ppm
    'asstd': 2000.0, # ppm
    'corrtime': 1.0, # hr
}

_IMU_NOISE_PRESETS = {
    'adis16465': _IMU_NOISE_ADIS16465,
    'xsens_mti10': _IMU_NOISE_XSENS_MTI10,
}

# Default to ADIS-16465 (i2Nav-Robot) so existing pipeline is unchanged.
# Override via env var, e.g. KFGINS_IMU_PRESET=xsens_mti10.
IMU_NOISE = _IMU_NOISE_PRESETS[os.environ.get('KFGINS_IMU_PRESET', 'adis16465')]

ANTLEVER = np.array([0.0, -0.07, -0.365])


def load_raw_data(seq_name: str, gnss_source: str = 'F9P'):
    """Load raw IMU, GNSS, and GT data from text files."""
    base = os.path.join(RAW_DIR, seq_name)

    imu_file = os.path.join(base, f'{seq_name}_ADIS16465_IMU.txt')
    gnss_file = os.path.join(base, f'{seq_name}_{gnss_source}_GNSS.pos')
    gt_file = os.path.join(base, f'{seq_name}_groundtruth.nav')

    # IMU: t, dtheta_x, dtheta_y, dtheta_z, dvel_x, dvel_y, dvel_z
    imu = np.loadtxt(imu_file)
    # GNSS: t, lat(deg), lon(deg), alt(m), std_lat, std_lon, std_alt
    gnss = np.genfromtxt(gnss_file, usecols=range(7), invalid_raise=False)
    gnss = gnss[~np.isnan(gnss).any(axis=1)]
    # GT: t, pos_n, pos_e, pos_d, vel_n, vel_e, vel_d, roll, pitch, yaw
    gt = np.genfromtxt(gt_file, usecols=range(10), invalid_raise=False)
    gt = gt[~np.isnan(gt).any(axis=1)]

    return imu, gnss, gt


def create_engine(gnss, gt, gnss_start_time):
    """Create a KF-GINS engine matching the C++ binary's init convention.

    Init position: from first GNSS fix (geodetic).
    Init velocity + attitude: from the GT sample nearest to GNSS start time.
    This matches the C++ binary's YAML config pattern where initvel/initatt
    come from the first GT line.
    """
    import kfgins_py

    opts = kfgins_py.GINSOptions()

    # Init position from first GNSS fix (geodetic)
    opts.initstate.pos = np.array([
        gnss[0, 1] * math.pi / 180,
        gnss[0, 2] * math.pi / 180,
        gnss[0, 3],
    ])

    # Find GT sample nearest to GNSS start time (matches C++ binary behavior)
    gt_t = gt[:, 0]
    gt_idx = np.argmin(np.abs(gt_t - gnss_start_time))

    opts.initstate.vel = gt[gt_idx, 4:7].copy()
    opts.initstate.euler = gt[gt_idx, 7:10].copy() * (math.pi / 180)

    # Initial state std
    opts.initstate_std.pos = np.array([0.1, 0.1, 0.2])
    opts.initstate_std.vel = np.array([0.05, 0.05, 0.05])
    opts.initstate_std.euler = np.array([
        0.5 * math.pi / 180,
        0.5 * math.pi / 180,
        1.0 * math.pi / 180,
    ])
    opts.initstate_std.imuerror.gyrbias = np.array([
        50 * math.pi / 180 / 3600,
    ] * 3)
    opts.initstate_std.imuerror.accbias = np.array([250 * 1e-5] * 3)
    opts.initstate_std.imuerror.gyrscale = np.array([1000 * 1e-6] * 3)
    opts.initstate_std.imuerror.accscale = np.array([1000 * 1e-6] * 3)

    # IMU noise
    opts.imunoise.gyr_arw = np.array([IMU_NOISE['arw']] * 3) * math.pi / 180 / 60
    opts.imunoise.acc_vrw = np.array([IMU_NOISE['vrw']] * 3) / 60
    opts.imunoise.gyrbias_std = np.array([IMU_NOISE['gbstd']] * 3) * math.pi / 180 / 3600
    opts.imunoise.accbias_std = np.array([IMU_NOISE['abstd']] * 3) * 1e-5
    opts.imunoise.gyrscale_std = np.array([IMU_NOISE['gsstd']] * 3) * 1e-6
    opts.imunoise.accscale_std = np.array([IMU_NOISE['asstd']] * 3) * 1e-6
    opts.imunoise.corr_time = IMU_NOISE['corrtime'] * 3600

    opts.antlever = ANTLEVER

    engine = kfgins_py.GIEngine(opts)
    return engine


def run_kfgins(seq_name, gnss_source='F9P', adapter=None, verbose=True,
               use_mu_v=True, use_R=True, use_mu_w=False,
               theta_w=None, theta_v=None,
               r_ceiling_mult=0.0,
               mu_v_clip=2.0):
    """Run KF-GINS on one sequence, optionally with adapter and/or DR.

    When ``adapter`` is not None, the adapter is queried after every GNSS
    attempt, and its EMA-smoothed (mu_v, R) predictions are injected as
    KF-GINS overrides BEFORE the NEXT GNSS update.

    When ``theta_w`` and ``theta_v`` are not None, the DR-MMSE solver wraps
    the GNSS measurement update. The DR solver computes worst-case process
    and measurement noise within Wasserstein balls of radius theta_w/theta_v
    around the nominal noise. DR and adapter can be combined: the adapter
    predicts mu_v (bias) and R (nominal noise), and the DR solver adds
    robustness around that predicted R.

    The ``use_mu_v`` and ``use_R`` flags are diagnostics for isolating
    which prediction head is helping vs hurting. Set one to False to
    disable that override path (the adapter still runs; only the
    injection is skipped).
    """
    import kfgins_py
    from kfgins_features import _resample_imu_chunk, L_IMU

    if verbose:
        print(f'[{seq_name}] loading raw data...', flush=True)

    imu_data, gnss_data, gt_data = load_raw_data(seq_name, gnss_source)
    N_imu = len(imu_data)
    M_gnss = len(gnss_data)

    if verbose:
        print(f'[{seq_name}] IMU: {N_imu} samples, GNSS: {M_gnss} fixes, '
              f'GT: {len(gt_data)} samples', flush=True)

    # Align to GNSS start time, matching the C++ binary's `starttime` behavior
    # exactly (kf_gins.cpp:132-150):
    #   do { imu_cur = imufile.next(); } while (imu_cur.time < starttime);
    #   do { gnss    = gnssfile.next(); } while (gnss.time <= starttime);
    #   giengine.addImuData(imu_cur, true);  // FIRST imu at/after starttime
    #   giengine.addGnssData(gnss);          // FIRST gnss strictly after starttime
    # Note:
    #   (1) IMU priming = first sample with time >= starttime  (NOT the one
    #       before). The predecessor sample is never seen by the filter.
    #   (2) GNSS at exactly starttime is DISCARDED (`<=` in the skip loop).
    # Keep a separate `gnss_start_time` for NED origin interpolation so the
    # origin still anchors to the first GNSS epoch (for GT comparison) even
    # though that fix is not fed to the filter.
    gnss_start_time = gnss_data[0, 0]
    imu_start = int(np.searchsorted(imu_data[:, 0], gnss_start_time))
    imu_data = imu_data[imu_start:]
    N_imu = len(imu_data)

    # Skip the first GNSS if it sits exactly on starttime (matches C++ `<=`).
    gnss_skip = 0
    while gnss_skip < len(gnss_data) and gnss_data[gnss_skip, 0] <= gnss_start_time:
        gnss_skip += 1

    if verbose:
        print(f'[{seq_name}] aligned to GNSS t0={gnss_start_time:.1f}, '
              f'skipped {imu_start} IMU (priming = first imu>=t0), '
              f'skipped {gnss_skip} GNSS<=t0, '
              f'{N_imu} IMU remaining', flush=True)

    engine = create_engine(gnss_data, gt_data, gnss_start_time)

    # Reference geodetic origin for NED conversion
    # NED reference origin: interpolate GNSS geodetic at the GT start time.
    # The GT file's position is [0, 0, 0] at t = gt_data[0, 0], meaning the
    # GT NED frame is centered wherever the vehicle was at that moment.
    # We must use the SAME reference for our nav→NED conversion, otherwise
    # the GT and nav trajectories are in different NED frames, producing a
    # constant bias in the RMSE that scales with how much the vehicle moved
    # between gnss[0, 0] and gt[0, 0]. For building01/parking00 the vehicle
    # is moving at init, causing 10+ m of systematic bias.
    gt_start_t = gt_data[0, 0]
    ref_lat = np.interp(gt_start_t, gnss_data[:, 0],
                        gnss_data[:, 1]) * math.pi / 180
    ref_lon = np.interp(gt_start_t, gnss_data[:, 0],
                        gnss_data[:, 2]) * math.pi / 180
    ref_alt = float(np.interp(gt_start_t, gnss_data[:, 0], gnss_data[:, 3]))

    # Precompute earth radii for geodetic→NED conversion
    e1 = 0.00669437999014
    Rm = 6378137.0 * (1 - e1) / (1 - e1 * math.sin(ref_lat)**2)**1.5
    Rn = 6378137.0 / (1 - e1 * math.sin(ref_lat)**2)**0.5

    def geodetic_to_ned(lat_rad, lon_rad, alt):
        n = (lat_rad - ref_lat) * (Rm + ref_alt)
        e = (lon_rad - ref_lon) * (Rn + ref_alt) * math.cos(ref_lat)
        d = -(alt - ref_alt)
        return np.array([n, e, d])

    # Drive the filter
    j_gnss = gnss_skip  # GNSS pointer — start past any fix at/before starttime
    t_prev_gnss = imu_data[0, 0]
    positions = []
    times = []

    # Filter diagnostics: NIS and NEES per GNSS attempt
    nis_log = []       # (t_gnss, nis_value, dof=3)
    filter_std_h_log = []  # (t_gnss, sqrt(P[0,0]+P[1,1]))  -- per-step filter horizontal posterior std
    nees_pos_log = []  # (t_gnss, nees_pos, dof=3)

    # Derive rate-form IMU arrays once for adapter IMU-chunk extraction.
    # (Cheap to precompute; keeps per-step cost low.)
    if adapter is not None:
        imu_t_arr = imu_data[:, 0].astype(np.float64)
        dts = np.diff(imu_t_arr, prepend=imu_t_arr[0] - 0.005)
        dts = np.clip(dts, 4e-3, None)
        gyro_arr  = imu_data[:, 1:4].astype(np.float64) / dts[:, None]
        accel_arr = imu_data[:, 4:7].astype(np.float64) / dts[:, None]
        last_t_for_chunk = float(imu_t_arr[0])

    # CRITICAL: Prime the IMU pair before the main loop.
    # KF-GINS's addImuData does `imupre_ = imucur_; imucur_ = imu;`
    # On the very first call, imupre_ is uninitialized. The C++ binary
    # reads one IMU sample before entering the processing loop to set
    # imucur_ = first_sample. Then the second addImuData correctly
    # computes dt = imucur_.time - imupre_.time = imu[1].time - imu[0].time.
    # Without this priming, dt is garbage → immediate divergence.
    #
    # IMPORTANT: Only the priming call passes compensate=True. Subsequent
    # calls pass False because insPropagation internally calls imuCompensate
    # on imucur_ every step. The priming call must compensate because by the
    # time it has become imupre_, insPropagation expects it already compensated
    # (see gi_engine.cpp:177 comment "imupre has been compensated"). Passing
    # True to every call causes double compensation → over-corrected IMU
    # biases → indoor drift. This matches kf_gins.cpp:146 (true) vs :177 (default).
    prime_imu = kfgins_py.IMU()
    prime_imu.time = imu_data[0, 0]
    prime_imu.dt = 0.005  # 200 Hz nominal
    prime_imu.dtheta = imu_data[0, 1:4]
    prime_imu.dvel = imu_data[0, 4:7]
    engine.addImuData(prime_imu, True)

    # DR wrapper (if theta_w/theta_v provided)
    dr_wrapper = None
    if theta_w is not None and theta_v is not None:
        from dr_kfgins import DR_KFGINS
        dr_wrapper = DR_KFGINS(engine, theta_w, theta_v)
        if verbose:
            print(f'[{seq_name}] DR mode: theta_w={theta_w}, theta_v={theta_v}',
                  flush=True)

    t0 = time.time()
    for k in range(1, N_imu):
        # Feed IMU sample
        imu = kfgins_py.IMU()
        imu.time = imu_data[k, 0]
        imu.dt = imu_data[k, 0] - imu_data[k - 1, 0]
        imu.dtheta = imu_data[k, 1:4].copy()
        imu.dvel = imu_data[k, 4:7].copy()

        # Phase 12: Python-side mu_w correction (no C++ pybind change).
        # KF-GINS accepts raw IMU and subtracts its own bias estimate (ba,bg)
        # internally. The adapter's mu_w is an ADDITIONAL correction on top:
        # if the adapter detects a residual IMU bias the filter hasn't
        # absorbed yet, mu_w specifies the amount to subtract from the
        # IMU sample BEFORE the filter sees it. Multiply by dt so the
        # correction acts on the delta (dtheta, dvel) units.
        #
        # During adapter warmup (< T attempts), adapter.mu_w is None and
        # we skip the correction. --no_mu_w disables injection entirely
        # even when the adapter has output.
        if (use_mu_w and adapter is not None
                and getattr(adapter, 'mu_w', None) is not None):
            dt_step = float(imu.dt)
            mu_w_a = np.asarray(adapter.mu_w[0:3], dtype=np.float64)
            mu_w_g = np.asarray(adapter.mu_w[3:6], dtype=np.float64)
            imu.dvel   = imu.dvel   - mu_w_a * dt_step    # accel correction
            imu.dtheta = imu.dtheta - mu_w_g * dt_step    # gyro correction

        engine.addImuData(imu, False)

        # Check if any GNSS fixes fall in (t[k-1], t[k]].
        # Record metadata about the pending attempt BEFORE newImuProcess
        # so the adapter can extract features after the update.
        pending_gnss_t = None
        while j_gnss < M_gnss and gnss_data[j_gnss, 0] <= imu_data[k, 0]:
            gnss = kfgins_py.GNSS()
            gnss.time = gnss_data[j_gnss, 0]
            gnss.blh = np.array([
                gnss_data[j_gnss, 1] * math.pi / 180,
                gnss_data[j_gnss, 2] * math.pi / 180,
                gnss_data[j_gnss, 3],
            ])
            gnss.std = gnss_data[j_gnss, 4:7]
            gnss.isvalid = True
            engine.addGnssData(gnss)
            pending_gnss_t = float(gnss_data[j_gnss, 0])
            j_gnss += 1

        # Adapter override injection — set BEFORE newImuProcess so the
        # override hooks are live when gnssUpdate runs. Overrides auto-clear
        # after each gnssUpdate (gi_engine.cpp:358-359) so we re-set them
        # every time a GNSS attempt is pending. During warmup (< T=60
        # attempts), adapter.mu_v is None and we skip the override entirely.
        #
        # SIGN FLIP on mu_v: the adapter is trained to predict the ESKF-
        # convention measurement bias `mu_v ≈ z_raw - h_gt`. But KF-GINS's
        # innovation uses the opposite sign convention:
        #     dz = Dr * (antenna_pred - gnssdata.blh) = pred - z
        # and the override hook does `dz -= mu_v_override`. To cancel the
        # bias, we need dz_adjusted = dz + (z - h_gt) ≈ 0, which requires
        # mu_v_override = -(z - h_gt) = -adapter.mu_v. Flipping here keeps
        # the training pipeline (and the cached features/labels) unchanged
        # — the flip is purely an injection-time adjustment for KF-GINS's
        # dz sign convention (gi_engine.cpp:317).
        if adapter is not None and pending_gnss_t is not None and adapter.mu_v is not None:
            mu_v_inject = -adapter.mu_v  # sign flip for KF-GINS convention

            # Clip mu_v per axis to limit adapter influence in extreme cases.
            # Default 2.0 m is a safety rail tuned for i2Nav (drift-induced OOD
            # innovations on long indoor stretches). Lift to ~5-10 m for
            # UrbanNav-style multipath regimes where real GNSS biases reach
            # 5-30 m and a 2 m clip would discard most of the adapter signal.
            mu_v_inject = np.clip(mu_v_inject, -mu_v_clip, mu_v_clip)

            if use_mu_v:
                # Innovation-norm gating: the adapter was trained on features
                # from a near-GT filter (GT resets every 120 attempts). At
                # training time, innovations are small (<3m). At inference,
                # after long indoor segments (building01), the filter can drift
                # 10-50m, producing innovations of 20-200m. The adapter's mu_v
                # prediction is OUT-OF-DISTRIBUTION in this regime and applying
                # it causes systematic drift.
                #
                # Gate: only inject mu_v when the approximate innovation norm
                # is within the training distribution. The threshold 5m covers
                # ~95th percentile of training innovations (verified from cache
                # stats). Above this, let KF-GINS's native update handle it.
                INNOV_GATE_M = float(os.environ.get('KFGINS_INNOV_GATE_M', '3.0'))
                cur_state = engine.getNavState()
                pred_ned = geodetic_to_ned(cur_state.pos[0], cur_state.pos[1], cur_state.pos[2])
                gnss_ned = geodetic_to_ned(
                    gnss_data[j_gnss - 1, 1] * math.pi / 180,
                    gnss_data[j_gnss - 1, 2] * math.pi / 180,
                    gnss_data[j_gnss - 1, 3],
                )
                approx_innov_norm = np.linalg.norm(pred_ned - gnss_ned)
                if approx_innov_norm < INNOV_GATE_M:
                    engine.setOverrideMuV(mu_v_inject)

            if use_R:
                R_apply = adapter.R.copy()
                if r_ceiling_mult > 0:
                    # PSD-preserving ceiling: if the diagonal of R exceeds
                    # K × receiver_variance on any axis, scale the WHOLE
                    # matrix down by the max-axis ratio. This keeps ρ_ij
                    # exactly where the adapter put them — clipping only
                    # diagonals would violate ρ² ≤ 1 and KF-GINS would
                    # crash on the next Joseph-form update with negative
                    # covariance. Prior simple per-axis min() had this bug.
                    gnss_std_raw = gnss_data[j_gnss - 1, 4:7]
                    cap_diag = r_ceiling_mult * (gnss_std_raw ** 2)
                    diag_R = np.diag(R_apply)
                    ratios = diag_R / (cap_diag + 1e-12)
                    scale_down = float(max(1.0, ratios.max()))
                    if scale_down > 1.0:
                        R_apply = R_apply / scale_down
                engine.setOverrideR(R_apply)

        # Tell DR wrapper the original receiver-reported R for this GNSS
        # attempt BEFORE process_step, so it uses the unmodified R as
        # Sigma_v_hat (not the adapter-overridden R = double-robustification).
        if dr_wrapper is not None and pending_gnss_t is not None:
            gnss_std_raw = gnss_data[j_gnss - 1, 4:7]
            dr_wrapper.set_original_R(np.diag(gnss_std_raw ** 2))

        if dr_wrapper is not None:
            dr_wrapper.process_step()
        else:
            engine.newImuProcess()

        # After newImuProcess, if a GNSS attempt just ran, collect
        # diagnostics (NIS, NEES) and feed the adapter.
        if pending_gnss_t is not None:
            diag = engine.lastGNSSDiagnostics()
            if bool(diag.valid):
                # --- NIS: normalized innovation squared (from KF-GINS) ---
                nis_log.append((pending_gnss_t, float(diag.nis), 3))

                # --- NEES_pos: position estimation error vs covariance ---
                # Compare post-update filter position to interpolated GT.
                post_state = engine.getNavState()
                post_ned = geodetic_to_ned(
                    post_state.pos[0], post_state.pos[1], post_state.pos[2])
                gt_t_arr = gt_data[:, 0]
                if gt_t_arr[0] <= pending_gnss_t <= gt_t_arr[-1]:
                    gt_n = float(np.interp(pending_gnss_t, gt_t_arr, gt_data[:, 1]))
                    gt_e = float(np.interp(pending_gnss_t, gt_t_arr, gt_data[:, 2]))
                    gt_d = float(np.interp(pending_gnss_t, gt_t_arr, gt_data[:, 3]))
                    err_pos = np.array([post_ned[0] - gt_n,
                                        post_ned[1] - gt_e,
                                        post_ned[2] - gt_d])
                    P_full = np.array(engine.getCovariance())
                    P_pos = P_full[0:3, 0:3]
                    P_pos = 0.5 * (P_pos + P_pos.T)
                    try:
                        nees_pos = float(err_pos @ np.linalg.solve(P_pos, err_pos))
                    except np.linalg.LinAlgError:
                        nees_pos = float('nan')
                    nees_pos_log.append((pending_gnss_t, nees_pos, 3))
                    # Per-GNSS-update filter posterior horizontal std
                    # (sqrt(P_pos[0,0] + P_pos[1,1])). Useful for per-segment
                    # indicator analysis. Stored alongside NIS/NEES.
                    try:
                        filter_std_h_post_step = float(
                            np.sqrt(max(0.0, P_pos[0, 0]) + max(0.0, P_pos[1, 1])))
                    except Exception:
                        filter_std_h_post_step = float('nan')
                    # innovation horizontal magnitude (pre-update err_pos
                    # is post-update; the diag.nis already encodes the
                    # innovation chi^2 — we record it too for kurtosis/
                    # whiteness analyses)
                    filter_std_h_log.append((pending_gnss_t, filter_std_h_post_step))

                # --- Adapter step ---
                if adapter is not None:
                    navstate_for_adapter = post_state
                    pos_ned_for_adapter = post_ned
                    imu_chunk = _resample_imu_chunk(
                        imu_t_arr, accel_arr, gyro_arr,
                        last_t_for_chunk, pending_gnss_t, L_IMU,
                    )
                    adapter.step(
                        diag, navstate_for_adapter, pos_ned_for_adapter,
                        imu_chunk, pending_gnss_t,
                    )
                    last_t_for_chunk = pending_gnss_t

        # Record position every step (matching C++ binary output)
        state = engine.getNavState()
        pos_ned = geodetic_to_ned(state.pos[0], state.pos[1], state.pos[2])
        positions.append(pos_ned)
        times.append(imu_data[k, 0])

    wall = time.time() - t0
    positions = np.array(positions)
    times = np.array(times)

    # Compute ATE against GT — only within the GT time range.
    # np.interp clamps outside the range, which gives wrong errors for
    # times before GT starts or after GT ends.
    gt_t_min = gt_data[0, 0]
    gt_t_max = gt_data[-1, 0]
    valid = (times >= gt_t_min) & (times <= gt_t_max)
    times_v = times[valid]
    positions_v = positions[valid]

    gt_n = np.interp(times_v, gt_data[:, 0], gt_data[:, 1])
    gt_e = np.interp(times_v, gt_data[:, 0], gt_data[:, 2])
    gt_d = np.interp(times_v, gt_data[:, 0], gt_data[:, 3])

    err_n = positions_v[:, 0] - gt_n
    err_e = positions_v[:, 1] - gt_e
    err_d = positions_v[:, 2] - gt_d

    rmse = {
        'rmse_n': float(np.sqrt((err_n**2).mean())),
        'rmse_e': float(np.sqrt((err_e**2).mean())),
        'rmse_d': float(np.sqrt((err_d**2).mean())),
        'rmse_h': float(np.sqrt((err_n**2 + err_e**2).mean())),
        'rmse_3d': float(np.sqrt((err_n**2 + err_e**2 + err_d**2).mean())),
    }

    # Paper-comparable RMSE: decimate filter trajectory to GNSS epochs,
    # exclude epochs that fall inside any GNSS gap > GAP_THRESH (a fix is
    # missing — paper drops these via their availability%). Match the convention
    # used in Hsu et al. 2023 NAVIGATION Tab. 4–6 for cross-method comparison.
    GAP_THRESH = 5.0  # seconds — anything larger is considered a receiver dropout
    rmse_paper = {}
    if len(gnss_data) > 1:
        gnss_t = gnss_data[:, 0]
        # Per-GNSS-epoch error: NN-interpolate filter state onto each GNSS time
        # within the GT range. Drop epochs after the previous one by > GAP_THRESH.
        in_gt = (gnss_t >= gt_t_min) & (gnss_t <= gt_t_max)
        gt_n_g = np.interp(gnss_t, gt_data[:, 0], gt_data[:, 1])
        gt_e_g = np.interp(gnss_t, gt_data[:, 0], gt_data[:, 2])
        gt_d_g = np.interp(gnss_t, gt_data[:, 0], gt_data[:, 3])
        est_n_g = np.interp(gnss_t, times_v, positions_v[:, 0])
        est_e_g = np.interp(gnss_t, times_v, positions_v[:, 1])
        est_d_g = np.interp(gnss_t, times_v, positions_v[:, 2])
        en_g = est_n_g - gt_n_g; ee_g = est_e_g - gt_e_g; ed_g = est_d_g - gt_d_g
        gaps = np.diff(gnss_t, prepend=gnss_t[0])
        in_outage = (gaps > GAP_THRESH)
        keep = in_gt & (~in_outage)
        if keep.sum() > 0:
            rmse_paper = {
                'rmse_h': float(np.sqrt((en_g[keep]**2 + ee_g[keep]**2).mean())),
                'rmse_3d': float(np.sqrt((en_g[keep]**2 + ee_g[keep]**2
                                          + ed_g[keep]**2).mean())),
                'mean_h': float(np.sqrt(en_g[keep]**2 + ee_g[keep]**2).mean()),
                'median_h': float(np.median(np.sqrt(en_g[keep]**2 + ee_g[keep]**2))),
                'max_h': float(np.sqrt(en_g[keep]**2 + ee_g[keep]**2).max()),
                'n_epochs': int(keep.sum()),
                'n_epochs_total': int(len(gnss_t)),
                'availability_pct': float(100.0 * keep.sum() / len(gnss_t)),
                'gap_thresh_s': GAP_THRESH,
            }

    # NIS statistics (chi²₃ thresholds: 95%=7.815, 99%=11.345)
    nis_stats = {}
    if nis_log:
        nis_vals = np.array([x[1] for x in nis_log])
        nis_stats = {
            'nis_mean': float(nis_vals.mean()),
            'nis_median': float(np.median(nis_vals)),
            'nis_pass_95': float((nis_vals < 7.815).mean()),
            'nis_pass_99': float((nis_vals < 11.345).mean()),
            'nis_count': len(nis_vals),
        }

    # NEES_pos statistics (chi²₃ thresholds: same)
    nees_stats = {}
    if nees_pos_log:
        nees_vals = np.array([x[1] for x in nees_pos_log if np.isfinite(x[1])])
        if len(nees_vals) > 0:
            nees_stats = {
                'nees_pos_mean': float(nees_vals.mean()),
                'nees_pos_median': float(np.median(nees_vals)),
                'nees_pos_pass_95': float((nees_vals < 7.815).mean()),
                'nees_pos_pass_99': float((nees_vals < 11.345).mean()),
                'nees_pos_count': len(nees_vals),
            }

    parts = ['KF-GINS']
    if dr_wrapper is not None:
        parts.append(f'DR(tw={theta_w},tv={theta_v})')
    if adapter is not None:
        parts.append('adapter')
    tag = '+'.join(parts).ljust(30)
    if verbose:
        print(f'[{seq_name}] {tag} ATE-RMSE: '
              f'N={rmse["rmse_n"]:.3f}  E={rmse["rmse_e"]:.3f}  '
              f'D={rmse["rmse_d"]:.3f}  H={rmse["rmse_h"]:.3f}  '
              f'3D={rmse["rmse_3d"]:.3f}  m  '
              f'(wall={wall:.1f}s)', flush=True)
        if nis_stats:
            print(f'[{seq_name}]   NIS: mean={nis_stats["nis_mean"]:.2f}  '
                  f'median={nis_stats["nis_median"]:.2f}  '
                  f'pass95={nis_stats["nis_pass_95"]:.1%}  '
                  f'pass99={nis_stats["nis_pass_99"]:.1%}  '
                  f'(n={nis_stats["nis_count"]})', flush=True)
        if nees_stats:
            print(f'[{seq_name}]   NEES_pos: mean={nees_stats["nees_pos_mean"]:.2f}  '
                  f'median={nees_stats["nees_pos_median"]:.2f}  '
                  f'pass95={nees_stats["nees_pos_pass_95"]:.1%}  '
                  f'pass99={nees_stats["nees_pos_pass_99"]:.1%}  '
                  f'(n={nees_stats["nees_pos_count"]})', flush=True)

    return {
        'seq': seq_name,
        'gnss_source': gnss_source,
        'filter': 'kfgins',
        'adapter': adapter is not None,
        'dr': dr_wrapper is not None,
        'theta_w': theta_w,
        'theta_v': theta_v,
        'rmse': rmse,
        'rmse_paper': rmse_paper,
        'nis': nis_stats,
        'nees_pos': nees_stats,
        'wall_time_s': wall,
        'n_imu': N_imu,
        'n_gnss': M_gnss,
        # Per-timestep trajectory arrays (for plotting). Only filled when
        # the caller sets save_traj=True via run_kfgins_traj(). Otherwise
        # the JSON sidecar omits these to keep file sizes small.
        '_traj': {
            't': times_v,
            'est_ned': positions_v,
            'gt_ned': np.stack([gt_n, gt_e, gt_d], axis=1),
            'err_ned': np.stack([err_n, err_e, err_d], axis=1),
            # Per-GNSS-update logs (different cadence than IMU; arrays may be
            # empty if no valid GNSS update fired during the run).
            'gnss_t': np.array([x[0] for x in nis_log], dtype=float),
            'gnss_nis': np.array([x[1] for x in nis_log], dtype=float),
            'gnss_nees_pos': np.array([x[1] for x in nees_pos_log], dtype=float),
            'gnss_filter_std_h_post': np.array([x[1] for x in filter_std_h_log], dtype=float),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--gnss_source', type=str, default='F9P')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seqs', type=str, nargs='+', default=None,
                        help='Run multiple sequences')
    parser.add_argument('--no_mu_v', action='store_true',
                        help='Diagnostic: disable mu_v override injection')
    parser.add_argument('--no_R', action='store_true',
                        help='Diagnostic: disable R override injection')
    parser.add_argument('--mu_w', action='store_true',
                        help='(Phase 12) Enable Python-side IMU-bias '
                             'correction using adapter.mu_w. Subtracts '
                             'mu_w × dt from each IMU sample before '
                             'addImuData(). Default off (mu_w=0).')
    parser.add_argument('--save_traj', action='store_true',
                        help='Save per-timestep trajectory (t, est_ned, '
                             'gt_ned, err_ned) as .npz sidecar next to '
                             '--output. For plotting.')
    parser.add_argument('--theta_w', type=float, default=None,
                        help='DR process noise Wasserstein radius (enables DR mode)')
    parser.add_argument('--theta_v', type=float, default=None,
                        help='DR measurement noise Wasserstein radius (enables DR mode)')
    parser.add_argument('--r_ceiling_mult', type=float, default=0.0,
                        help='If >0, cap adapter R at K × receiver R per axis. '
                             'Prevents over-inflated R from making filter '
                             'dead-reckon via IMU bias drift.')
    parser.add_argument('--mu_v_clip', type=float, default=2.0,
                        help='Per-axis clip on injected mu_v (m). Default 2.0 '
                             '(safety rail for i2Nav drift-induced OOD '
                             'innovations). Raise to ~5-10 for UrbanNav-style '
                             'multipath regimes where real biases reach 5-30 m.')
    args = parser.parse_args()

    sequences = args.seqs if args.seqs else [args.seq]
    results = []

    # Load the adapter once (shared across sequences); its rolling buffers
    # are reset per sequence by re-instantiating inside the loop so
    # cross-sequence context doesn't leak.
    checkpoint = args.checkpoint
    if checkpoint is not None and not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint}')

    for seq in sequences:
        adapter = None
        if checkpoint is not None:
            # Re-instantiate per sequence so buffers start empty
            from theta_muse_adapter_kfgins import MeasNoiseKFGINSAdapter
            adapter = MeasNoiseKFGINSAdapter(checkpoint)
        result = run_kfgins(
            seq, args.gnss_source, adapter=adapter,
            use_mu_v=not args.no_mu_v,
            use_R=not args.no_R,
            use_mu_w=bool(args.mu_w),
            theta_w=args.theta_w,
            theta_v=args.theta_v,
            r_ceiling_mult=args.r_ceiling_mult,
            mu_v_clip=args.mu_v_clip,
        )
        results.append(result)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # Split _traj out: .json holds summaries; .npz holds the arrays.
        traj_list = []
        results_json = []
        for r in results:
            traj = r.pop('_traj', None)
            results_json.append(r)
            if traj is not None:
                traj_list.append(traj)
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f'Saved to {args.output}')

        if getattr(args, 'save_traj', False) and traj_list:
            npz_path = os.path.splitext(args.output)[0] + '.npz'
            # For multi-seq runs, prefix arrays with index; usually 1 seq
            # per invocation so this is fine.
            if len(traj_list) == 1:
                np.savez_compressed(npz_path, **traj_list[0])
            else:
                np.savez_compressed(npz_path, **{
                    f'seq{i}_{k}': v
                    for i, t in enumerate(traj_list)
                    for k, v in t.items()
                })
            print(f'Saved trajectory to {npz_path}')


if __name__ == '__main__':
    main()
