"""
KF-GINS feature extraction matching the training-cache schema.

At each GNSS attempt, this module builds the same (Xgnss[16], Ximu[200,6],
Xodom[22]) tokens that theta_sweep_worker_gnss.py emits for the vanilla
ESKF_GNSS reference. The adapter (ThetaMUSEModelGNSS) can then be trained on
KF-GINS-derived caches and used at inference time via main_kfgins.py.

Design notes
------------
* Feature semantics MUST be consistent between training cache and inference.
  They do NOT have to match the ESKF feature definitions byte-for-byte; the
  adapter is retrained for KF-GINS, and the normalization stats are cache-
  specific.
* `trace_Ppr` uses the full 21x21 prior trace (not just the 3x3 position
  block). Magnitude is slightly larger than ESKF's 15x15 trace but the
  network learns to handle it after normalization.
* `gain_ratio` is computed as trace(K @ S @ K.T) / trace(Cov_prior), which
  is the trace-sense magnitude of the covariance reduction from the update.
  This is a proxy for the ESKF's (tr(Ppr) - tr(Ppo))/tr(Ppr) — not the same
  scalar, but consistent within the KF-GINS pipeline.
* `corr_norm` is ||K @ innov||, i.e. the L2 norm of the full 21-element error
  state correction at this update. Matches the ESKF definition exactly.
* `predicted_meas` is converted to local NED via an externally-supplied
  `geodetic_to_ned` closure so both training and inference use the same
  origin as the RMSE evaluator in main_kfgins.py.
* `gnss_std_in` uses the measurement noise stored in diag.gnss_std_in
  (i.e. what KF-GINS actually used), not the raw file value — these agree
  unless the adapter overrides R, in which case the original std is still
  the raw file value (diag.gnss_std_in is populated BEFORE the override is
  applied).
"""
import math
import numpy as np


D_GNSS = 16   # matches theta_sweep_worker_gnss.py
D_ODOM = 22
L_IMU  = 200


def _resample_imu_chunk(imu_t, imu_a, imu_w, t_prev, t_curr, L_out=L_IMU):
    """Slice IMU samples in (t_prev, t_curr] and resample to L_out × 6.

    Mirrors theta_sweep_worker_gnss._get_imu_chunk. Returns zeros if
    fewer than 2 samples fall in the window.
    """
    mask = (imu_t > t_prev) & (imu_t <= t_curr)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return np.zeros((L_out, 6), dtype=np.float64)
    t_chunk = imu_t[idx]
    chunk = np.column_stack([imu_a[idx], imu_w[idx]])
    t_resamp = np.linspace(t_chunk[0], t_chunk[-1], L_out)
    out = np.zeros((L_out, 6), dtype=np.float64)
    for ch in range(6):
        out[:, ch] = np.interp(t_resamp, t_chunk, chunk[:, ch])
    return out


def build_gnss_token(diag, pred_meas_ned, dt_gnss, accepted):
    """Assemble the 16-D GNSS token from a populated GNSSDiagnostics struct.

    Parameters
    ----------
    diag : kfgins_py.GNSSDiagnostics
        Must have innov, S, Cov_prior, K, gnss_std_in, nis populated
        (i.e. gnssUpdate just finished for this attempt).
    pred_meas_ned : (3,) np.ndarray
        Predicted GNSS measurement in local NED meters (computed by the
        caller using the same NED origin as the rest of the pipeline).
    dt_gnss : float
        Seconds since the previous GNSS attempt.
    accepted : float
        1.0 if this attempt was accepted (updates applied), 0.0 otherwise.
        KF-GINS's vanilla gnssUpdate has no gating, so this is always 1.0
        unless callers add their own gating layer.

    Returns
    -------
    (16,) float64 numpy array, schema matching theta_sweep_worker_gnss.py
    lines 245-256.
    """
    innov       = np.asarray(diag.innov, dtype=np.float64).reshape(3)
    S           = np.asarray(diag.S, dtype=np.float64)
    Cov_prior   = np.asarray(diag.Cov_prior, dtype=np.float64)
    gnss_std_in = np.asarray(diag.gnss_std_in, dtype=np.float64).reshape(3)
    pred        = np.asarray(pred_meas_ned, dtype=np.float64).reshape(3)

    nis       = float(diag.nis)
    z_norm    = math.sqrt(max(nis, 0.0))
    trace_S   = float(np.trace(S))
    trace_Ppr = float(np.trace(Cov_prior))
    sigma_h   = float(np.sqrt(max(Cov_prior[0, 0] + Cov_prior[1, 1], 0.0)))
    gating    = z_norm   # Mahalanobis distance — KF-GINS has no explicit gate

    return np.array([
        innov[0], innov[1], innov[2],
        z_norm,
        math.log(max(trace_S, 1e-12)),
        float(dt_gnss),
        float(accepted),
        gnss_std_in[0], gnss_std_in[1], gnss_std_in[2],
        pred[0], pred[1], pred[2],
        gating,
        math.log(max(trace_Ppr / max(trace_S, 1e-12), 1e-12)),
        math.log(max(sigma_h, 1e-12)),
    ], dtype=np.float64)


def build_odom_token(diag, navstate, pos_ned, imu_chunk):
    """Assemble the 22-D odom/state token.

    Parameters
    ----------
    diag : kfgins_py.GNSSDiagnostics
        Populated with Cov_prior, K, S, dx.
    navstate : kfgins_py.NavState
        Current nav state (pos geodetic, vel NED, euler rad, imuerror).
        Must be fetched right after the GNSS update (engine.getNavState()).
    pos_ned : (3,) np.ndarray
        The nav position converted to local NED meters — supplied by the
        caller because the NED origin is known only at the runner level.
    imu_chunk : (L_IMU, 6) np.ndarray
        The IMU chunk already computed for this attempt (accel[0:3], gyro[3:6]).

    Returns
    -------
    (22,) float64 numpy array, schema matching theta_sweep_worker_gnss.py
    lines 262-287.
    """
    Cov_prior = np.asarray(diag.Cov_prior, dtype=np.float64)
    S         = np.asarray(diag.S, dtype=np.float64)
    K         = np.asarray(diag.K, dtype=np.float64)
    dx        = np.asarray(diag.dx, dtype=np.float64).reshape(-1)

    trace_Ppr = float(np.trace(Cov_prior))
    log_trace_Ppr = math.log(max(trace_Ppr, 1e-12))

    # gain_ratio (proxy): trace-sense fractional covariance reduction.
    # For the standard update Ppo = Ppr - K S K.T, so
    # trace(Ppr) - trace(Ppo) = trace(K S K.T).
    trace_KSK = float(np.trace(K @ S @ K.T))
    gain_ratio = trace_KSK / max(trace_Ppr, 1e-12)

    # corr_norm: L2 norm of full state correction (= K @ innov)
    corr_norm = float(np.linalg.norm(dx))

    vel = np.asarray(navstate.vel, dtype=np.float64).reshape(3)
    euler = np.asarray(navstate.euler, dtype=np.float64).reshape(3)  # (roll, pitch, yaw) rad
    ba = np.asarray(navstate.imuerror.accbias, dtype=np.float64).reshape(3)
    bg = np.asarray(navstate.imuerror.gyrbias, dtype=np.float64).reshape(3)

    speed = float(np.linalg.norm(vel))

    chunk = np.asarray(imu_chunk, dtype=np.float64)
    if chunk.shape != (L_IMU, 6):
        # Defensive: should never happen if caller uses _resample_imu_chunk
        chunk = np.zeros((L_IMU, 6), dtype=np.float64)
    an = np.linalg.norm(chunk[:, :3], axis=1)
    gn = np.linalg.norm(chunk[:, 3:], axis=1)

    return np.array([
        pos_ned[0], pos_ned[1], pos_ned[2],
        vel[0], vel[1], vel[2],
        float(euler[0]), float(euler[1]), float(euler[2]),
        log_trace_Ppr, gain_ratio, corr_norm, speed,
        ba[0], ba[1], ba[2],
        bg[0], bg[1], bg[2],
        float(an.mean()), float(gn.mean()), float(gn.max()),
    ], dtype=np.float64)
