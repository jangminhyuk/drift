"""
KF-GINS-flavored rolling-buffer inference adapter for the learned GNSS
noise predictor (ThetaMUSEModelGNSS).

Mirrors scripts/gnss_imu/adaptive/theta_muse_adapter_gnss.py but reads
features from the pybind11-wrapped KF-GINS engine instead of our ESKF:

    diag = engine.lastGNSSDiagnostics()  -> innov, S, Cov_prior, K, dx, ...
    navstate = engine.getNavState()      -> pos(geodetic), vel, euler, imuerror

and uses kfgins_features.build_{gnss,odom}_token to build the 16-D / 22-D
tokens in the same layout as the training cache.

Design
------
The model/normalization loading, EMA smoothing, warmup behavior, and
throttling logic are copied verbatim from MeasNoiseGnssAdapter so the
two adapters behave identically at the inference surface. Only the
token extraction differs.

Override flow (important)
-------------------------
KF-GINS auto-clears mu_v/R overrides inside gnssUpdate (gi_engine.cpp:358),
so the inference runner must set overrides RIGHT BEFORE newImuProcess on
any iteration that is expected to trigger a GNSS update. The runner
pattern in main_kfgins.py:

    engine.addImuData(imu, False)
    if gnss_pending_this_step and adapter.mu_v is not None:
        engine.setOverrideMuV(adapter.mu_v)
        engine.setOverrideR(adapter.R)
    engine.newImuProcess()
    if gnss_was_just_processed:
        mu_v, R_pred, _, _ = adapter.step(engine, navstate, pos_ned, imu_chunk, t_gnss)
        # next iteration uses mu_v/R_pred via setOverride*

During warmup (< T=60 attempts), adapter.step returns the defaults
(zero mu_v, identity-ish R) and the caller simply does not set overrides,
letting KF-GINS run with its native measurement noise.
"""
import math
import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from theta_muse_model_gnss import ThetaMUSEModelGNSS, ThetaMUSEConfigGNSS  # noqa: E402
from kfgins_features import (                                              # noqa: E402
    build_gnss_token, build_odom_token, _resample_imu_chunk, L_IMU,
)


# Defaults during warmup (same values as MeasNoiseGnssAdapter)
_DEFAULT_Q_DIAG = [0.05**2]*3 + [5e-4**2]*3
_DEFAULT_R_DIAG = [1.0, 1.0, 4.0]


class MeasNoiseKFGINSAdapter:
    """Rolling-buffer inference adapter driving KF-GINS override hooks.

    Usage inside main_kfgins.py run_kfgins loop:

        adapter = MeasNoiseKFGINSAdapter(checkpoint_path)
        ...
        for k in range(1, N):
            engine.addImuData(imu_k, False)

            if gnss_pending and adapter.mu_v is not None:
                engine.setOverrideMuV(adapter.mu_v)
                engine.setOverrideR(adapter.R)

            if gnss_pending:
                engine.addGnssData(gnss_k)

            engine.newImuProcess()

            if gnss_was_just_processed:
                diag = engine.lastGNSSDiagnostics()
                navstate = engine.getNavState()
                pos_ned = geodetic_to_ned(navstate.pos)
                imu_chunk = resample(...)
                adapter.step(diag, navstate, pos_ned, imu_chunk, t_gnss)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        ema_alpha: float = 0.3,
        muse_interval: float = 1.0,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.cfg: ThetaMUSEConfigGNSS = ckpt['config']
        self.model = ThetaMUSEModelGNSS(self.cfg).to(self.device)
        # strict=False: backward-compatible loading across R head redesigns.
        # Old checkpoints (v1/v2) have _R_log_scale_default which the new
        # model doesn't; new checkpoints may have different head_R shapes.
        # Missing keys get random-init (use --no_R with old checkpoints).
        self.model.load_state_dict(ckpt['model_state'], strict=False)
        self.model.eval()

        # Normalization stats (the trainer stores them under historical
        # `uwb_*` keys — the field name is just legacy).
        self.gnss_mean = ckpt['uwb_mean']
        self.gnss_std  = ckpt['uwb_std']
        self.odom_mean = ckpt['odom_mean']
        self.odom_std  = ckpt['odom_std']
        self.imu_mean  = ckpt['imu_mean']
        self.imu_std   = ckpt['imu_std']

        self.Q_default    = np.diag(_DEFAULT_Q_DIAG)
        self.R_default    = np.diag(_DEFAULT_R_DIAG)
        self.mu_v_default = np.zeros(3)
        self.mu_w_default = np.zeros(6)

        self.T = self.cfg.T
        self.ema_alpha     = float(ema_alpha)
        self.muse_interval = float(muse_interval)

        # Rolling buffers
        self._buf_gnss = deque(maxlen=self.T)
        self._buf_imu  = deque(maxlen=self.T)
        self._buf_odom = deque(maxlen=self.T)

        # EMA state (None until the first non-warmup call)
        self._ema_mu_v: Optional[np.ndarray] = None
        self._ema_R:    Optional[np.ndarray] = None
        self._ema_Q:    Optional[np.ndarray] = None
        self._ema_mu_w: Optional[np.ndarray] = None

        self._last_muse_call_time = -float('inf')
        self.prediction_log = []
        self._step_count = 0
        self._last_gnss_time = None

    # ---------- public properties (read by the runner) ----------
    @property
    def mu_v(self):  return self._ema_mu_v
    @property
    def R(self):     return self._ema_R
    @property
    def Q(self):     return self._ema_Q
    @property
    def mu_w(self):  return self._ema_mu_w

    # ---------- main step ----------
    def step(self, diag, navstate, pos_ned, imu_chunk, t_gnss):
        """Process one GNSS attempt; returns (mu_v[3], R[3,3], Q[6,6], mu_w[6]).

        Parameters
        ----------
        diag : kfgins_py.GNSSDiagnostics
            Fresh diagnostics from engine.lastGNSSDiagnostics() — must be
            valid (diag.valid == True).
        navstate : kfgins_py.NavState
            Post-update state from engine.getNavState().
        pos_ned : (3,) np.ndarray
            navstate.pos converted to local NED by the caller.
        imu_chunk : (L_IMU, 6) np.ndarray
            IMU samples in (t_prev_attempt, t_gnss], already resampled to
            L_IMU points. Columns [0:3] = accel, [3:6] = gyro.
        t_gnss : float
            Timestamp of this GNSS attempt (seconds of week).
        """
        self._step_count += 1

        dt_gnss = 0.0
        if self._last_gnss_time is not None:
            dt_gnss = float(t_gnss - self._last_gnss_time)
        self._last_gnss_time = t_gnss

        # Compute predicted measurement in local NED (same convention as
        # kfgins_cache_worker uses for the training cache).
        # The cache worker computes pred_meas_ned via euler→R; here we
        # read it off Xodom would be circular, so we recompute from the
        # builder directly (kfgins_features does not take pred_meas — the
        # caller supplies it).
        #
        # For simplicity we use KF-GINS's own `predicted_meas` field via
        # the Python-computed version: pos_ned + R_cbn @ lever_arm, where
        # R_cbn comes from navstate.euler (ZYX convention, kfgins_features
        # does NOT do this step — we do it here to share the rotation math
        # between cache worker and adapter).
        from main_kfgins import ANTLEVER  # reuse the same lever
        pred_meas_ned = _euler_pos_to_antenna_ned(pos_ned, navstate.euler)

        # Build tokens (RAW std in cols 7:10 — log-transform applied below)
        gnss_tok_raw = build_gnss_token(diag, pred_meas_ned, dt_gnss, accepted=1.0)
        odom_tok     = build_odom_token(diag, navstate, pos_ned, imu_chunk)

        # Log-transform std columns (7:10) BEFORE normalization, matching
        # the trainer's dataloader __getitem__ at line 426 of
        # theta_muse_trainer_gnss.py.
        gnss_tok = gnss_tok_raw.copy()
        gnss_tok[7:10] = np.log(np.clip(gnss_tok[7:10], 0, None) + 1e-3)

        # Ensure IMU chunk is the expected shape
        if imu_chunk is None or len(imu_chunk) == 0:
            imu_tok = np.zeros((L_IMU, 6), dtype=np.float64)
        else:
            imu_tok = np.asarray(imu_chunk, dtype=np.float64)
            if imu_tok.shape[0] != L_IMU:
                t_in  = np.linspace(0, 1, imu_tok.shape[0])
                t_out = np.linspace(0, 1, L_IMU)
                rs = np.zeros((L_IMU, 6))
                for ch in range(6):
                    rs[:, ch] = np.interp(t_out, t_in, imu_tok[:, ch])
                imu_tok = rs

        self._buf_gnss.append(gnss_tok)
        self._buf_imu.append(imu_tok)
        self._buf_odom.append(odom_tok)

        # Warmup
        if len(self._buf_gnss) < self.T:
            mu_v = self.mu_v_default.copy()
            R    = self.R_default.copy()
            Q    = self.Q_default.copy()
            mu_w = self.mu_w_default.copy()
            self.prediction_log.append({
                'step': self._step_count,
                'warmup': True,
            })
            return mu_v, R, Q, mu_w

        # Throttle: only call the model every muse_interval seconds
        t_now = float(t_gnss)
        if (t_now - self._last_muse_call_time < self.muse_interval
                and self._ema_mu_v is not None):
            return (self._ema_mu_v.copy(), self._ema_R.copy(),
                    self._ema_Q.copy(),    self._ema_mu_w.copy())
        self._last_muse_call_time = t_now

        # Build batch tensors
        x_gnss = np.array(list(self._buf_gnss))
        x_imu  = np.array(list(self._buf_imu))
        x_odom = np.array(list(self._buf_odom))

        # Normalize with training stats
        x_gnss = (x_gnss - self.gnss_mean) / self.gnss_std
        x_odom = (x_odom - self.odom_mean) / self.odom_std
        x_imu  = (x_imu  - self.imu_mean)  / self.imu_std

        x_gnss_t = torch.from_numpy(x_gnss.astype(np.float32)).unsqueeze(0).to(self.device)
        x_imu_t  = torch.from_numpy(x_imu.astype(np.float32)).unsqueeze(0).to(self.device)
        x_odom_t = torch.from_numpy(x_odom.astype(np.float32)).unsqueeze(0).to(self.device)

        # Raw gnss_std at this attempt — the R head uses it to scale
        # R = diag(std² + floor²) * exp(scale_pred). Pulled from the
        # diagnostics, not from the log-transformed token.
        last_std_raw = np.asarray(diag.gnss_std_in, dtype=np.float32).reshape(3)
        last_std_t = torch.from_numpy(last_std_raw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu_v_raw, R_raw, Q_raw, mu_w_raw = self.model(
                x_gnss_t,
                x_imu_t  if self.cfg.use_imu else None,
                x_odom_t if self.cfg.use_odom else None,
                last_std=last_std_t,
            )

        mu_v = mu_v_raw[0].cpu().numpy().astype(np.float64)   # (3,)
        R    = R_raw[0].cpu().numpy().astype(np.float64)       # (3, 3)
        Q    = Q_raw[0].cpu().numpy().astype(np.float64)       # (6, 6)
        mu_w = mu_w_raw[0].cpu().numpy().astype(np.float64)    # (6,)

        # R floor removed: the model config now constrains R_log_scale_min=0.0
        # (inflate-only), so the network can never predict R below the receiver
        # std². The old 50% floor is redundant and was masking the over-tightening
        # problem on KF-GINS (21-state filter is more sensitive than 15-state ESKF).

        # EMA smoothing
        if self._ema_mu_v is not None:
            a = self.ema_alpha
            mu_v = (1 - a) * self._ema_mu_v + a * mu_v
            R    = (1 - a) * self._ema_R    + a * R
            Q    = (1 - a) * self._ema_Q    + a * Q
            mu_w = (1 - a) * self._ema_mu_w + a * mu_w
        self._ema_mu_v = mu_v.copy()
        self._ema_R    = R.copy()
        self._ema_Q    = Q.copy()
        self._ema_mu_w = mu_w.copy()

        self.prediction_log.append({
            'step': self._step_count,
            'mu_v': mu_v.tolist(),
            'R_diag': float(np.trace(R)),
            'Q_diag': np.diag(Q).tolist(),
            'mu_w': mu_w.tolist(),
            'warmup': False,
        })
        return mu_v, R, Q, mu_w


# ---------- helpers ----------

def _euler_pos_to_antenna_ned(pos_ned, euler_rad):
    """KF-GINS ZYX euler → cbn @ lever_arm, then add to pos_ned.

    Matches kfgins_cache_worker._euler_to_cbn_kfgins so cache and
    inference share the same antenna-position formula.
    """
    # Inline imports to avoid a circular dependency with main_kfgins
    from main_kfgins import ANTLEVER

    roll, pitch, yaw = float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2])
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    R_cbn = Rz @ Ry @ Rx
    return np.asarray(pos_ned, dtype=np.float64) + R_cbn @ ANTLEVER
