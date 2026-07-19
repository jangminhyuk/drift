'''
    Online rolling-buffer inference adapter for the MUSE measurement noise
    predictor (mu_v + R regression version).

    Maintains rolling buffers of UWB, IMU, and odom tokens. Once the buffer
    reaches window length T, runs the model forward, predicts mu_v (bias)
    and R (variance), and applies EMA smoothing.

    Usage:
        adapter = MeasNoiseMUSEAdapter('results/meas_noise_muse.pt')
        ...
        for k in range(1, K):
            eskf.predict(...)
            if uwb_check:
                eskf.UWB_correct(...)
                imu_chunk = get_imu_chunk(...)
                mu_v, R = adapter.step(eskf, k, imu_chunk, t_uwb)
                eskf.mu_v = mu_v
                eskf.std_uwb_tdoa = math.sqrt(max(R, 1e-6))
'''
import math
import numpy as np
import torch
from collections import deque
from pyquaternion import Quaternion

R_MIN = 0.001
R_MAX = 10.0


class MeasNoiseMUSEAdapter:
    def __init__(self, checkpoint_path, device='cpu', ema_alpha=0.3,
                 theta_w_fixed=0.01, theta_v_fixed=0.01,
                 muse_interval=1.0):
        from theta_muse_model import ThetaMUSEModel

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.cfg = ckpt['config']
        # Backward compat: old checkpoints lack backbone_type
        if not hasattr(self.cfg, 'backbone_type'):
            self.cfg.backbone_type = 'mamba' if getattr(self.cfg, 'use_mamba', True) else 'transformer'
        self.model = ThetaMUSEModel(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # Whether this is an e2e model (outputs Q and mu_w too)
        self._is_e2e = (self.cfg.task == 'e2e')

        # Normalization stats
        self.uwb_mean = ckpt['uwb_mean']
        self.uwb_std = ckpt['uwb_std']
        self.odom_mean = ckpt['odom_mean']
        self.odom_std = ckpt['odom_std']
        self.imu_mean = ckpt['imu_mean']
        self.imu_std = ckpt['imu_std']

        # Fixed theta values (DR margin, not adapted)
        self.theta_w_fixed = theta_w_fixed
        self.theta_v_fixed = theta_v_fixed

        # Default Q for warmup (match eskf_class.py defaults)
        self.Q_default = np.diag([4.0, 4.0, 4.0, 0.01, 0.01, 0.01])

        self.T = self.cfg.T
        self.ema_alpha = ema_alpha

        # Rolling buffers
        self._buf_uwb = deque(maxlen=self.T)
        self._buf_imu = deque(maxlen=self.T)
        self._buf_odom = deque(maxlen=self.T)

        # EMA state
        self._ema_mu_v = None
        self._ema_R = None
        self._ema_Q = None
        self._ema_mu_w = None

        # Running NIS statistics (EMA) for odom token features
        self._running_nis_mean = 1.0
        self._running_accept_rate = 1.0
        self._nis_ema_alpha = 0.1

        # MUSE call interval (seconds); reuse last prediction in between
        self.muse_interval = muse_interval
        self._last_muse_call_time = -float('inf')

        # History for logging
        self.prediction_log = []
        self._step_count = 0
        self._last_uwb_time = None

    @property
    def mu_v(self):
        return self._ema_mu_v

    @property
    def R(self):
        return self._ema_R

    def _extract_uwb_token(self, eskf, dt_uwb):
        '''Extract D_uwb=16 UWB token from eskf diagnostics.'''
        diag = eskf._last_dr_diagnostics
        if diag is None:
            return np.zeros(self.cfg.d_uwb)

        err_uwb = diag.get('err_uwb', 0.0)
        S_val = diag.get('S_star', eskf.std_uwb_tdoa**2)
        z_norm = diag.get('z_norm', err_uwb / math.sqrt(S_val + 1e-12))
        logS = math.log(S_val + 1e-12)
        accepted = 1.0 if diag.get('success', False) else 0.0
        gating_metric = diag.get('gating_metric', 0.0)
        G_norm = diag.get('G_norm', 0.0)
        trace_Ppr = diag.get('trace_Ppr', 0.0)
        log_trace_Ppr_S = math.log(trace_Ppr / (S_val + 1e-12) + 1e-12)

        # Anchor 3D positions (replaces integer IDs)
        an_A = diag.get('anchor_A_pos', np.zeros(3))
        an_B = diag.get('anchor_B_pos', np.zeros(3))

        return np.array([
            z_norm, logS, diag.get('meas_raw', 0.0), dt_uwb, accepted,
            an_A[0], an_A[1], an_A[2],
            an_B[0], an_B[1], an_B[2],
            diag.get('predicted_meas', 0.0), gating_metric,
            G_norm, log_trace_Ppr_S, err_uwb,
        ], dtype=np.float64)

    def _extract_odom_token(self, eskf, k, imu_chunk):
        '''Extract D_odom=19 odom token from eskf state.'''
        diag = eskf._last_dr_diagnostics or {}

        pos = eskf.Xpo[k, 0:3]
        vel = eskf.Xpo[k, 3:6]

        q = eskf.q_list[k]
        qobj = Quaternion(q)
        R = qobj.rotation_matrix
        pitch = math.asin(max(-1, min(1, -R[2, 0])))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

        trace_Ppr = diag.get('trace_Ppr', float(np.trace(eskf.Ppr[k])))
        log_trace_Ppr = math.log(trace_Ppr + 1e-12)
        gain_ratio = diag.get('gain_ratio', 0.0)
        corr_norm = diag.get('corr_norm', 0.0)
        speed = float(np.linalg.norm(vel))

        if imu_chunk is not None and len(imu_chunk) > 0:
            acc_mean = imu_chunk[:, :3].mean(axis=0)
            gyro_mean = imu_chunk[:, 3:].mean(axis=0)
        else:
            acc_mean = np.zeros(3)
            gyro_mean = np.zeros(3)

        return np.concatenate([
            pos, vel, [roll, pitch, yaw],
            [log_trace_Ppr, gain_ratio, corr_norm, speed],
            acc_mean, gyro_mean,
        ]).astype(np.float64)

    def step(self, eskf, k, imu_chunk, t_uwb):
        '''Process one UWB attempt. Returns (mu_v, R, Q, mu_w).

        For backward-compatible (meas_noise) checkpoints, Q and mu_w
        are returned as defaults (Q_default and zeros).

        Args:
            eskf: DR_ESKF_TAC instance (after UWB_correct)
            k: timestep index
            imu_chunk: (L, 6) IMU samples for this interval, or None
            t_uwb: physical UWB timestamp

        Returns:
            mu_v: predicted measurement bias (float)
            R: predicted measurement variance (float)
            Q: (6, 6) process noise covariance (numpy)
            mu_w: (6,) process noise bias (numpy)
        '''
        self._step_count += 1

        # Compute dt
        dt_uwb = 0.0
        if self._last_uwb_time is not None:
            dt_uwb = float(t_uwb - self._last_uwb_time)
        self._last_uwb_time = t_uwb

        # Update running NIS statistics from last diagnostics
        diag = eskf._last_dr_diagnostics
        if diag is not None:
            S_val = diag.get('S_star', eskf.std_uwb_tdoa**2)
            err_uwb = diag.get('err_uwb', 0.0)
            nis_val = err_uwb**2 / (S_val + 1e-12)
            accepted_val = 1.0 if diag.get('success', False) else 0.0
            alpha = self._nis_ema_alpha
            self._running_nis_mean = (1 - alpha) * self._running_nis_mean \
                + alpha * nis_val
            self._running_accept_rate = (1 - alpha) * \
                self._running_accept_rate + alpha * accepted_val

        # Extract tokens
        uwb_tok = self._extract_uwb_token(eskf, dt_uwb)
        odom_tok = self._extract_odom_token(eskf, k, imu_chunk)

        # IMU chunk: ensure shape (L, 6)
        if imu_chunk is None or len(imu_chunk) == 0:
            imu_tok = np.zeros((self.cfg.L_imu, 6))
        else:
            imu_tok = imu_chunk.copy()
            if imu_tok.shape[0] != self.cfg.L_imu:
                t_in = np.linspace(0, 1, imu_tok.shape[0])
                t_out = np.linspace(0, 1, self.cfg.L_imu)
                out = np.zeros((self.cfg.L_imu, 6))
                for ch in range(6):
                    out[:, ch] = np.interp(t_out, t_in, imu_tok[:, ch])
                imu_tok = out

        self._buf_uwb.append(uwb_tok)
        self._buf_imu.append(imu_tok)
        self._buf_odom.append(odom_tok)

        # Warmup: return defaults until buffer is full
        if len(self._buf_uwb) < self.T:
            mu_v = 0.0
            R = eskf.std_uwb_tdoa**2
            Q = self.Q_default.copy()
            mu_w = np.zeros(6)
            self.prediction_log.append({
                'step': self._step_count,
                'mu_v': mu_v, 'R': R,
                'warmup': True,
            })
            return mu_v, R, Q, mu_w

        # Skip model call if interval hasn't elapsed; reuse cached values
        t_now = float(t_uwb)
        if (t_now - self._last_muse_call_time < self.muse_interval
                and self._ema_mu_v is not None):
            return (self._ema_mu_v, self._ema_R,
                    self._ema_Q.copy(), self._ema_mu_w.copy())
        self._last_muse_call_time = t_now

        # Build batch tensors
        xuwb = np.array(list(self._buf_uwb))
        ximu = np.array(list(self._buf_imu))
        xodom = np.array(list(self._buf_odom))

        # Normalize
        xuwb = (xuwb - self.uwb_mean) / self.uwb_std
        xodom = (xodom - self.odom_mean) / self.odom_std
        ximu = (ximu - self.imu_mean) / self.imu_std

        # To tensors, add batch dim
        xuwb_t = torch.from_numpy(xuwb.astype(np.float32)).unsqueeze(0).to(self.device)
        ximu_t = torch.from_numpy(ximu.astype(np.float32)).unsqueeze(0).to(self.device)
        xodom_t = torch.from_numpy(xodom.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                xuwb_t,
                ximu_t if self.cfg.use_imu else None,
                xodom_t if self.cfg.use_odom else None,
            )

        if self._is_e2e:
            # E2E model returns (mu_v, R, Q, mu_w)
            mu_v_raw, R_raw, Q_raw, mu_w_raw = outputs
            mu_v = float(mu_v_raw[0])
            R = float(R_raw[0])
            Q_np = Q_raw[0].cpu().numpy()       # (6, 6)
            mu_w_np = mu_w_raw[0].cpu().numpy()  # (6,)
        else:
            # meas_noise model returns (mu_v, log_R)
            mu_v_raw, log_R_raw = outputs
            mu_v = float(mu_v_raw[0])
            R = float(torch.clamp(torch.exp(log_R_raw[0]), R_MIN, R_MAX))
            Q_np = self.Q_default.copy()
            mu_w_np = np.zeros(6)

        # EMA smoothing on all outputs
        if self._ema_mu_v is not None:
            alpha = self.ema_alpha
            mu_v = (1 - alpha) * self._ema_mu_v + alpha * mu_v
            R = (1 - alpha) * self._ema_R + alpha * R
            Q_np = (1 - alpha) * self._ema_Q + alpha * Q_np
            mu_w_np = (1 - alpha) * self._ema_mu_w + alpha * mu_w_np
        self._ema_mu_v = mu_v
        self._ema_R = R
        self._ema_Q = Q_np.copy()
        self._ema_mu_w = mu_w_np.copy()

        self.prediction_log.append({
            'step': self._step_count,
            'mu_v': mu_v, 'R': R,
            'Q_acc_diag': np.diag(Q_np[:3, :3]).astype(float).tolist(),
            'Q_gyro_diag': np.diag(Q_np[3:, 3:]).astype(float).tolist(),
            'mu_w_acc': mu_w_np[:3].astype(float).tolist(),
            'mu_w_gyro': mu_w_np[3:].astype(float).tolist(),
            'warmup': False,
        })

        return mu_v, R, Q_np, mu_w_np


# Backward compatibility alias
ThetaMUSEAdapter = MeasNoiseMUSEAdapter
