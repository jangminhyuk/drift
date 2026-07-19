'''
    DR-ESKF-TAC class: Distributionally Robust Error-State Kalman Filter (TAC variant)
    Subclasses ESKF, replacing only the measurement update with a
    TAC-style DR-MMSE estimator via the C++ Frank-Wolfe solver.

    TAC: Ambiguity set on process noise Q_w (6x6) and measurement noise Sigma_v (1x1).
    FW optimizes over (Q_w, Sigma_v).

    When theta_w = theta_v = 0, reduces exactly to vanilla ESKF.
'''
import numpy as np
import math
import sys
import os
from scipy import linalg
from pyquaternion import Quaternion

from eskf_class import ESKF, w_accxyz, w_gyro_rpy, DEG_TO_RAD, GRAVITY_MAGNITUDE

# Add drekf_cpp build directory to path
cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cwd, '..', 'drekf_cpp', 'build'))
import dr_ekf_cpp


class DR_ESKF_TAC(ESKF):
    def __init__(self, X0, q0, P0, K, theta_w=0.1, theta_v=0.1,
                 solver='fw_exact', fw_params=None, mu_w=None, mu_v=0.0):
        super().__init__(X0, q0, P0, K, mu_w=mu_w, mu_v=mu_v)

        # DR parameters
        self.theta_w = theta_w
        self.theta_v = theta_v
        self.solver = solver
        self.fw_params = fw_params or {}

        # APA decomposition tracking
        # Phi accumulates the product of Fx matrices between measurement updates
        self._Phi = np.eye(9)
        # G_list accumulates G_k = Phi_{N,k+1} @ Fi * dt_k matrices between meas updates
        self._G_list = []
        # Q_base: nominal noise in reduced 6x6 space (no dt factor — dt is in G_k)
        self._Q_base = np.diag([w_accxyz**2]*3 + [w_gyro_rpy**2]*3)
        # P_post at last accepted measurement
        self._P_post_last_meas = P0.copy()

        # Warm-start matrices (None = cold start)
        self._warm_Q_w = None
        self._warm_Sigma_v = None

        # NIS logging (override parent's list since we log from DR path too)
        self.nis_log = []

        # Diagnostics
        self.dr_solve_count = 0
        self.dr_fallback_count = 0
        self._last_dr_diagnostics = None  # populated per UWB update for external consumers

        # Attempt-level diagnostics for MUSE training
        self.attempt_log = []       # list of per-attempt dicts
        self._last_uwb_time = None  # timestamp of previous UWB attempt
        self._attempt_count = 0

    def set_Q_base(self, value):
        '''Set Q_base for both vanilla ESKF predict and DR solver APA.'''
        self.Q_base = value
        if value is not None:
            self._Q_base = np.asarray(value, dtype=float)

    def predict(self, imu, dt, imu_check, k):
        '''
        Override predict to track Phi and G_list for APA decomposition.
        Computes Fx BEFORE calling super().predict(), which is safe
        because super().predict(k) writes q_list[k] but reads q_list[k-1].
        '''
        # Compute Fx (state transition Jacobian) using q_list[k-1]
        if imu_check:
            omega_k = imu[3:] * DEG_TO_RAD + self.mu_w[3:6]
            f_k = imu[0:3] * GRAVITY_MAGNITUDE + self.mu_w[0:3]
            dw = omega_k * dt
            R_prev = Quaternion(self.q_list[k - 1, :]).rotation_matrix
            Fx = np.block([
                [np.eye(3),       dt * np.eye(3),  -0.5 * dt**2 * R_prev.dot(self.cross(f_k))],
                [np.zeros((3,3)), np.eye(3),       -dt * R_prev.dot(self.cross(f_k))          ],
                [np.zeros((3,3)), np.zeros((3,3)), linalg.expm(self.cross(dw)).T               ]
            ])
        else:
            Fx = np.eye(9)

        # Update APA decomposition tracking:
        # Propagate existing G_k matrices through this Fx
        for i in range(len(self._G_list)):
            self._G_list[i] = Fx.dot(self._G_list[i])
        # Append new G_k = Fi * dt  (9x6, dt absorbed into G_k)
        self._G_list.append(self.Fi * dt)
        self._Phi = Fx.dot(self._Phi)

        # Call vanilla ESKF predict (unchanged)
        super().predict(imu, dt, imu_check, k)

    def UWB_correct(self, uwb, anchor_position, k, t_uwb=None):
        '''
        Override UWB_correct to use DR-MMSE measurement update.
        Gates with nominal S first, then calls C++ solver for robust update.
        Falls back to vanilla ESKF on solver failure.

        Now also populates attempt_log with rich per-attempt diagnostics
        for MUSE training.  The optional `t_uwb` arg is the physical timestamp
        (used for dt_uwb); callers that don't need attempt_log can omit it.
        '''
        an_A = anchor_position[int(uwb[0]), :]
        an_B = anchor_position[int(uwb[1]), :]

        qk_pr = Quaternion(self.q_list[k])
        C_iv = qk_pr.rotation_matrix

        # Measurement model (same as parent)
        p_uwb = C_iv.dot(self.t_uv) + self.Xpr[k, 0:3].reshape(-1, 1)
        d_A = linalg.norm(an_A - np.squeeze(p_uwb))
        d_B = linalg.norm(an_B - np.squeeze(p_uwb))
        predicted = d_B - d_A
        err_uwb = uwb[2] - predicted - self.mu_v

        # Measurement Jacobian G (1 x 9)
        G = self.computeG_grad(an_A, an_B, self.t_uv, self.Xpr[k, 0:3], self.q_list[k])
        G_norm = float(linalg.norm(G))

        # Nominal gating (use nominal covariances)
        Q = self.std_uwb_tdoa**2
        M = np.squeeze(G.dot(self.Ppr[k]).dot(G.T) + Q)
        d_m = math.sqrt(err_uwb**2 / M)

        # Nominal innovation diagnostics for classical adaptive estimators
        # (kept separate from _last_dr_diagnostics, which MUSE consumes)
        self._last_innov_diagnostics = {
            'err_uwb': float(err_uwb),   # innovation after mu_v removal
            'S_nom': float(M),           # nominal innovation variance
            'R_used': float(Q),          # injected measurement variance
            'accepted': bool(d_m < 5),
        }

        trace_Ppr = float(np.trace(self.Ppr[k]))

        # --- Attempt diagnostics (always computed) ---
        self._attempt_count += 1
        dt_uwb = 0.0
        if t_uwb is not None and self._last_uwb_time is not None:
            dt_uwb = float(t_uwb - self._last_uwb_time)
        if t_uwb is not None:
            self._last_uwb_time = t_uwb

        attempt = {
            'attempt_idx': self._attempt_count,
            'k': k,
            't_uwb': t_uwb,
            'dt_uwb': dt_uwb,
            'meas_raw': float(uwb[2]),
            'anchor_A_id': int(uwb[0]),
            'anchor_B_id': int(uwb[1]),
            'anchor_A_pos': an_A.copy(),
            'anchor_B_pos': an_B.copy(),
            'predicted_meas': float(predicted),
            'err_uwb': float(err_uwb),
            'G_norm': G_norm,
            'gating_metric': float(d_m),
            'gate_threshold': 5.0,
            'trace_Ppr': trace_Ppr,
        }

        if d_m < 5:
            # Build solver inputs
            APA = self._Phi.dot(self._P_post_last_meas).dot(self._Phi.T)
            APA = 0.5 * (APA + APA.T)  # symmetrize

            Sigma_v_hat = np.array([[Q]])  # 1x1
            C_mat = G.reshape(1, 9)

            try:
                result = dr_ekf_cpp.solve_dr_mmse_tac_factored(
                    APA, C_mat, self._G_list, self._Q_base, Sigma_v_hat,
                    self.theta_w, self.theta_v,
                    self.solver,
                    self._warm_Q_w, self._warm_Sigma_v,
                    **self.fw_params
                )

                if result.success:
                    # Robust Kalman update
                    wc_Xprior = result.wc_Xprior
                    wc_Sigma_v = result.wc_Sigma_v
                    S_star = C_mat.dot(wc_Xprior).dot(C_mat.T) + wc_Sigma_v  # 1x1
                    S_star_val = float(S_star[0, 0])
                    # NIS using robust innovation covariance
                    nis_value = float(err_uwb**2 / S_star_val)
                    self.nis_log.append((k, nis_value, 1))
                    Kk = (wc_Xprior.dot(C_mat.T) / S_star_val).reshape(-1, 1)  # 9x1
                    derror = Kk * err_uwb

                    # Update posterior covariance
                    self.Ppo[k] = result.wc_Xpost
                    self.Ppo[k] = 0.5 * (self.Ppo[k] + self.Ppo[k].T)

                    # Update nominal states
                    self.Xpo[k] = self.Xpr[k] + np.squeeze(derror[0:6])

                    # Quaternion injection (same as parent)
                    dq_k = Quaternion(self.zeta(np.squeeze(derror[6:])))
                    qk_po = qk_pr * dq_k
                    self.q_list[k] = np.array([qk_po.w, qk_po.x, qk_po.y, qk_po.z])

                    # Reset tracking for next inter-measurement interval
                    self._Phi = np.eye(9)
                    self._G_list = []
                    self._P_post_last_meas = self.Ppo[k].copy()

                    # Update warm-start (6x6 Q_w instead of 9x9 Sigma_w)
                    self._warm_Q_w = result.wc_Q_w.copy()
                    self._warm_Sigma_v = result.wc_Sigma_v.copy()

                    self.dr_solve_count += 1

                    trace_Ppo = float(np.trace(self.Ppo[k]))
                    corr_norm = float(linalg.norm(np.squeeze(derror)))
                    gain_ratio = (trace_Ppr - trace_Ppo) / trace_Ppr if trace_Ppr > 0 else 0.0
                    z_norm = float(err_uwb / np.sqrt(S_star_val)) if S_star_val > 0 else 0.0

                    self._last_dr_diagnostics = {
                        'success': True,
                        'k': k,
                        'err_uwb': float(err_uwb),
                        'wc_Sigma_v': wc_Sigma_v.copy(),
                        'S_star': S_star_val,
                        'wc_Q_w': result.wc_Q_w.copy(),
                        'derror': np.squeeze(derror),
                        'iterations': result.iterations,
                        'trace_Ppr': trace_Ppr,
                        'trace_Ppo': trace_Ppo,
                        'gain_ratio': gain_ratio,
                        'corr_norm': corr_norm,
                        'z_norm': z_norm,
                        'G_norm': G_norm,
                        'predicted_meas': float(predicted),
                        'gating_metric': float(d_m),
                        'anchor_A_id': int(uwb[0]),
                        'anchor_B_id': int(uwb[1]),
                        'anchor_A_pos': an_A.copy(),
                        'anchor_B_pos': an_B.copy(),
                        'meas_raw': float(uwb[2]),
                    }

                    # Attempt log entry -- accepted, solver success
                    attempt.update({
                        'accepted': True,
                        'S_star': S_star_val,
                        'derror': np.squeeze(derror).copy(),
                        'z_norm': z_norm,
                        'trace_Ppo': trace_Ppo,
                        'gain_ratio': gain_ratio,
                        'corr_norm': corr_norm,
                        'wc_Q_w': result.wc_Q_w.copy(),
                        'wc_Sigma_v': wc_Sigma_v.copy(),
                        'iterations': result.iterations,
                    })
                    self.attempt_log.append(attempt)
                    return

            except Exception:
                pass

            # Fallback to vanilla ESKF on solver failure
            self.dr_fallback_count += 1
            super().UWB_correct(uwb, anchor_position, k)
            # Reset tracking after vanilla update too
            self._Phi = np.eye(9)
            self._G_list = []
            self._P_post_last_meas = self.Ppo[k].copy()

            trace_Ppo = float(np.trace(self.Ppo[k]))
            self._last_dr_diagnostics = {
                'success': False,
                'k': k,
                'err_uwb': float(err_uwb),
                'trace_Ppr': trace_Ppr,
                'trace_Ppo': trace_Ppo,
                'G_norm': G_norm,
                'predicted_meas': float(predicted),
                'gating_metric': float(d_m),
                'anchor_A_id': int(uwb[0]),
                'anchor_B_id': int(uwb[1]),
                'anchor_A_pos': an_A.copy(),
                'anchor_B_pos': an_B.copy(),
                'meas_raw': float(uwb[2]),
            }

            # Attempt log -- accepted but solver failed (fallback)
            attempt.update({
                'accepted': True,
                'S_star': float(Q),
                'derror': np.zeros(9),
                'z_norm': float(err_uwb / np.sqrt(Q)) if Q > 0 else 0.0,
                'trace_Ppo': trace_Ppo,
                'gain_ratio': (trace_Ppr - trace_Ppo) / trace_Ppr if trace_Ppr > 0 else 0.0,
                'corr_norm': 0.0,
                'wc_Q_w': None,
                'wc_Sigma_v': None,
                'iterations': 0,
            })
            self.attempt_log.append(attempt)
        else:
            # Gating rejected: keep prior, do NOT reset tracking
            self._last_dr_diagnostics = {
                'success': False,
                'k': k,
                'err_uwb': float(err_uwb),
                'trace_Ppr': trace_Ppr,
                'trace_Ppo': trace_Ppr,
                'G_norm': G_norm,
                'predicted_meas': float(predicted),
                'gating_metric': float(d_m),
                'anchor_A_id': int(uwb[0]),
                'anchor_B_id': int(uwb[1]),
                'anchor_A_pos': an_A.copy(),
                'anchor_B_pos': an_B.copy(),
                'meas_raw': float(uwb[2]),
            }
            self.Xpo[k] = self.Xpr[k]
            self.Ppo[k] = self.Ppr[k]

            # Attempt log -- rejected
            attempt.update({
                'accepted': False,
                'S_star': float(Q),
                'derror': np.zeros(9),
                'z_norm': float(err_uwb / np.sqrt(Q)) if Q > 0 else 0.0,
                'trace_Ppo': trace_Ppr,
                'gain_ratio': 0.0,
                'corr_norm': 0.0,
                'wc_Q_w': None,
                'wc_Sigma_v': None,
                'iterations': 0,
            })
            self.attempt_log.append(attempt)


# Backward-compatible alias
DR_ESKF = DR_ESKF_TAC
