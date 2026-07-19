"""
DR-KF-GINS: Distributionally Robust measurement update for KF-GINS.

Uses the UNFACTORED DR-MMSE TAC solver (solve_dr_mmse_tac) which takes
the full (21×21) accumulated process noise Sigma_w directly, instead of
factored G_list. This avoids the discrete-time noise formula mismatch
between KF-GINS's `Qd = (Phi*G*Qc*G'*Phi' + G*Qc*G')*dt/2` and the
factored solver's simpler `sum(G_k Qc G_k' dt_k)`.

Key insight: KF-GINS already computes the correct Cov_prior through its
standard propagation. Since Cov_prior = APA + Sigma_w (where APA =
Phi_accum @ P_post_prev @ Phi_accum'), we can extract Sigma_w exactly:
    Sigma_w_hat = Cov_prior - APA

At theta_w = theta_v = 0, the DR solver reduces to the vanilla Kalman
update with the EXACT same prior — verified as a parity check.

Usage:
    dr = DR_KFGINS(engine, theta_w=0.5, theta_v=0.1)
    for k in range(1, N):
        engine.addImuData(imu, False)
        if gnss_pending:
            engine.addGnssData(gnss)
        dr.process_step()  # replaces engine.newImuProcess()
"""
import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'drekf_cpp', 'build')))

import dr_ekf_cpp  # noqa: E402


class DR_KFGINS:
    """Distributionally Robust wrapper for KF-GINS (unfactored solver).

    The engine must be in DR mode (engine.setDRMode(True)) so that
    gnssUpdate populates diagnostics but skips EKFUpdate, and
    newImuProcess skips stateFeedback.
    """

    def __init__(self, engine, theta_w, theta_v,
                 solver='fw_exact', fw_params=None):
        self.engine = engine
        self.theta_w = float(theta_w)
        self.theta_v = float(theta_v)
        self.solver = solver
        self.fw_params = fw_params or {
            'fw_max_iters': 200,
            'fw_exact_iters': 16,
        }
        # When adapter+DR are combined, the adapter overrides R in KF-GINS,
        # so lastRGnss() returns the adapter's R (already inflated). DR must
        # use the ORIGINAL receiver-reported R as its nominal Sigma_v_hat to
        # avoid double-robustification. The runner sets this per GNSS attempt.
        self._original_R = None  # set by runner before process_step()

        # Enable DR mode in the engine
        engine.setDRMode(True)

        # APA tracking: only need Phi_accum and P_post_last_meas
        self._Phi_accum = np.eye(21)
        self._P_post_last_meas = np.array(engine.getCovariance(), dtype=np.float64)

        # Warm-start for the DR solver
        self._warm_Sigma_w = None
        self._warm_Sigma_v = None

        # Counters
        self.dr_solve_count = 0
        self.dr_fallback_count = 0

    def process_step(self):
        """Call instead of engine.newImuProcess().

        Handles: IMU propagation, GNSS update (if any) via DR-MMSE,
        and state feedback.
        """
        self.engine.newImuProcess()

        # Accumulate Phi from this step's propagation records
        for rec in self.engine.propagationRecords():
            Phi_k = np.array(rec.Phi, dtype=np.float64)
            self._Phi_accum = Phi_k @ self._Phi_accum

        # If GNSS update occurred, apply DR
        if self.engine.lastCallHadGnssUpdate():
            self._apply_dr_update()

    def set_original_R(self, R):
        """Set the original receiver-reported R for this GNSS attempt.

        Called by the runner BEFORE process_step(). When an adapter overrides
        R, lastRGnss() returns the adapter's R. DR needs the original R as
        its nominal Sigma_v_hat to avoid double-robustification.
        """
        self._original_R = np.array(R, dtype=np.float64)

    def _apply_dr_update(self):
        """Compute DR-optimal measurement update and inject into engine."""
        # In DR mode, Cov_ is the pre-update prior (gnssUpdate skipped EKFUpdate)
        Cov_prior = np.array(self.engine.getCovariance(), dtype=np.float64)
        H   = np.array(self.engine.lastHGnss(), dtype=np.float64)     # (3, 21)
        dz  = np.array(self.engine.lastInnovGnss(), dtype=np.float64) # (3, 1)
        # Use original receiver R as nominal Sigma_v_hat (not adapter-overridden R)
        if self._original_R is not None:
            R = self._original_R
        else:
            R = np.array(self.engine.lastRGnss(), dtype=np.float64)   # (3, 3)
        if dz.ndim == 1:
            dz = dz.reshape(-1, 1)

        # Build APA = Phi_accum @ P_post_prev @ Phi_accum.T
        APA = self._Phi_accum @ self._P_post_last_meas @ self._Phi_accum.T
        APA = 0.5 * (APA + APA.T)  # enforce symmetry

        # Extract accumulated process noise from the actual prior:
        # Cov_prior = APA + Sigma_w, so Sigma_w = Cov_prior - APA
        Sigma_w_hat = Cov_prior - APA
        Sigma_w_hat = 0.5 * (Sigma_w_hat + Sigma_w_hat.T)
        # Project to PSD (numerical errors can make it slightly non-PSD)
        eigvals, eigvecs = np.linalg.eigh(Sigma_w_hat)
        if eigvals.min() < -1e-10:
            eigvals = np.maximum(eigvals, 0.0)
            Sigma_w_hat = eigvecs @ np.diag(eigvals) @ eigvecs.T

        try:
            result = dr_ekf_cpp.solve_dr_mmse_tac(
                APA, H, Sigma_w_hat, R,
                self.theta_w, self.theta_v,
                self.solver,
                self._warm_Sigma_w, self._warm_Sigma_v,
                **self.fw_params
            )

            if result.success:
                self.dr_solve_count += 1

                # DR-optimal Kalman update
                wc_Xprior  = result.wc_Xprior
                wc_Sigma_v = result.wc_Sigma_v
                S_star     = H @ wc_Xprior @ H.T + wc_Sigma_v
                S_star     = 0.5 * (S_star + S_star.T)
                K_dr       = wc_Xprior @ H.T @ np.linalg.inv(S_star)
                dx_dr      = K_dr @ dz                # (21, 1)

                # DR posterior covariance
                P_post_dr = result.wc_Xpost
                P_post_dr = 0.5 * (P_post_dr + P_post_dr.T)

                # Inject into engine
                self.engine.setCov(P_post_dr)
                self.engine.setDx(dx_dr.flatten())
                self.engine.doStateFeedback()
                self.engine.finalizeDR()  # pvapre_ = pvacur_ (corrected)

                # Reset tracking for next inter-measurement interval
                self._Phi_accum = np.eye(21)
                self._P_post_last_meas = P_post_dr.copy()

                # Update warm-start
                self._warm_Sigma_w = result.wc_Sigma_w.copy()
                self._warm_Sigma_v = result.wc_Sigma_v.copy()
                return
        except Exception as e:
            if self.dr_fallback_count < 3:
                import warnings
                warnings.warn(f"DR solver failed (fallback #{self.dr_fallback_count}): {e}")


        # Fallback: standard Kalman update
        self.dr_fallback_count += 1
        S = H @ Cov_prior @ H.T + R
        S = 0.5 * (S + S.T)
        K = Cov_prior @ H.T @ np.linalg.inv(S)
        dx = K @ dz
        I21 = np.eye(21)
        IKH = I21 - K @ H
        P_post = IKH @ Cov_prior @ IKH.T + K @ R @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        self.engine.setCov(P_post)
        self.engine.setDx(dx.flatten())
        self.engine.doStateFeedback()
        self.engine.finalizeDR()  # pvapre_ = pvacur_ (corrected)

        self._Phi_accum = np.eye(21)
        self._P_post_last_meas = P_post.copy()
