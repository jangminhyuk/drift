'''
    Classical (statistical) adaptive measurement-noise estimators for the
    UWB ESKF / DR-ESKF pipeline.

    Drop-in alternatives to the learned MUSE adapter at the theta_callback
    seam (same factory contract as make_muse_callback in main_muse_theta.py):
    the callback sets eskf.mu_v and eskf.std_uwb_tdoa as side effects and
    optionally returns fixed (theta_w, theta_v) DR margins.

    Process noise (Q, mu_w) is left at the filter defaults: classical joint
    Q+R estimation is known to be unreliable, and the measurement side is
    where the UWB NLOS mismatch lives (matching the mu_v/R-centric MUSE
    deployment).

    Estimators (both update only on gate-accepted attempts):

    1. 'sage_husa' -- Modified Sage-Husa adaptive estimator with exponential
       fading factor b (Sage & Husa 1969; the fading-memory recursion below
       is the standard formulation in recent UWB/INS adaptive-filter
       literature). With zh_k = z_k - h(x_pr), eps_k = zh_k - mu_v^{k-1},
       and d_k = (1 - b) / (1 - b^{k+1}):
           mu_v^k = (1 - d_k) mu_v^{k-1} + d_k zh_k
           R^k    = (1 - d_k) R^{k-1}    + d_k (eps_k^2 - G P_pr G^T)

    2. 'vbakf' -- Variational Bayesian adaptive Kalman filter for the scalar
       measurement-noise variance (Sarkka & Nummenmaa 2009; inverse-Wishart
       generalization in Huang et al. 2018, IEEE TAC). The scalar R has an
       inverse-Gamma posterior IG(alpha, beta) propagated with forgetting
       factor rho:
           alpha <- rho * alpha + 1/2
           beta  <- rho * beta  + 1/2 * E[(z - h(x))^2 | posterior]
       with E[...] = eps_post^2 + G P_po G^T evaluated from the nominal
       posterior quantities (lag-one VB: one VB iteration per measurement
       at the adapter seam; the filter update itself is untouched).
           R^k = beta / alpha
       VB-AKF estimates the variance only; mu_v stays 0.

    Both adapters read eskf._last_innov_diagnostics, populated by
    ESKF.UWB_correct and DR_ESKF_TAC.UWB_correct with the *nominal*
    innovation statistics, so the recursions are identical regardless of
    whether the deployed filter is vanilla or DR.
'''
import math

# Same clamp range as the MUSE adapter (theta_muse_adapter.py)
R_MIN = 0.001
R_MAX = 10.0
MU_V_MAX = 2.0   # |mu_v| safety clamp [m]
N_WARMUP = 20    # accepted updates before estimates are injected
R_DEFAULT = 0.05  # eskf_class.py default std_uwb_tdoa**2


class SageHusaAdapter:
    '''Modified Sage-Husa recursive estimator of (mu_v, R).'''

    name = 'sage_husa'

    def __init__(self, b=0.98, R0=R_DEFAULT,
                 theta_w_fixed=0.01, theta_v_fixed=0.01):
        self.b = float(b)
        self.theta_w_fixed = theta_w_fixed
        self.theta_v_fixed = theta_v_fixed

        self._n_accepted = 0   # accepted-update counter (k in d_k)
        self._mu_v_hat = 0.0   # internal bias estimate
        self._R_hat = float(R0)
        self._R0 = float(R0)
        self._applied_mu_v = 0.0  # bias actually injected at last update

        self.prediction_log = []
        self._step_count = 0

    @property
    def mu_v(self):
        return self._mu_v_hat

    @property
    def R(self):
        return self._R_hat

    def step(self, eskf):
        '''Update estimates from the just-completed UWB correction.
        Returns (mu_v, R) to inject for the NEXT update.'''
        self._step_count += 1
        diag = getattr(eskf, '_last_innov_diagnostics', None)

        if diag is not None and diag['accepted']:
            # Recover raw innovation zh = z - h (err_uwb had applied mu_v
            # subtracted, which may differ from the internal estimate
            # during warmup).
            zh = diag['err_uwb'] + self._applied_mu_v
            eps = zh - self._mu_v_hat
            GPG = max(diag['S_nom'] - diag['R_used'], 0.0)

            d_k = (1.0 - self.b) / (1.0 - self.b ** (self._n_accepted + 1))
            self._mu_v_hat = (1.0 - d_k) * self._mu_v_hat + d_k * zh
            self._mu_v_hat = max(-MU_V_MAX, min(MU_V_MAX, self._mu_v_hat))
            self._R_hat = (1.0 - d_k) * self._R_hat + d_k * (eps * eps - GPG)
            self._R_hat = max(R_MIN, min(R_MAX, self._R_hat))
            self._n_accepted += 1

        warmup = self._n_accepted < N_WARMUP
        mu_v_out = 0.0 if warmup else self._mu_v_hat
        R_out = self._R0 if warmup else self._R_hat
        self._applied_mu_v = mu_v_out

        self.prediction_log.append({
            'step': self._step_count,
            'mu_v': float(mu_v_out), 'R': float(R_out),
            'warmup': warmup,
        })
        return mu_v_out, R_out


class VBAKFAdapter:
    '''Variational Bayes inverse-Gamma estimator of scalar R (mu_v = 0).'''

    name = 'vbakf'

    def __init__(self, rho=0.99, alpha0=2.0, R0=R_DEFAULT,
                 theta_w_fixed=0.01, theta_v_fixed=0.01):
        self.rho = float(rho)
        self.theta_w_fixed = theta_w_fixed
        self.theta_v_fixed = theta_v_fixed

        # IG(alpha, beta) prior with mean-style estimate R = beta/alpha = R0
        self._alpha = float(alpha0)
        self._beta = float(alpha0) * float(R0)
        self._n_accepted = 0
        self._R0 = float(R0)

        self.prediction_log = []
        self._step_count = 0

    @property
    def mu_v(self):
        return 0.0

    @property
    def R(self):
        return max(R_MIN, min(R_MAX, self._beta / self._alpha))

    def step(self, eskf):
        '''Update IG(alpha, beta) from the just-completed UWB correction.
        Returns (mu_v, R) to inject for the NEXT update.'''
        self._step_count += 1
        diag = getattr(eskf, '_last_innov_diagnostics', None)

        if diag is not None and diag['accepted']:
            eps = diag['err_uwb']
            S = diag['S_nom']
            R_used = diag['R_used']
            GPG = max(S - R_used, 0.0)
            # Nominal posterior quantities (scalar Kalman algebra):
            # post-fit residual = eps * R/S, G P_po G^T = GPG * R/S
            eps_post = eps * R_used / S
            GPpoG = GPG * R_used / S

            # Forgetting (spread) then measurement update
            self._alpha = self.rho * self._alpha + 0.5
            self._beta = self.rho * self._beta \
                + 0.5 * (eps_post * eps_post + GPpoG)
            self._n_accepted += 1

        warmup = self._n_accepted < N_WARMUP
        R_out = self._R0 if warmup else self.R

        self.prediction_log.append({
            'step': self._step_count,
            'mu_v': 0.0, 'R': float(R_out),
            'warmup': warmup,
        })
        return 0.0, R_out


CLASSICAL_ADAPTERS = {
    'sage_husa': SageHusaAdapter,
    'vbakf': VBAKFAdapter,
}


def make_classical_callback(method, theta_w_fixed=0.01, theta_v_fixed=0.01,
                            return_theta=True, sh_b=0.98, vb_rho=0.99,
                            R0=R_DEFAULT):
    '''Create a theta_callback for run_estimation using a classical
    adaptive estimator. Mirrors make_muse_callback's contract:

    The callback sets eskf.mu_v and eskf.std_uwb_tdoa (side effects) and
    returns fixed (theta_w, theta_v) for the DR margin, or None when
    return_theta=False (predictor + vanilla ESKF mode). Q_base and mu_w
    are NOT touched (filter defaults).

    Returns (callback_fn, adapter, diag_log).
    '''
    if method == 'sage_husa':
        adapter = SageHusaAdapter(b=sh_b, R0=R0,
                                  theta_w_fixed=theta_w_fixed,
                                  theta_v_fixed=theta_v_fixed)
    elif method == 'vbakf':
        adapter = VBAKFAdapter(rho=vb_rho, R0=R0,
                               theta_w_fixed=theta_w_fixed,
                               theta_v_fixed=theta_v_fixed)
    else:
        raise ValueError('Unknown classical method: %s' % method)

    diag_log = []

    def callback(eskf, k, t_uwb_now, imu_raw, t_imu_raw):
        mu_v, R = adapter.step(eskf)
        eskf.mu_v = mu_v
        eskf.std_uwb_tdoa = math.sqrt(max(R, 1e-6))
        if return_theta:
            return adapter.theta_w_fixed, adapter.theta_v_fixed
        return None

    return callback, adapter, diag_log
