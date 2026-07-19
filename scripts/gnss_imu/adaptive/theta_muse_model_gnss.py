"""
GNSS-flavored ThetaMUSE model.

Reuses scripts/estimation/theta_muse_model.py:ThetaMUSEModel verbatim
(encoders, modality embeddings, Mamba/Transformer/GRU backbone, statistical
pooling) — see plan §2 for the rationale that the backbone is measurement-
agnostic. We only swap the output heads for GNSS:

  Head           UWB ('e2e')          GNSS ('e2e_gnss')
  ─────────────  ───────────────────  ────────────────────────────────────
  mu_v           1 (scalar TDOA bias) 3 (vector position bias in NED)
  R              1 (scalar variance)  3×3 (full PSD matrix via LDL params, 6)
  Q              6×6 LDL (12 params)  6×6 LDL (12 params)   ← reused
  mu_w           6 (acc + gyro bias)  6 (acc + gyro bias)   ← reused

The new R head outputs 6 LDL parameters (3 strict-lower + 3 log-diag), same
parametrization as one of the parent's Q sub-blocks, then reuses the parent's
`ldl_to_psd(params, block_size=3)` helper.

Sole cross-package import — the model file in scripts/estimation/ is NEVER
edited per plan §2.
"""
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make scripts/estimation importable for the parent class
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ESTIMATION_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'estimation'))
sys.path.insert(0, ESTIMATION_DIR)

from theta_muse_model import (   # noqa: E402
    ThetaMUSEConfig,
    ThetaMUSEModel,
    ScaledHead,
    ldl_to_psd,
)


@dataclass
class ThetaMUSEConfigGNSS(ThetaMUSEConfig):
    """GNSS-flavored config. Defaults match the plan §3.2 token sizes."""
    # Token dimensions (note: parent uses `d_uwb` for "the per-attempt
    # measurement-side token". We keep that name and just override its size.)
    # D_gnss = 16: innov(3) + z_norm + log_trace_S + dt + accepted + std(3)
    #              + pred(3) + gating + log_ratio + log_sigma_h
    # D_odom = 22: pos(3) + vel(3) + rpy(3) + log_trace_Ppr + gain_ratio
    #              + corr_norm + speed + ba(3) + bg(3) + |a|_mean + |g|_mean + |g|_max
    d_uwb: int = 16       # GNSS per-attempt token (see adapter)
    d_odom: int = 22      # GNSS odom token (extended with bias states)

    # GNSS at 1 Hz, IMU 200 Hz → 200 IMU samples per inter-attempt interval
    L_imu: int = 200

    # Rolling window length in GNSS attempts. The trainer passes T explicitly
    # through args.T; this default applies only to direct construction.
    T: int = 20

    # New task name (parent's forward() ignores this; we handle it ourselves)
    task: str = 'e2e_gnss'

    # ---- R head: RESIDUAL per-axis log-variance correction on receiver std.
    # The R head predicts a per-axis delta in log-space on top of the
    # receiver-reported std² (which is a strong prior for measurement noise):
    #     R_pred = diag(std_n², std_e², std_d²) * diag(exp(δ_n), exp(δ_e), exp(δ_d))
    # Equivalently: log(R_diag) = log(std² + floor²) + δ
    #
    # The receiver's std is used DIRECTLY in the architecture as a prior,
    # not just as an input feature. The network learns the RESIDUAL:
    #   δ = 0  → R = receiver std² (baseline, no correction)
    #   δ > 0  → inflate (receiver underestimates, e.g., urban multipath)
    #   δ < 0  → tighten (receiver overestimates, e.g., indoor std=3000m)
    # Each axis (N, E, D) has its own δ, so the model can independently
    # adjust horizontal vs vertical noise.
    #
    # Zero-init head → δ=0 → R = receiver std² → identical to baseline filter.
    # Inflate-only constraint (Option A): the adapter can never tighten R
    # below the receiver-reported std². Physical motivation — consumer-grade
    # receivers routinely *under*-estimate noise in degraded conditions
    # (multipath, indoor, canyon) but very rarely *over*-estimate it. Letting
    # the adapter drive δ<0 was a dominant failure mode in Phase 3-6 (R
    # tightened in the wrong places → filter over-confident → continuous-run
    # drift blew up). Setting floor=0 is a hard inductive bias that removes
    # that failure mode entirely without losing expressivity for the helpful
    # direction (inflation).
    R_delta_min: float =  0.0   # exp(0) = 1.0×   (never tighten below receiver std²)
    R_delta_max: float = +3.0   # exp(+3) = 20×   (inflate to 20× receiver std²)
    R_floor_m:   float = 0.1    # floor on receiver std to prevent log(0)

    # ---- LDL off-diagonal parameterization ----
    # R = L · D · Lᵀ, where
    #   D    = diag(std² + floor²) · diag(exp(δ))             ← current behavior
    #   L    = I + strict_lower(ℓ_21, ℓ_31, ℓ_32)             ← new cross-term
    # Off-diagonal ℓ_ij are tanh-soft-bounded to [-L_limit, +L_limit].
    # L_limit=1.0 lets |ρ_ij| approach 1 (any correlation sign/strength),
    # but bounds diagonal inflation: max R_ii ≤ sum(D_jj for j≤i). This is
    # enough expressivity to capture the multipath/canyon cross-correlation
    # structure DR's Σ_v solver produces, while preventing pathological
    # near-singular R. Zero-init of head_R's last Linear (see __init__)
    # makes ℓ start at 0 ⇒ R = diag(std²·exp(δ)) = current behavior.
    R_offdiag_limit: float = 1.0

    # GNSS Q clamps (separate accel and gyro because they have different scales).
    # Without these, the untrained Q head can drive Ppr to huge values over
    # thousands of predict steps → numerical blowup (observed: NaN loss on
    # the e2e smoke run before clamping was added).
    Q_acc_min:  float = 1e-6   # m²/s⁴   (σ_a ≥ 1 mm/s²)
    Q_acc_max:  float = 1.0    # m²/s⁴   (σ_a ≤ 1 m/s²)
    Q_gyro_min: float = 1e-10  # rad²/s² (σ_g ≥ 1e-5 rad/s)
    Q_gyro_max: float = 1e-2   # rad²/s² (σ_g ≤ 0.1 rad/s)

    # Tanh soft-clamp limits for mu_v (3-vec, GNSS position bias) and
    # mu_w (6-vec, IMU bias offset). These bound physically-implausible
    # predictions over a long training run; tanh keeps gradient flow smooth
    # so the optimizer never gets stuck at the boundary.
    #
    # Important: these do NOT affect the "ignore this sensor" semantic,
    # which is governed by R (large R → low Kalman gain → ignored). The
    # clamps only prevent the model from claiming the IMU has a 5 m/s²
    # bias or that the GNSS has a 100 m systematic offset, both of which
    # are physically impossible for the hardware in i2Nav-Robot.
    # Reduced from 10 m after sup_v1 diagnostic showed spurious mu_v
    # predictions up to 14 m on clean sequences (aux loss dominated clean NLL
    # at lambda_noise_label=1.0). Physical plausibility for ground-vehicle
    # GNSS: NLOS biases in the i2Nav-Robot dataset are at most ~15 m per the
    # audit of building00. The soft tanh means values near the limit are
    # still reachable if strongly indicated; clean data just can't drift far.
    mu_v_limit_h:    float = 5.0     # m, horizontal GNSS pos bias
    mu_v_limit_d:    float = 5.0     # m, vertical
    mu_w_limit_acc:  float = 0.05    # m/s²,  100× ADIS bias instability
    mu_w_limit_gyro: float = 0.005   # rad/s, 100× ADIS gyro bias spec


class ThetaMUSEModelGNSS(ThetaMUSEModel):
    """ThetaMUSE with GNSS-flavored output heads.

    Shares everything with the parent EXCEPT the prediction heads. The
    parent's __init__ already builds the encoders/backbone/pooling and the
    e2e heads (which output a scalar mu_v and a scalar R that we'd never
    use). We let the parent build those (small wasted parameters), then
    override them with the correct GNSS heads in our own __init__.
    """

    def __init__(self, cfg: ThetaMUSEConfigGNSS):
        # Force the parent to build under the 'e2e' task so it constructs
        # the heads we need to override (and not the 'theta' classification
        # heads). After super().__init__, we replace the heads we care about.
        cfg_for_parent = ThetaMUSEConfig(**{
            **{k: getattr(cfg, k) for k in ThetaMUSEConfig.__dataclass_fields__},
        })
        cfg_for_parent.task = 'e2e'
        super().__init__(cfg_for_parent)

        # Restore the GNSS task tag on the public cfg so external code
        # (adapter, trainer) can introspect it.
        self.cfg = cfg

        d_model = cfg.d_model

        # ----- override mu_v: scalar (parent) → 3-vector (NED bias)
        self.head_mu_v = ScaledHead(d_model, d_model // 2, 3)

        # ----- override R: scalar (parent) → 6-D LDL parameterization
        # applied to the receiver-reported std². Layout:
        #   [0:3] = per-axis δ (log-scale on receiver std²)
        #   [3:6] = L off-diagonal ℓ_21, ℓ_31, ℓ_32 (unit-lower triangular)
        # Zero-init (via the last-Linear zeroing at the end of __init__) →
        # δ=0 and ℓ=0 → R = diag(std² + floor²) = baseline filter.
        self.head_R = ScaledHead(d_model, d_model // 2, 6)

        # Phase 13: gated mu_w. Scalar sigmoid gate that multiplies the raw
        # mu_w output. Trained via BCE on {is_imu_bias} scenario. The gate
        # logit has a LEARNABLE NEGATIVE OFFSET (-5) so the gate starts
        # near 0 — at inference on clean data, mu_w output = sigmoid(-5)·raw
        # ≈ 0.007·raw, which is 150× smaller than a naive unit gate.
        # Inference-time drift from stray mu_w values is thus attenuated
        # by 150× relative to Phase 12's ungated head_mu_w.
        self.head_mu_w_gate = ScaledHead(d_model, d_model // 2, 1)
        # ScaledHead's final Linear has bias=False; we add a separate
        # learnable offset so the gate can be biased negative from init.
        self._mu_w_gate_offset = nn.Parameter(torch.tensor(-5.0))
        # Default log-scale = 0 → scale = 1.0 → R matches reported std²
        # Drop the parent's scalar log_R machinery (replace with no-ops to
        # avoid stale params lurking).
        if hasattr(self, 'head_log_R'):
            del self.head_log_R
        if hasattr(self, '_log_R_default'):
            self._log_R_default = None

        # R head: absolute per-axis log-variance. No dependency on receiver std.
        # scale_R already exists from parent's e2e construction — reuse it.

        # Q and mu_w heads + their scales come from the parent (e2e branch).
        # The parent's _Q_default_params are sized for UWB:
        #    Q_acc ≈ diag(4, 4, 4) m²/s⁴   (accel noise std ≈ 2 m/s²)
        # which is 1600× larger than our GNSS ESKF's actual process-noise
        # scale (sigma_a_n = 0.05 m/s² ⇒ σ² = 0.0025 m²/s⁴).  If the adapter
        # ever applies the adapter's Q output to the filter, this mismatch
        # will blow the covariance up and cause divergence — as observed
        # empirically on the v1 smoke test. Override the default buffer so
        # the GNSS model starts near the correct scale.
        _gnss_sigma_a2 = 0.05 ** 2     # 2.5e-3 m²/s⁴
        _gnss_sigma_g2 = 5e-4 ** 2     # 2.5e-7 rad²/s²
        _Q_default_gnss = torch.tensor([
            # Q_acc: L_lower(3) = 0, log_diag(3) = log(σ_a²)
            0.0, 0.0, 0.0,
            math.log(_gnss_sigma_a2), math.log(_gnss_sigma_a2), math.log(_gnss_sigma_a2),
            # Q_gyro: L_lower(3) = 0, log_diag(3) = log(σ_g²)
            0.0, 0.0, 0.0,
            math.log(_gnss_sigma_g2), math.log(_gnss_sigma_g2), math.log(_gnss_sigma_g2),
        ])
        # register_buffer overrides the existing buffer
        self._Q_default_params = _Q_default_gnss.detach()
        # Same trick on the buffer-registry side so torch.save serializes it
        if '_Q_default_params' in self._buffers:
            self._buffers['_Q_default_params'] = _Q_default_gnss.detach()

        # ----- ZERO-INIT the final Linear of every head.
        #
        # Why: kaiming-default initialized heads emit values of magnitude
        # ~0.3 (softplus(-1) scale × ~1.0 raw output). For mu_w in particular
        # this means an initial accelerometer-bias offset of ~0.3 m/s², which
        # over the 90-second grad zone integrates to:
        #     0.5 · 0.3 · 90² ≈ 1215 m of position drift
        # — exactly the kilometer-scale RMSE observed on the first profile
        # run. The model can never recover from this in early epochs.
        #
        # Solution: zero the LAST Linear of each head so the model emits
        # `default + scale·0 = default` at initialization. The defaults
        # (mu_v=0, R=diag([1,1,4]), Q=GNSS-scale, mu_w=0) are sensible
        # starting points — vanilla ESKF behavior — and the model has
        # to learn to deviate from them via the pos-loss gradient.
        with torch.no_grad():
            for head in (self.head_mu_v, self.head_R, self.head_Q, self.head_mu_w):
                # ScaledHead.net = Sequential(Linear, ReLU, Linear(bias=False))
                last = head.net[-1]
                last.weight.zero_()
                if getattr(last, 'bias', None) is not None:
                    last.bias.zero_()

            # Phase 13: mu_w gate — zero the head weights so its contribution
            # starts at 0; the learnable offset (initialized to -5 in
            # __init__) ensures σ(gate_logit) ≈ 0.007 at init regardless of
            # features.
            if hasattr(self, 'head_mu_w_gate'):
                last = self.head_mu_w_gate.net[-1]
                last.weight.zero_()
                if getattr(last, 'bias', None) is not None:
                    last.bias.zero_()

    # ------------------------------------------------------------------
    def _compute_e2e_gnss(self, h_pool, last_std=None):
        """The e2e_gnss prediction head. Returns (mu_v[B,3], R[B,3,3],
        Q[B,6,6], mu_w[B,6]).

        Args:
            h_pool   : (B, d_model) pooled backbone features
            last_std : (B, 3) RAW gnss_std from the most recent attempt (m).
                       REQUIRED for the scaled-R parametrization — the head
                       outputs a log-scale, and R = diag(std² + floor²) *
                       exp(log_scale). If None, falls back to absolute R
                       from exp(log_scale) alone (backward-compat path).
        """
        cfg = self.cfg
        B = h_pool.shape[0]
        device = h_pool.device
        dtype  = h_pool.dtype

        s_mu_v = F.softplus(self.scale_mu_v)
        s_R    = F.softplus(self.scale_R)
        s_Q    = F.softplus(self.scale_Q)
        s_mu_w = F.softplus(self.scale_mu_w)

        # mu_v (B, 3) — tanh-soft-clamped to physically-plausible GNSS bias range.
        # Limits = [h, h, d]; tanh(x/L)*L is smooth, gradient-flowing, and
        # asymptotically bounded.
        mu_v_raw = s_mu_v * self.head_mu_v(h_pool)
        mu_v_limits = torch.tensor(
            [cfg.mu_v_limit_h, cfg.mu_v_limit_h, cfg.mu_v_limit_d],
            device=device, dtype=dtype,
        )
        mu_v = mu_v_limits * torch.tanh(mu_v_raw / mu_v_limits)

        # R: LDL-residual parameterization with receiver-std skip connection.
        #
        #   D = diag(std² + floor²) · diag(exp(δ))      ← current behavior
        #   L = I + strict_lower(ℓ_21, ℓ_31, ℓ_32)       ← new cross-terms
        #   R = L · D · Lᵀ                                (guaranteed PSD)
        #
        # Zero-init of the last Linear ⇒ δ=0, ℓ=0 ⇒ R = diag(std²+floor²) =
        # identical to the non-adaptive baseline (skip connection holds).
        r_params = s_R * self.head_R(h_pool)                             # (B, 6)

        # ---- D: diagonal, residual log-scale on receiver std²
        delta_raw = r_params[:, :3]
        delta_R = torch.clamp(
            delta_raw, min=cfg.R_delta_min, max=cfg.R_delta_max,
        )                                                                # (B, 3)

        if last_std is not None:
            if last_std.dim() == 1:
                last_std = last_std.unsqueeze(0)
            base_var = (last_std.to(dtype=dtype, device=device)) ** 2 \
                       + (cfg.R_floor_m ** 2)                            # (B, 3)
        else:
            # Fallback: 1m² base when receiver std not available (e.g. unit tests)
            base_var = torch.ones(B, 3, dtype=dtype, device=device)

        D_diag = base_var * torch.exp(delta_R)                           # (B, 3)

        # ---- L: unit-lower-triangular off-diagonal. Tanh-bounded so the
        # correlation magnitude |ρ_ij| can approach 1 but never equal it
        # (R stays strictly PD). Never replaces the diagonal — only adds
        # coupling between axes.
        L_lim = cfg.R_offdiag_limit
        L_off_raw = r_params[:, 3:]                                      # (B, 3)
        L_off = L_lim * torch.tanh(L_off_raw / L_lim)                    # (B, 3)

        L_mat = torch.eye(3, dtype=dtype, device=device).expand(B, 3, 3).contiguous()
        # In-place clone-free write via index put keeps gradient flow:
        #   L[:, 1, 0] = ℓ_21;  L[:, 2, 0] = ℓ_31;  L[:, 2, 1] = ℓ_32
        # We construct with torch.zeros + eye-add to stay autograd-safe.
        L_mat = L_mat.clone()
        L_mat[:, 1, 0] = L_off[:, 0]
        L_mat[:, 2, 0] = L_off[:, 1]
        L_mat[:, 2, 1] = L_off[:, 2]

        D_mat = torch.diag_embed(D_diag)                                 # (B, 3, 3)
        R_mat = L_mat @ D_mat @ L_mat.transpose(-1, -2)                  # (B, 3, 3)
        # Symmetrize for numerical safety (round-off can make it slightly
        # non-symmetric before downstream solves).
        R_mat = 0.5 * (R_mat + R_mat.transpose(-1, -2))

        # Q: same 6×6 LDL block-diagonal as parent's e2e, but with GNSS-scale
        # log-diag clamps to prevent the untrained Q head from blowing up the
        # filter's covariance propagation over long segments.
        Q_params = self._Q_default_params.unsqueeze(0).expand(B, -1).clone()
        Q_params = Q_params + s_Q * self.head_Q(h_pool)            # (B, 12)

        # Layout per block: [L_lower(3), log_diag(3)] → indices 3:6 and 9:12
        Q_params_clamped = Q_params.clone()
        log_acc_min  = math.log(cfg.Q_acc_min)
        log_acc_max  = math.log(cfg.Q_acc_max)
        log_gyro_min = math.log(cfg.Q_gyro_min)
        log_gyro_max = math.log(cfg.Q_gyro_max)
        Q_params_clamped[:, 3:6] = torch.clamp(
            Q_params[:, 3:6], min=log_acc_min,  max=log_acc_max
        )
        Q_params_clamped[:, 9:12] = torch.clamp(
            Q_params[:, 9:12], min=log_gyro_min, max=log_gyro_max
        )

        Q_acc  = ldl_to_psd(Q_params_clamped[:, :6])               # (B, 3, 3)
        Q_gyro = ldl_to_psd(Q_params_clamped[:, 6:])               # (B, 3, 3)
        Q = torch.zeros(B, 6, 6, device=h_pool.device, dtype=h_pool.dtype)
        Q[:, :3, :3] = Q_acc
        Q[:, 3:, 3:] = Q_gyro

        # mu_w (B, 6) — tanh-soft-clamped to physically-plausible IMU bias range.
        # Layout: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]. Different
        # limits per group: ±0.05 m/s² for accel, ±0.005 rad/s for gyro.
        # These prevent the runaway integrated drift mode that would otherwise
        # corrupt training over many Adam updates (~1215 m drift / 90 s
        # segment if mu_w[acc] could reach ±0.3 m/s²).
        mu_w_raw = s_mu_w * self.head_mu_w(h_pool)
        mu_w_limits = torch.tensor(
            [cfg.mu_w_limit_acc, cfg.mu_w_limit_acc, cfg.mu_w_limit_acc,
             cfg.mu_w_limit_gyro, cfg.mu_w_limit_gyro, cfg.mu_w_limit_gyro],
            device=device, dtype=dtype,
        )
        mu_w_tanh = mu_w_limits * torch.tanh(mu_w_raw / mu_w_limits)

        # Phase 13: apply the mu_w gate. gate ∈ [0,1]; initialized near 0
        # (bias=-5 in __init__). BCE-trained on imu_bias scenario label.
        # At inference on clean-feature data: gate ≈ 0 → mu_w → 0, regardless
        # of what head_mu_w itself emits.  Expose gate as buffer for the
        # trainer's BCE loss (needs to match stored gate_logit).
        if hasattr(self, 'head_mu_w_gate'):
            gate_logit = self.head_mu_w_gate(h_pool) + self._mu_w_gate_offset  # (B, 1)
            gate = torch.sigmoid(gate_logit)                    # (B, 1) ∈ [0, 1]
            self._last_mu_w_gate_logit = gate_logit             # exposed for BCE loss
            mu_w = gate * mu_w_tanh
        else:
            mu_w = mu_w_tanh

        return mu_v, R_mat, Q, mu_w

    # ------------------------------------------------------------------
    def forward(self, x_uwb, x_imu=None, x_odom=None, logtheta_prev=None,
                last_std=None):
        """For task='e2e_gnss' returns (mu_v, R, Q, mu_w) with GNSS shapes.

        Args:
            x_uwb, x_imu, x_odom : usual token tensors
            last_std : (B, 3) raw gnss_std at the most recent attempt. Used
                       to scale the R head output. If None (e.g. old call
                       sites), R is computed absolutely from exp(log_scale).

        For any other task, falls back to the parent's forward(). This makes
        the subclass usable as a drop-in replacement for ThetaMUSEModel.
        """
        if self.cfg.task != 'e2e_gnss':
            return super().forward(x_uwb, x_imu, x_odom, logtheta_prev)

        # Re-implement only the encode + pool parts; we don't share self
        # with the parent's e2e head logic.
        B, T, _ = x_uwb.shape

        tok_uwb = self.proj_uwb(self.enc_uwb(x_uwb))
        tokens = [tok_uwb]
        mod_ids = [0]

        if self.cfg.use_imu and x_imu is not None:
            tok_imu = self.proj_imu(self.enc_imu(x_imu))
            tokens.append(tok_imu)
            mod_ids.append(len(mod_ids))
        if self.cfg.use_odom and x_odom is not None:
            tok_odom = self.proj_odom(self.enc_odom(x_odom))
            tokens.append(tok_odom)
            mod_ids.append(len(mod_ids))

        n_mod = len(tokens)
        stacked = torch.stack(tokens, dim=2)
        fused = stacked.reshape(B, T * n_mod, self.cfg.d_model)

        mod_id_seq = torch.tensor(mod_ids, device=fused.device).repeat(T)
        fused = fused + self.modality_embed(mod_id_seq).unsqueeze(0)

        h = self.backbone(fused)
        h_pool = self.pool(h)

        return self._compute_e2e_gnss(h_pool, last_std=last_std)
