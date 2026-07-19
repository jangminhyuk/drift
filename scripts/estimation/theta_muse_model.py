'''
    MUSE-style neural theta predictor for DR-ESKF-TAC.

    Multi-modal encoder (UWB MLP, IMU 1D-CNN, Odom linear) fused via
    Mamba (or Transformer fallback) backbone, with scaled decoder heads
    that predict noise parameters (mu_v, R, Q, mu_w).

    Key design: decoder heads use bias=False on the last layer, with
    defaults stored as buffers. A shared learned output_scale (sigmoid)
    controls how much input-dependent predictions deviate from defaults.
    Starts small (~0.05) to prevent mode collapse, grows during training.
'''
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ThetaMUSEConfig:
    d_model: int = 256
    d_uwb: int = 16       # input UWB feature dim per timestep
    d_odom: int = 19      # input odom feature dim per timestep
    L_imu: int = 200      # IMU samples per inter-measurement interval
    T: int = 100          # window length (number of UWB attempts)
    n_blocks: int = 2     # number of backbone blocks (Mamba or Transformer)
    n_heads: int = 4      # Transformer heads (ignored when use_mamba=True)
    use_mamba: bool = True
    use_imu: bool = True
    use_odom: bool = True
    n_classes_tw: int = 6  # number of theta_w grid values (backward compat)
    n_classes_tv: int = 7  # number of theta_v grid values (backward compat)
    task: str = 'meas_noise'  # 'meas_noise' | 'e2e' | 'theta' (backward compat)
    R_min: float = 0.001
    R_max: float = 10.0
    dropout: float = 0.1
    # Q block dimensions for LDL parametrization (e2e task)
    d_Q_acc: int = 3      # acceleration noise block size
    d_Q_gyro: int = 3     # gyroscope noise block size
    # Backbone selection: 'mamba' | 'transformer' | 'gru'
    backbone_type: str = 'mamba'
    # RoNIN pretrained IMU encoder
    ronin_checkpoint: Optional[str] = None  # path to RoNIN ResNet18 checkpoint
    d_ronin: int = 512     # RoNIN encoder output dim (after global avg pool)
    # GRU variant config (only relevant when backbone_type='gru'). The default
    # corresponds to the v3 architectural fix: unidirectional, hidden=d_model.
    # Set gru_bidirectional=True + gru_hidden_size=d_model//2 to recover the
    # original v1 architecture, or set gru_output_proj=True to add a projection
    # back to d_model after a wider bidirectional GRU.
    gru_bidirectional: bool = False
    gru_hidden_size: Optional[int] = None  # None -> d_model (v3 default)
    gru_output_proj: bool = False


# ------------------------------------------------------------------ #
#  Encoders
# ------------------------------------------------------------------ #
class UWBEncoder(nn.Module):
    '''MLP encoder for per-timestep UWB features.'''
    def __init__(self, d_in, d_hidden=128, d_out=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, T, d_in) -> (B, T, d_out)
        return self.net(x)


class IMUEncoder(nn.Module):
    '''Compact 1D-CNN encoder for IMU chunks.'''
    def __init__(self, d_out=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, d_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.d_out = d_out

    def forward(self, x):
        # x: (B, T, L, 6) -> (B, T, d_out)
        B, T, L, C = x.shape
        x = x.reshape(B * T, L, C).permute(0, 2, 1)   # (B*T, 6, L)
        out = self.conv(x).squeeze(-1)                  # (B*T, d_out)
        return out.reshape(B, T, self.d_out)


class RoNINIMUEncoder(nn.Module):
    '''Frozen pretrained RoNIN ResNet18 encoder for IMU chunks.

    Loads RoNIN ResNet18 (input_block + 4 residual groups), discards the
    FC output head, and applies global average pooling to get a 512-dim
    feature vector per IMU chunk. All RoNIN weights are frozen.
    '''
    def __init__(self, checkpoint_path, d_out=512):
        super().__init__()
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / 'ronin'))
        from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule

        _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5,
                      'trans_planes': 128}
        full_model = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
                              base_plane=64, output_block=FCOutputModule,
                              kernel_size=3, **_fc_config)
        ckpt = torch.load(checkpoint_path, map_location='cpu',
                          weights_only=False)
        full_model.load_state_dict(ckpt['model_state_dict'])

        # Keep only the encoder (input_block + residual_groups)
        self.input_block = full_model.input_block
        self.residual_groups = full_model.residual_groups
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.d_out = d_out  # 512 for ResNet18

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, T, L, 6) -> (B, T, 512)
        B, T, L, C = x.shape
        x = x.reshape(B * T, L, C).permute(0, 2, 1)   # (B*T, 6, L)
        with torch.no_grad():
            h = self.input_block(x)
            h = self.residual_groups(h)
        h = self.pool(h).squeeze(-1)                     # (B*T, 512)
        return h.reshape(B, T, self.d_out)


class OdomEncoder(nn.Module):
    '''Linear encoder for per-timestep odometry / state features.'''
    def __init__(self, d_in, d_out=128):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        # x: (B, T, d_in) -> (B, T, d_out)
        return self.linear(x)


# ------------------------------------------------------------------ #
#  Zero-init MLP (MUSE-style residual decoder)
# ------------------------------------------------------------------ #
class ZeroInitMLP(nn.Module):
    '''MLP whose last linear layer is initialized to zero so the
    initial prediction is exactly logtheta_prev (pure residual).'''
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_hidden, d_out)
        # Zero-init last layer
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ScaledHead(nn.Module):
    '''MLP head with bias=False on last layer. Output is scaled by a
    shared learned scalar and offset by a fixed default.'''
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ #
#  Statistical Pooling (replaces single-query attention pooling)
# ------------------------------------------------------------------ #
class StatisticalPooling(nn.Module):
    """Pool a token sequence into a fixed-size vector that captures both
    content (multi-query attention means) and distributional statistics
    (variance, max) of the backbone embeddings.

    This gives decoder heads access to within-window distributional info
    (e.g. "how variable is this sequence?"), which single-query attention
    pooling destroys. The variance channel naturally differentiates
    easy (uniform tokens) from hard (diverse tokens) environments.
    """
    def __init__(self, d_model, n_queries=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, n_queries, bias=False)
        self.n_queries = n_queries
        self.d_model = d_model
        # n_queries attention means + 1 variance + 1 max
        self.out_proj = nn.Linear((n_queries + 2) * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        """
        Args:
            h: (B, T, d_model) backbone output tokens
        Returns:
            pooled: (B, d_model)
        """
        B, T, D = h.shape

        # Multi-query attention pooling
        attn_logits = self.query_proj(h)                    # (B, T, n_queries)
        attn_weights = torch.softmax(attn_logits, dim=1)    # (B, T, n_queries)
        attn_means = torch.bmm(attn_weights.transpose(1, 2), h)  # (B, n_q, D)
        attn_flat = attn_means.reshape(B, self.n_queries * D)

        # Token-level variance (captures "how variable is this sequence?")
        h_mean = h.mean(dim=1, keepdim=True)                # (B, 1, D)
        h_var = ((h - h_mean) ** 2).mean(dim=1)             # (B, D)

        # Token-level max (captures extremes / outliers)
        h_max = h.max(dim=1).values                         # (B, D)

        # Concatenate and project back to d_model
        pooled = torch.cat([attn_flat, h_var, h_max], dim=1)  # (B, (n_q+2)*D)
        pooled = self.out_proj(pooled)                         # (B, D)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return pooled


# ------------------------------------------------------------------ #
#  Backbone helpers
# ------------------------------------------------------------------ #
def _build_mamba_backbone(d_model, n_blocks):
    '''Build Mamba backbone.  Imports mamba_ssm; caller should catch
    ImportError to fall back to Transformer.'''
    from mamba_ssm import Mamba
    layers = []
    for _ in range(n_blocks):
        layers.append(nn.LayerNorm(d_model))
        layers.append(Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2))
    return layers


class MambaResidualBlock(nn.Module):
    def __init__(self, norm, mamba):
        super().__init__()
        self.norm = norm
        self.mamba = mamba

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class TransformerBackbone(nn.Module):
    '''Standard TransformerEncoder fallback.'''
    def __init__(self, d_model, n_heads, n_blocks, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)

    def forward(self, x):
        return self.encoder(x)


class GRUBackbone(nn.Module):
    '''GRU backbone returning full sequence; supports several variants.

    Variants (controlled by constructor args; `theta_muse_trainer.py`
    forwards the cfg.gru_* fields):

    - **v3 (default)**: unidirectional, hidden_size=d_model, n_layers=2.
      Restores parameter count to the same order of magnitude as Mamba /
      Transformer (~198k vs ~75k for v1), addressing the under-
      parameterisation that made GRU+ESKF underperform vanilla ESKF on
      the UTIL UWB task.
    - **v1 (legacy)**: bidirectional, hidden_size=d_model//2, n_layers=2.
      Output dim already matches d_model via bidirectional concatenation;
      no output projection.
    - **bidir + projection**: bidirectional, hidden_size=d_model + a
      Linear(2*d_model -> d_model) projection. Larger capacity than v1
      while preserving the bidirectional context aggregation.

    Args:
        d_model: backbone width (matches the rest of the model).
        n_layers: stacked GRU layers.
        dropout: applied between layers (PyTorch GRU dropout).
        bidirectional: if True, run a bidirectional GRU.
        hidden_size: per-direction recurrent state size; default is
            d_model (unidirectional) or d_model//2 (bidirectional, no
            projection — keeps output dim equal to d_model). When
            output_proj=True, hidden_size defaults to d_model.
        output_proj: if True, apply a Linear projection from the
            (possibly 2*hidden_size) GRU output back to d_model.
    '''
    def __init__(self, d_model, n_layers=2, dropout=0.1,
                 bidirectional=False, hidden_size=None,
                 output_proj=False):
        super().__init__()
        if hidden_size is None:
            if output_proj:
                hidden_size = d_model
            elif bidirectional:
                hidden_size = d_model // 2  # so concat output == d_model
            else:
                hidden_size = d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self._raw_out_dim = hidden_size * (2 if bidirectional else 1)
        if output_proj or self._raw_out_dim != d_model:
            self.proj = nn.Linear(self._raw_out_dim, d_model)
        else:
            self.proj = nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, raw_out_dim)
        out = self.proj(out)          # (B, T, d_model)
        return self.norm(out + x)     # residual + LayerNorm


# ------------------------------------------------------------------ #
#  LDL reconstruction utility (for e2e Q parametrization)
# ------------------------------------------------------------------ #
def ldl_to_psd(params, block_size=3):
    """Convert LDL parameters to PSD matrix.

    Args:
        params: (B, 6) — [L_lower(3), log_D(3)] for one 3×3 block
    Returns:
        Q_block: (B, 3, 3) positive definite matrix = L @ D @ L^T
    """
    B = params.shape[0]
    device = params.device
    L = torch.eye(block_size, device=device, dtype=params.dtype) \
        .unsqueeze(0).expand(B, -1, -1).clone()
    # Fill lower triangle: indices (1,0), (2,0), (2,1)
    L[:, 1, 0] = params[:, 0]
    L[:, 2, 0] = params[:, 1]
    L[:, 2, 1] = params[:, 2]
    D = torch.diag_embed(torch.exp(params[:, 3:6]))  # (B, 3, 3)
    return L @ D @ L.transpose(-1, -2)  # (B, 3, 3) PD


# ------------------------------------------------------------------ #
#  Main model
# ------------------------------------------------------------------ #
class ThetaMUSEModel(nn.Module):
    def __init__(self, cfg: ThetaMUSEConfig):
        super().__init__()
        self.cfg = cfg

        # --- Encoders ---
        self.enc_uwb = UWBEncoder(cfg.d_uwb, 128, 256, cfg.dropout)
        self.proj_uwb = nn.Linear(256, cfg.d_model)

        if cfg.use_imu:
            if cfg.ronin_checkpoint is not None:
                self.enc_imu = RoNINIMUEncoder(cfg.ronin_checkpoint,
                                               d_out=cfg.d_ronin)
                self.proj_imu = nn.Linear(cfg.d_ronin, cfg.d_model)
            else:
                self.enc_imu = IMUEncoder(128, cfg.dropout)
                self.proj_imu = nn.Linear(128, cfg.d_model)

        if cfg.use_odom:
            self.enc_odom = OdomEncoder(cfg.d_odom, 128)
            self.proj_odom = nn.Linear(128, cfg.d_model)

        # --- Modality type embeddings ---
        # Learnable embeddings so the backbone can distinguish UWB/IMU/Odom
        # tokens in the interleaved sequence (analogous to token_type_embeddings
        # in BERT / segment embeddings in Transformers)
        n_modalities = 1 + int(cfg.use_imu) + int(cfg.use_odom)
        self.modality_embed = nn.Embedding(n_modalities, cfg.d_model)

        # --- Backbone ---
        # Resolve backbone_type (backward compat: use_mamba maps to 'mamba')
        bt = getattr(cfg, 'backbone_type', 'mamba' if cfg.use_mamba else 'transformer')
        self._use_mamba = False

        if bt == 'mamba':
            try:
                mamba_parts = _build_mamba_backbone(cfg.d_model, cfg.n_blocks)
                blocks = []
                for i in range(0, len(mamba_parts), 2):
                    blocks.append(MambaResidualBlock(mamba_parts[i], mamba_parts[i + 1]))
                self.backbone = nn.Sequential(*blocks)
                self._use_mamba = True
            except ImportError:
                print('[ThetaMUSEModel] mamba_ssm not found, falling back to transformer')
                bt = 'transformer'

        if bt == 'transformer':
            self.backbone = TransformerBackbone(
                cfg.d_model, cfg.n_heads, cfg.n_blocks, cfg.dropout)
        elif bt == 'gru':
            self.backbone = GRUBackbone(
                cfg.d_model, n_layers=cfg.n_blocks, dropout=cfg.dropout,
                bidirectional=getattr(cfg, 'gru_bidirectional', False),
                hidden_size=getattr(cfg, 'gru_hidden_size', None),
                output_proj=getattr(cfg, 'gru_output_proj', False),
            )

        # --- Statistical pooling ---
        # Multi-query attention + variance + max pooling so decoder heads
        # can see within-window distributional information (not just a
        # single attention-weighted mean).
        self.pool = StatisticalPooling(cfg.d_model, n_queries=4,
                                       dropout=cfg.dropout)

        # --- Task-specific heads ---
        # Scaled heads: last layer has bias=False. A shared learned scalar
        # (output_scale) controls how much the input-dependent predictions
        # deviate from fixed defaults. Starts small (~0.05) so the filter
        # runs stably with near-default params, then grows as the model
        # learns useful input-dependent features. This prevents the mode
        # collapse seen in v18 where all heads converged to constant output.
        if cfg.task in ('meas_noise', 'e2e'):
            self.head_mu_v = ScaledHead(cfg.d_model, cfg.d_model // 2, 1)
            self.head_log_R = ScaledHead(cfg.d_model, cfg.d_model // 2, 1)
            self.register_buffer('_log_R_default',
                                 torch.tensor([math.log(0.05)]))

        if cfg.task == 'e2e':
            # Process noise Q: LDL parametrization (12 params)
            self.head_Q = ScaledHead(cfg.d_model, cfg.d_model // 2, 12)
            _Q_default = [
                0, 0, 0, math.log(4.0), math.log(4.0), math.log(4.0),
                0, 0, 0, math.log(0.01), math.log(0.01), math.log(0.01),
            ]
            self.register_buffer('_Q_default_params',
                                 torch.tensor(_Q_default))

            # Process bias mu_w: 6D (default = 0)
            self.head_mu_w = ScaledHead(cfg.d_model, cfg.d_model // 2, 6)

            # Per-head output scales using softplus(α) = log(1 + exp(α)).
            # Init α=-1 → s≈0.31, giving 36x faster effective LR than
            # sigmoid at α=-3. Softplus has no upper saturation (unlike
            # sigmoid), so the scale can grow past 1.0 if needed.
            self.scale_mu_v = nn.Parameter(torch.tensor(-1.0))
            self.scale_R = nn.Parameter(torch.tensor(-1.0))
            self.scale_Q = nn.Parameter(torch.tensor(-1.0))
            self.scale_mu_w = nn.Parameter(torch.tensor(-1.0))

        if cfg.task == 'theta':
            # Classification heads (backward compat for theta task)
            self.head_tw = nn.Sequential(
                nn.Linear(cfg.d_model, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(128, cfg.n_classes_tw),
            )
            self.head_tv = nn.Sequential(
                nn.Linear(cfg.d_model, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(128, cfg.n_classes_tv),
            )

    def forward(self, x_uwb, x_imu=None, x_odom=None, logtheta_prev=None):
        '''
        Args:
            x_uwb:  (B, T, d_uwb)
            x_imu:  (B, T, L, 6) or None
            x_odom: (B, T, d_odom) or None
            logtheta_prev: ignored (kept for backward compat)

        Returns (task-dependent):
            e2e:        (mu_v, R, Q, mu_w)  -- (B,), (B,), (B,6,6), (B,6)
            meas_noise: (mu_v, log_R)       -- (B,), (B,)
            theta:      (logits_tw, logits_tv) -- (B, n_tw), (B, n_tv)
        '''
        B, T, _ = x_uwb.shape

        # Encode each modality
        tok_uwb = self.proj_uwb(self.enc_uwb(x_uwb))          # (B, T, d_model)
        tokens = [tok_uwb]
        mod_ids = [0]  # UWB = modality 0

        if self.cfg.use_imu and x_imu is not None:
            tok_imu = self.proj_imu(self.enc_imu(x_imu))       # (B, T, d_model)
            tokens.append(tok_imu)
            mod_ids.append(len(mod_ids))  # IMU = modality 1

        if self.cfg.use_odom and x_odom is not None:
            tok_odom = self.proj_odom(self.enc_odom(x_odom))    # (B, T, d_model)
            tokens.append(tok_odom)
            mod_ids.append(len(mod_ids))  # Odom = modality 2

        # Interleave: for each timestep, stack modality tokens
        # Result shape: (B, n_mod*T, d_model)
        n_mod = len(tokens)
        stacked = torch.stack(tokens, dim=2)   # (B, T, n_mod, d_model)
        fused = stacked.reshape(B, T * n_mod, self.cfg.d_model)

        # Add modality type embeddings
        # Build ids: [0,1,2, 0,1,2, ...] repeated T times
        mod_id_seq = torch.tensor(mod_ids, device=fused.device).repeat(T)  # (T*n_mod,)
        fused = fused + self.modality_embed(mod_id_seq).unsqueeze(0)  # broadcast over B

        # Backbone
        h = self.backbone(fused)                # (B, T*n_mod, d_model)

        # Statistical pooling: multi-query attention + variance + max
        h_pool = self.pool(h)                               # (B, d_model)

        if self.cfg.task == 'e2e':
            # Per-head scaled predictions: default + scale_i * head_i(h_pool).
            # Each head has its own scale. softplus(α) = log(1+exp(α)):
            # no upper saturation, always positive, healthy gradients.
            s_mu_v = F.softplus(self.scale_mu_v)
            s_R = F.softplus(self.scale_R)
            s_Q = F.softplus(self.scale_Q)
            s_mu_w = F.softplus(self.scale_mu_w)

            mu_v = (s_mu_v * self.head_mu_v(h_pool)).squeeze(-1)          # (B,)
            log_R = (self._log_R_default
                     + s_R * self.head_log_R(h_pool)).squeeze(-1)         # (B,)
            Q_params = self._Q_default_params
            Q_params = Q_params + s_Q * self.head_Q(h_pool)              # (B, 12)
            mu_w = s_mu_w * self.head_mu_w(h_pool)                        # (B, 6)

            R = torch.clamp(torch.exp(log_R), self.cfg.R_min, self.cfg.R_max)

            # Q: LDL parametrization → PSD (block-diagonal acc + gyro)
            Q_acc = ldl_to_psd(Q_params[:, :6])                # (B, 3, 3)
            Q_gyro = ldl_to_psd(Q_params[:, 6:])               # (B, 3, 3)
            Q = torch.zeros(B, 6, 6, device=h_pool.device, dtype=h_pool.dtype)
            Q[:, :3, :3] = Q_acc
            Q[:, 3:, 3:] = Q_gyro

            return mu_v, R, Q, mu_w

        elif self.cfg.task == 'meas_noise':
            mu_v = self.head_mu_v(h_pool).squeeze(-1)      # (B,)
            log_R = (self._log_R_default
                     + self.head_log_R(h_pool)).squeeze(-1) # (B,)
            return mu_v, log_R

        else:
            # Backward compat: theta classification
            logits_tw = self.head_tw(h_pool)    # (B, n_classes_tw)
            logits_tv = self.head_tv(h_pool)    # (B, n_classes_tv)
            return logits_tw, logits_tv
