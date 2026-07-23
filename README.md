# WRAP: Wasserstein-Robust Adaptive Plug-in for Robot Localization

Code and precomputed results for the WRAP paper. WRAP combines a learned
adaptive nominal-noise module (Mamba / GRU / Transformer backbone) with a
Wasserstein distributionally-robust covariance layer, evaluated on UWB–IMU
(UTIL dataset) and GNSS–INS (i2Nav / KF-GINS) localization.

## Layout

```
reproduce/           one script per paper figure / table (fast path, no GPU)
data/precomputed/    committed inputs the reproduce scripts read
scripts/estimation/  UWB pipeline (ESKF, DR-ESKF, adaptive module, eval)
scripts/gnss_imu/    GNSS/INS pipeline (KF-GINS adapter + DR, eval)
scripts/drekf_cpp/   C++ Wasserstein DR-MMSE solver (pybind11)
checkpoints/         trained adapter weights
```

## Reproduce the paper figures and table

These read the committed data in `data/precomputed/` — no GPU, dataset, or
C++ build required.

```bash
pip install -r requirements.txt

python reproduce/fig2_height_tracking.py   # Fig. 2  -> figures/height_tracking_C4_traj3_upper.pdf
python reproduce/fig3_improvement.py       # Fig. 3  -> figures/improvement_tdoa2.pdf
python reproduce/fig4_regimes.py           # Fig. 4  -> figures/FIG_REGIMES_UWB.pdf
python reproduce/table1_uwb_rmse.py        # Table I -> tables/uwb_rmse_table.tex
python reproduce/fig_gnss_per_seq.py       # GNSS    -> figures/F_per_seq_improvement.pdf
```

## Evaluating from the raw datasets

Optional — only needed to regenerate the precomputed inputs from scratch.
Requires a GPU environment; extra dependencies beyond `requirements.txt`:
`torch`, `mamba-ssm`, `pybind11`, `cmake`, `eigen3`.

1. Build the two C++ pybind modules:
   ```bash
   cmake -S scripts/drekf_cpp -B scripts/drekf_cpp/build && cmake --build scripts/drekf_cpp/build -j
   cmake -S scripts/gnss_imu/cpp_kfgins -B scripts/gnss_imu/cpp_kfgins/build && cmake --build scripts/gnss_imu/cpp_kfgins/build -j
   ```
   (A prebuilt `kfgins_py*.so` for Linux/py3.10 is included; rebuild for other platforms.)
2. Datasets: UTIL UWB (https://utiasdsl.github.io/util-uwb-dataset/) and i2Nav
   (https://github.com/i2Nav-WHU). Point the pipeline at them via the
   `WRAP_DATA_ROOT` / `WRAP_SHARD_ROOT` environment variables.
3. RoNIN encoder (third-party, required to load the UWB adapters): download the
   RoNIN ResNet checkpoint from the official RoNIN project and place it at
   `checkpoints/ronin/checkpoint_gsn_latest.pt`.
4. Run the evaluation entry points: `scripts/estimation/eval_8way.py` (UWB) and
   `scripts/gnss_imu/runners/main_kfgins.py` (GNSS), then the `reproduce/`
   scripts to render the figures.

## Checkpoints

The adapter weights (`checkpoints/muse_e2e_*_v47_tail003.pt`,
`checkpoints/muse_kfgins_sup_phase12_v6_c.pt`) are included directly (~20 MB
each). The RoNIN encoder is third-party and must be downloaded separately.
