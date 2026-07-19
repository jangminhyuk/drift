import os
#!/usr/bin/env python3
"""Guideline table over ALL 41 tdoa2 flights (user request 2026-07-07) +
7 i2Nav sequences. Replaces the trial6-only view: eval_table_v1 covered every
tdoa2 sequence (const1/2: 6 trials, const3: 6 trials + 2 manual, const4:
6 trials x 3 trajs + 3 manual) with the 4 standard conditions at per-sequence
best-theta TAC DR and the v43 Mamba adapter.

Per flight, from the per-sequence shard eskf.npz + raw CSV:
  ratio  = bias_rms / resid_rms of baseline state error (win_decomp, W=5 s)
  H_fast = 1 - rms(z-h(gt)) / rms(h(x_hat)-h(gt))       (dr_predictor.py logic)
  x_q99  = q99(|windowed de-biased z-h(gt)|/sigma)/2.576 (moment_mismatch.py)
Gains vs ESKF baseline from results/eval_table_v1/uwb_rmse_per_sequence.csv.
i2Nav rows merged unchanged from the existing per-sequence artifacts.

Output: artifacts/guideline_units_alltdoa2.csv
"""
import glob
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path('.')
ART = ROOT / "research/dr_conclusions/artifacts"
SHARDS = os.path.join(os.environ.get('DRIFT_SHARD_ROOT','shards'), "eval_table_v1")
CSV_DIR = ROOT / "dataset/flight-dataset/csv-data"

T_UV = np.array([-0.01245, 0.00127, 0.0908])
SIG_UWB = np.sqrt(0.05)


def quat_rot(q, v):
    w, x, y, z = q.T
    R = np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
    return np.einsum("nij,j->ni", R, v)


def win_stats(t, err, sig, W):
    bias_n, resid_n = [], []
    t0 = t[0]
    for i in range(max(int((t[-1]-t0)/W), 1)):
        m = (t >= t0+i*W) & (t < t0+(i+1)*W)
        if m.sum() < 5:
            continue
        b = err[m].mean()
        bias_n.append(b/sig[m].mean())
        resid_n.extend((err[m]-b)/sig[m])
    return np.array(bias_n), np.array(resid_n)


def win_decomp(t, e, W=5.0):
    bias2, resid = [], []
    t0 = t[0]
    for i in range(max(int((t[-1]-t0)/W), 1)):
        m = (t >= t0+i*W) & (t < t0+(i+1)*W)
        if m.sum() < 5:
            continue
        w = e[m]
        b = w.mean(0)
        bias2.append(b @ b)
        resid.append(np.trace(np.cov(w.T)))
    return float(np.sqrt(np.mean(bias2))), float(np.sqrt(np.mean(resid)))


tab = pd.read_csv(ROOT / "results/eval_table_v1/uwb_rmse_per_sequence.csv")
tab = tab[tab.tdoa == 2].set_index("sequence")

rows = []
for name in tab.index:
    const = tab.loc[name, "constellation"]
    npzs = glob.glob(f"{SHARDS}/{name}__b*/{const}/{name}/eskf.npz")
    assert len(npzs) == 1, (name, npzs)
    z = np.load(npzs[0], allow_pickle=True)
    an = z["anchor_position"]

    # y: bias/noise of baseline state error
    b, r = win_decomp(z["t"], z["pos_error"], W=5.0)
    ratio = b / r

    df = pd.read_csv(CSV_DIR / const / f"{name}.csv", usecols=[
        "t_tdoa", "idA", "idB", "tdoa_meas", "t_acc", "t_pose", "pose_x",
        "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"])
    td = df[["t_tdoa", "idA", "idB", "tdoa_meas"]].dropna().to_numpy()
    gp = df[["t_pose", "pose_x", "pose_y", "pose_z",
             "pose_qw", "pose_qx", "pose_qy", "pose_qz"]].dropna().to_numpy()

    # H_fast on the filter timebase
    min_t = min(td[0, 0], float(df["t_acc"].dropna().iloc[0]), gp[0, 0])
    tt, tp = td[:, 0] - min_t, gp[:, 0] - min_t
    t_f, t_v = z["t"], z["t_vicon"]
    lo = max(t_f[0], tp[0], t_v[0]) + 1.0
    hi = min(t_f[-1], tp[-1], t_v[-1]) - 1.0
    m = (tt >= lo) & (tt <= hi)
    tdc, ttc = td[m], tt[m]
    pos_g = np.stack([np.interp(ttc, tp, gp[:, 1+k]) for k in range(3)], 1)
    q_g = np.stack([np.interp(ttc, tp, gp[:, 4+k]) for k in range(4)], 1)
    q_g /= np.linalg.norm(q_g, axis=1, keepdims=True)
    tag_g = pos_g + quat_rot(q_g, T_UV)
    pos_f = np.stack([np.interp(ttc, t_f, z["Xpo"][:, k]) for k in range(3)], 1)
    q_f = np.stack([np.interp(ttc, t_f, z["q_list"][:, k]) for k in range(4)], 1)
    q_f /= np.linalg.norm(q_f, axis=1, keepdims=True)
    tag_f = pos_f + quat_rot(q_f, T_UV)
    iA, iB = tdc[:, 1].astype(int), tdc[:, 2].astype(int)
    e_meas = tdc[:, 3] - (np.linalg.norm(an[iB] - tag_g, axis=1)
                          - np.linalg.norm(an[iA] - tag_g, axis=1))
    nu = tdc[:, 3] - (np.linalg.norm(an[iB] - tag_f, axis=1)
                      - np.linalg.norm(an[iA] - tag_f, axis=1))
    e_track = e_meas - nu
    H_fast = float(1 - np.sqrt((e_meas**2).mean())/np.sqrt((e_track**2).mean()))

    # tail ratio on the raw GT timebase
    tt2 = td[:, 0]
    inside = (tt2 >= gp[0, 0]) & (tt2 <= gp[-1, 0])
    td2, tt2 = td[inside], tt2[inside]
    pos = np.stack([np.interp(tt2, gp[:, 0], gp[:, 1+k]) for k in range(3)], 1)
    q = np.stack([np.interp(tt2, gp[:, 0], gp[:, 4+k]) for k in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ptag = pos + quat_rot(q, T_UV)
    dA = np.linalg.norm(an[td2[:, 1].astype(int)] - ptag, axis=1)
    dB = np.linalg.norm(an[td2[:, 2].astype(int)] - ptag, axis=1)
    err = td2[:, 3] - (dB - dA)
    sig = np.full_like(err, SIG_UWB)
    Rs = []
    for pair in np.unique(td2[:, 1:3], axis=0):
        pm = (td2[:, 1] == pair[0]) & (td2[:, 2] == pair[1])
        _, rr = win_stats(tt2[pm], err[pm], sig[pm], W=5.0)
        Rs.append(rr)
    x_q99 = float(np.quantile(np.abs(np.concatenate(Rs)), 0.99))/2.576

    base = tab.loc[name, "ESKF"]
    rows.append({
        "ds": "UWB", "unit": name, "const": const,
        "ratio": ratio, "H_fast": H_fast, "x_q99": x_q99,
        "rmse_base": base,
        "pct_adapter": 100*(1 - tab.loc[name, "Adapter-only"]/base),
        "pct_dr_alone": 100*(1 - tab.loc[name, "DR-only"]/base),
        "pct_both": 100*(1 - tab.loc[name, "DRiFt"]/base)})
    print(f"{name}: ratio={ratio:.2f} H_fast={H_fast:+.2f} T={x_q99:.2f} "
          f"adp={rows[-1]['pct_adapter']:+.1f} dr={rows[-1]['pct_dr_alone']:+.1f} "
          f"both={rows[-1]['pct_both']:+.1f}")

uwb = pd.DataFrame(rows)

pred = pd.read_csv(ART / "dr_predictor_units.csv")
mm = pd.read_csv(ART / "moment_mismatch_units.csv").rename(columns={"lab": "unit"})
bm = pd.read_csv(ART / "bias_map_units.csv").rename(columns={"lab": "unit"})
i2 = (pred[pred.ds == "i2Nav"][["ds", "unit", "H_fast"]]
      .merge(mm[["ds", "unit", "x_q99"]], on=["ds", "unit"])
      .merge(bm[["ds", "unit", "ratio", "b0", "pct_adapter", "pct_dr_alone",
                 "pct_both"]], on=["ds", "unit"])
      .rename(columns={"b0": "rmse_base"}))
i2["const"] = "i2Nav"

out = pd.concat([uwb, i2], ignore_index=True)
out["x"] = np.maximum(out.H_fast, out.x_q99 - 1)
out.to_csv(ART / "guideline_units_alltdoa2.csv", index=False)
print(f"\nwrote {ART/'guideline_units_alltdoa2.csv'} ({len(out)} units)")
