#!/usr/bin/env python3
"""FIG v5 — UWB-only decision guideline, camera-ready (user 2026-07-09):
no suptitle, no panel titles/stat, no footnote, no worked-example legend,
no in-figure prose — the LaTeX caption carries all explanation. Much larger
fonts. Same held-out data and colour rule as v4 (score = gain − 5 pts per
module; fill fades toward white as the winner's margin shrinks; trial-7
manual flights = out-of-scope diamonds; dagger = the one decisive mismatch,
explained in the caption).
"""
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "research/dr_conclusions/artifacts"
OUT = ROOT / "research/dr_conclusions/figures_guideline_v5_paper"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 21, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})

C_ADP, C_DR, C_BOTH, C_NONE = "#e34948", "#2a78d6", "#4a3aa7", "#898781"
INK, INK2 = "#0b0b0b", "#52514e"
CAT_COLOR = {"adaptive": C_ADP, "dr": C_DR, "drift": C_BOTH, "neither": C_NONE}
XB, PAY = 0.0, 5.0   # UWB-only: no flight in (0, 0.05), boundary = 0
# Fill = pure argmax of the net score (gain - 5 pts per module). The former
# prescription tie-break was REMOVED (user 2026-07-09, reviewer robustness):
# 6 near-tie flights (<5 net pts, flips split both ways) now color off-region
# alongside the one decisive exception (const4-t3-traj3).
# Manual flights are INCLUDED as normal points (user 2026-07-09; the earlier
# open-marker out-of-scope treatment is retired). Sequences where no variant
# pays (argmax = neither, the 3 const4 manuals) draw HOLLOW with an ink edge:
# "deploy nothing, keep the nominal ESKF" — avoids the gray fill that
# collided with the Well-calibrated region tint.
MANUALS_FILLED = True


def blend(c, w):
    r, g, b = mcolors.to_rgb(c)
    return (1 - w + w*r, 1 - w + w*g, 1 - w + w*b)


# Gains under the VALIDATION-THETA protocol (user 2026-07-09): DR radii
# fixed per constellation on trials 4-5, matching the paper table/Fig 3.
# The per-sequence best-theta (oracle) view = guideline_units_alltdoa2.csv.
# GUIDELINE_UNITS_CSV / FIG_SUFFIX env overrides let the v47 pipeline render
# a sibling figure (e.g. FIG_REGIMES_UWB_v47) without touching the paper one.
# default = v47 family (adopted 2026-07-12); v43-family view =
# guideline_units_alltdoa2_valtheta.csv
UNITS_CSV = os.environ.get("GUIDELINE_UNITS_CSV",
                           "guideline_units_alltdoa2_v47.csv")
FIG_SUFFIX = os.environ.get("FIG_SUFFIX", "")
# default 'group' = the ADOPTED canonical style (user 2026-07-12): fills are
# environment-level median verdicts; 'winner' = the retired per-trial fills.
STYLE = os.environ.get("FIG_STYLE", "group")
d = pd.read_csv(ART / UNITS_CSV)
d = d[d.ds == "UWB"].copy()
sc = np.stack([np.zeros(len(d)), d.pct_adapter - PAY,
               d.pct_dr_alone - PAY, d.pct_both - 2*PAY], 1)
d["obs"] = np.array(["neither", "adaptive", "dr", "drift"])[sc.argmax(1)]
srt = np.sort(sc, 1)
d["margin"] = srt[:, -1] - srt[:, -2]
d["region"] = np.where(d.ratio > 1,
                       np.where(d.x < XB, "adaptive", "drift"),
                       np.where(d.x >= XB, "dr", "neither"))
d["manual"] = d.unit.str.contains("manual").fillna(False)
# Group-level verdict (const x manual flag): trials within a group are
# repeated measurements of one deployment environment, and the variant
# choice is made per environment before flying — so the 'group'/'glyph'
# styles classify at that level (2026-07-12; the per-trial fills in const1/2
# are coin flips around the 5-pt threshold on statistically identical
# flights, x-axis study in EXPERIMENT_LOG).
gmed = d.groupby(["const", "manual"])[
    ["pct_adapter", "pct_dr_alone", "pct_both"]].transform("median")
gsc = np.stack([np.zeros(len(d)), gmed.pct_adapter - PAY,
                gmed.pct_dr_alone - PAY, gmed.pct_both - 2*PAY], 1)
d["grp_obs"] = np.array(["neither", "adaptive", "dr", "drift"])[gsc.argmax(1)]

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(24, 8.2), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.25, 1.0]})

# ============================ (a) decision map ============================
XL, XR, YB, YT = -1.02, 1.9, 0.3, 4.2
xmid = (XB - XL) / (XR - XL)
axa.axhspan(1, YT, xmin=0, xmax=xmid, color=C_ADP, alpha=0.06)
axa.axhspan(1, YT, xmin=xmid, xmax=1, color=C_BOTH, alpha=0.07)
axa.axhspan(YB, 1, xmin=xmid, xmax=1, color=C_DR, alpha=0.07)
axa.axhspan(YB, 1, xmin=0, xmax=xmid, color=C_NONE, alpha=0.07)
# DIAGNOSTIC region names (user 2026-07-13): the axes measure the two ways
# the nominal filter is mis-specified -- mean/bias (y) and covariance (x) --
# NOT which variant wins. Naming them by the defect (not the prescribed
# deployment) makes the map backbone-invariant and decouples the name from
# the module-specific FILL: a purple dot in the "Bias-dominated" region is
# then an exception to report, not a contradiction of the region's name.
# The colours now encode the module that TARGETS each defect (red adapter
# fixes bias, blue DR inflates covariance, purple both); the fill shows
# which module actually won. This is what lets us state the XFMR inversion
# honestly ("the Transformer adapter opens covariance headroom in the
# bias-dominated regime") instead of the figure looking broken.
# x-axis name is "heavy tails" not "covariance" (verified 2026-07-13): on
# EVERY positive-headroom sequence x == T-1 exactly (the tail term); H_w
# never contributes because the fused filter always beats the raw sensor
# (H_w<0 everywhere). So the right side literally IS the tail ratio.
axa.text(-0.98, 3.9, "Bias-dominated", fontsize=24, fontweight="bold",
         color=C_ADP, va="top")
axa.text(1.85, 3.9, "Bias + heavy tails", fontsize=24, fontweight="bold",
         color=C_BOTH, va="top", ha="right")
axa.text(1.85, 0.315, "Heavy-tailed", fontsize=24, fontweight="bold",
         color=C_DR, va="bottom", ha="right")
axa.text(-0.98, 0.315, "Well-specified", fontsize=24, fontweight="bold",
         color="#666666", va="bottom")

MK = {"const1": ("o", 380), "const2": ("s", 330),
      "const3": ("^", 430), "const4": ("D", 300)}
# fill = most economical deployment, strictly by net score
CATS = ["neither", "adaptive", "dr", "drift"]
for i, r in enumerate(d.itertuples()):
    mk, ms = MK[r.const]
    score_best = sc[i].max()
    score_presc = sc[i][CATS.index(r.region)]
    fill_cat = r.obs
    if fill_cat != r.region:
        print(f"  off-region: {r.unit}  region={r.region} fill={fill_cat} "
              f"(presc net {score_presc:+.1f} vs best {score_best:+.1f})")
    # FIG_STYLE variants (comparison renders, 2026-07-12):
    #   winner (default) - fill = observed best deployment
    #   fade   - winner fill, saturation ~ decisiveness (ties recede)
    #   clean  - fill = region color (map is purely diagnostic)
    #   ring   - region color + ink ring on decisive exceptions (>5 net pts)
    if STYLE == "fade":
        if fill_cat == "neither":
            axa.scatter(r.x, r.ratio, s=ms, marker=mk, facecolor="white",
                        edgecolor=INK, lw=2.4, zorder=5)
        else:
            w = float(np.clip(r.margin / 10.0, 0.25, 1.0))
            axa.scatter(r.x, r.ratio, s=ms, marker=mk,
                        facecolor=blend(CAT_COLOR[fill_cat], w),
                        edgecolor=CAT_COLOR[fill_cat], lw=1.4, zorder=5)
    elif STYLE == "clean":
        axa.scatter(r.x, r.ratio, s=ms, marker=mk,
                    facecolor=CAT_COLOR[r.region],
                    edgecolor="white", lw=1.6, zorder=5)
    elif STYLE == "group":
        # in-region fills draw ABOVE off-region ones (user 2026-07-12): the
        # region's own color reads as dominant where dots overlap
        zo = 6 if r.grp_obs == r.region else 5
        if r.grp_obs == "neither":
            axa.scatter(r.x, r.ratio, s=ms, marker=mk, facecolor="white",
                        edgecolor=INK, lw=2.4, zorder=zo)
        else:
            axa.scatter(r.x, r.ratio, s=ms, marker=mk,
                        facecolor=CAT_COLOR[r.grp_obs],
                        edgecolor="white", lw=1.6, zorder=zo)
    elif STYLE == "glyph":
        base = C_NONE if r.grp_obs == "neither" else CAT_COLOR[r.grp_obs]
        axa.scatter(r.x, r.ratio, s=ms*0.4, marker=mk,
                    facecolor=blend(base, 0.4), edgecolor="none", zorder=4)
    elif STYLE == "ring":
        decisive = (fill_cat != r.region) and (score_best - score_presc > PAY)
        axa.scatter(r.x, r.ratio, s=ms, marker=mk,
                    facecolor=CAT_COLOR[r.region],
                    edgecolor=INK if decisive else "white",
                    lw=2.6 if decisive else 1.6, zorder=6 if decisive else 5)
    elif fill_cat == "neither":
        axa.scatter(r.x, r.ratio, s=ms, marker=mk, facecolor="white",
                    edgecolor=INK, lw=2.4, zorder=5)
    else:
        axa.scatter(r.x, r.ratio, s=ms, marker=mk,
                    facecolor=CAT_COLOR[fill_cat],
                    edgecolor="white", lw=1.6, zorder=5)



if STYLE == "glyph":
    # one large glyph per deployment environment at the trial-median position
    for (cst, man), g in d.groupby(["const", "manual"]):
        cat = g.grp_obs.iloc[0]
        mk, ms = MK[cst]
        fc = "white" if cat == "neither" else CAT_COLOR[cat]
        ec = INK if cat == "neither" else "white"
        axa.scatter(g.x.median(), g.ratio.median(), s=ms*1.8, marker=mk,
                    facecolor=fc, edgecolor=ec, lw=2.2, zorder=6)

axa.axvline(XB, color=INK2, ls=(0, (6, 3)), lw=1.6)
axa.axhline(1, color=INK2, ls=(0, (6, 3)), lw=1.6)
axa.set_yscale("log")
axa.set_yticks([0.5, 1, 2, 4])
axa.set_yticklabels(["0.5", "1", "2", "4"])
axa.set_xlim(XL, XR)
axa.set_ylim(YB, YT)
axa.tick_params(labelsize=21)
# bare axis names only (user 2026-07-09); meaning + formulas live in the
# snippet's caption and text
axa.set_xlabel("DR headroom", fontsize=25, color=INK)
axa.set_ylabel("bias-to-noise ratio of the ESKF error", fontsize=25, color=INK)

# hand-built legend (axes-fraction coords): exact alignment, no auto-layout
def _lm(x, y, marker, size, label):
    axa.scatter([x], [y], transform=axa.transAxes, s=size, marker=marker,
                facecolor="#d7d6d0", edgecolor="#52514e", lw=1.4, zorder=8)
    axa.text(x + 0.033, y, label, transform=axa.transAxes, fontsize=21,
             va="center", ha="left", color=INK, zorder=8)

_lm(0.033, 0.285, "o", 280, "const1"); _lm(0.21, 0.285, "s", 250, "const2")
_lm(0.033, 0.215, "^", 320, "const3"); _lm(0.21, 0.215, "D", 220, "const4")
if STYLE == "ring":
    axa.scatter([0.033], [0.145], transform=axa.transAxes, s=220, marker="D",
                facecolor="#d7d6d0", edgecolor=INK, lw=2.6, zorder=8)
    axa.text(0.066, 0.145, "decisive exception", transform=axa.transAxes,
             fontsize=21, va="center", ha="left", color=INK, zorder=8)
elif STYLE != "clean":
    fills = d.grp_obs if STYLE in ("group", "glyph") else d.obs
    if (fills == "neither").any():
        axa.scatter([0.033], [0.145], transform=axa.transAxes, s=220,
                    marker="D", facecolor="white", edgecolor=INK, lw=2.4,
                    zorder=8)
        axa.text(0.066, 0.145, "none dominant", transform=axa.transAxes,
                 fontsize=21, va="center", ha="left", color=INK, zorder=8)

# ===================== (b) evidence by map region =====================
# ALL sequences incl. manual flights (user 2026-07-12, v47 family — the
# tail-supervised adapter now handles manuals, so no scope carve-out).
db = d
# order: WRAP/purple first, then bias/red, then heavy-tailed/blue (user 2026-07-13)
GROUPS = [("drift", "Bias +\nheavy tails"), ("adaptive", "Bias-\ndominated"),
          ("dr", "Heavy-\ntailed")]
DEPLOY = [("pct_adapter", C_ADP), ("pct_dr_alone", C_DR), ("pct_both", C_BOTH)]
W, GAP = 0.62, 0.95
rng = np.random.default_rng(7)   # seeded: deterministic dot jitter
for gi, (reg, gname) in enumerate(GROUPS):
    gg = db[db.region == reg]
    x0 = gi * (3*W + GAP)
    for di, (col, c) in enumerate(DEPLOY):
        xb = x0 + di*W
        med = gg[col].median()
        axb.bar(xb, med, W*0.92, color=c, alpha=0.35, zorder=2,
                edgecolor=c, lw=2.4)
        axb.text(xb, med + 1.4 if med >= 0 else 1.4, f"{med:+.0f}",
                 ha="center", va="bottom", fontsize=22, fontweight="bold",
                 color=INK, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.8))
        jit = rng.uniform(-W*0.26, W*0.26, len(gg))
        axb.scatter(xb + jit, gg[col].values, s=80, color=c,
                    zorder=5, edgecolor="white", lw=0.8)
    axb.text(x0 + W, 0.995, gname, transform=axb.get_xaxis_transform(),
             ha="center", va="top", fontsize=24, fontweight="bold",
             color=CAT_COLOR[reg], linespacing=1.2)
    axb.axvspan(x0 - 0.55, x0 + 2*W + 0.55, color=CAT_COLOR[reg],
                alpha=0.05, zorder=1)

axb.axhline(0, color=INK, lw=1.6, zorder=3)
axb.set_xticks([gi*(3*W + GAP) + di*W for gi in range(3) for di in range(3)])
axb.set_xticklabels(["Adp", "DR", "WRAP"]*3, fontsize=21)
axb.set_xlim(-0.75, 2*(3*W + GAP) + 2*W + 0.75)
axb.set_ylim(-22, 68)   # v47: dr-region adp/WRAP medians reach -16/-14
axb.tick_params(labelsize=21)
axb.set_ylabel("RMSE improvement vs baseline  (%)", fontsize=25, color=INK)
axb.grid(axis="y", color="#e1e0d9", lw=1.0, zorder=0)
axb.set_axisbelow(True)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)

for e in ("png", "pdf"):
    fig.savefig(OUT / f"FIG_REGIMES_UWB{FIG_SUFFIX}.{e}",
                bbox_inches="tight", dpi=140)
print("[v5 UWB paper] saved")
