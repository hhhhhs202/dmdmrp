"""
STAGE 1 - unsupervised
  (a) process-drift segments ("defect 구간"): contiguous runs where force/voltage leave the robust band
  (b) safe process window: envelope of the stable regime after removing point anomalies
  (c) defect candidates: every weld inside a drift segment + point anomalies in the stable regime
      voted by 3 detectors (safe-window rule, robust Mahalanobis, Isolation Forest)

Output (output/stage1/)
  defect_candidates.csv    candidates ONLY (normal welds are not written)
  defect_segments.csv      one row per drift segment
  safe_process_window.csv  per-variable safe operating range
  stage1_log.txt, fig_s1_*.png
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from common import OUT, load_raw, mod_z, preprocess

warnings.filterwarnings("ignore")
S1 = OUT / "stage1"
S1.mkdir(parents=True, exist_ok=True)
Z_CUT = 3.5          # Iglewicz-Hoaglin cut-off for the modified z-score
GAP = 5              # anomalies closer than this (welds) are merged into one segment
MIN_SEG = 10         # a run shorter than this is a point anomaly, not a segment
IF_CONT = 0.02       # Isolation Forest contamination inside the stable regime
MD_Q = 0.999         # chi-square quantile for the robust Mahalanobis distance
DAY_TOP = 0.02       # additionally: top 2% anomaly score of each day's stable welds
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


raw, counts = load_raw()
df, info = preprocess(raw)
say("== preprocessing ==", info, "| rows kept:", len(df))

# ------------------------------------------------------------------ (a) drift segments
zF, _, _ = mod_z(df["F"].values)
zV, _, _ = mod_z(df["V"].values)
df["mz_F_global"], df["mz_V_global"] = zF, zV
point = (np.abs(zF) > Z_CUT) | (np.abs(zV) > Z_CUT)

df["segment_id"] = ""
segs = []
for d, g in df.groupby("date"):
    pos = g.index[point[g.index]]
    if len(pos) == 0:
        continue
    # merge anomalies separated by < GAP welds
    runs, start, prev = [], pos[0], pos[0]
    for p in pos[1:]:
        if p - prev > GAP:
            runs.append((start, prev))
            start = p
        prev = p
    runs.append((start, prev))
    for a, b in runs:
        if b - a + 1 >= MIN_SEG:
            sid = f"{d}_S{len(segs) + 1:02d}"
            df.loc[a:b, "segment_id"] = sid
            seg = df.loc[a:b]
            segs.append({"segment_id": sid, "date": d, "idx_start": int(seg.idx.iloc[0]),
                         "idx_end": int(seg.idx.iloc[-1]), "n_welds": len(seg),
                         "F_max": seg.F.max(), "F_median": seg.F.median(), "V_min": seg.V.min(),
                         "V_max": seg.V.max(), "Q_min": round(seg.Q.min(), 1), "Q_max": round(seg.Q.max(), 1),
                         "R_min": round(seg.R.min(), 5)})
segs = pd.DataFrame(segs)
df["in_segment"] = df["segment_id"] != ""
say("\n== (a) drift segments ==")
say(segs.to_string(index=False))

# ------------------------------------------------------------------ (b)+(c) stable regime detectors
st = ~df["in_segment"]
say("\n== stable regime ==  rows:", int(st.sum()))

# D1: safe-window rule, modified z fitted on stable rows only
WIN_VARS = ["F", "I", "V", "T", "R", "Q"]
d1 = pd.Series(False, index=df.index)
zcols = {}
for k in WIN_VARS:
    _, c, s = mod_z(df.loc[st, k].values)
    z = (df[k] - c) / s
    df[f"mz_{k}"] = z.round(3)
    zcols[k] = (c, s)
    d1 |= z.abs() > Z_CUT
df["D1_window"] = d1 & st

# D2: robust Mahalanobis on the 4 measured signals
base4 = ["F", "I", "V", "T"]
sc4 = StandardScaler().fit(df.loc[st, base4])
mcd = MinCovDet(support_fraction=0.9, random_state=0).fit(sc4.transform(df.loc[st, base4]))
df["mahal_d2"] = mcd.mahalanobis(sc4.transform(df[base4]))
md_cut = chi2.ppf(MD_Q, df=len(base4))
df["D2_mahal"] = (df["mahal_d2"] > md_cut) & st

# D3: Isolation Forest on physical + temporal features
IF_FEATS = ["F", "I", "V", "T", "R", "Q", "d_F", "d_I", "d_V", "step_F", "step_I", "step_V"]
scf = StandardScaler().fit(df.loc[st, IF_FEATS])
iso = IsolationForest(n_estimators=400, contamination=IF_CONT, random_state=0).fit(scf.transform(df.loc[st, IF_FEATS]))
df["iforest_score"] = -iso.score_samples(scf.transform(df[IF_FEATS]))      # higher = more anomalous
df["D3_iforest"] = (iso.predict(scf.transform(df[IF_FEATS])) == -1) & st

df["votes"] = df[["D1_window", "D2_mahal", "D3_iforest"]].sum(axis=1)

# continuous anomaly score = mean percentile rank of the three detectors (ranked inside the stable regime)
maxz = pd.concat([df[f"mz_{k}"].abs() for k in WIN_VARS], axis=1).max(axis=1)
score = (maxz[st].rank(pct=True) + df.loc[st, "mahal_d2"].rank(pct=True) + df.loc[st, "iforest_score"].rank(pct=True)) / 3
df["anomaly_score"] = score.reindex(df.index).round(4)          # NaN inside drift segments
# day-relative rank: each day is its own reference (daily set-up / electrode dressing / inspection unit)
df["day_rank_pct"] = df[st].groupby("date")["anomaly_score"].rank(pct=True, ascending=False).reindex(df.index)

df["point_candidate"] = st & (df["votes"] >= 2)
df["day_candidate"] = st & ~df["point_candidate"] & (df["day_rank_pct"] <= DAY_TOP)
df["candidate"] = df["in_segment"] | df["point_candidate"] | df["day_candidate"]
df["candidate_source"] = np.select([df["in_segment"], df["point_candidate"], df["day_candidate"]],
                                   ["drift_segment", "point_anomaly", "day_relative"], "")

say("detector hits (stable regime):", {c: int(df[c].sum()) for c in ["D1_window", "D2_mahal", "D3_iforest"]})
say("point candidates (>=2 votes):", int(df.point_candidate.sum()), "| day-relative:", int(df.day_candidate.sum()),
    "| segment welds:", int(df.in_segment.sum()),
    "| total candidates:", int(df.candidate.sum()), f"({df.candidate.mean():.1%})")
per_day = df.groupby("date").agg(welds=("F", "size"), seg=("in_segment", "sum"), point=("point_candidate", "sum"), dayrel=("day_candidate", "sum"),
                                 cand=("candidate", "sum"))
per_day["result_defects"] = counts.sum(axis=1).reindex(per_day.index)
say(per_day.to_string())
short = per_day[(per_day.cand < per_day.result_defects)]
say("days with fewer candidates than recorded defects:", list(short.index))

# ------------------------------------------------------------------ (b) safe process window
safe = df[st & ~df["candidate"]]
win = []
for k, unit in [("F", "bar"), ("I", "kA"), ("V", "V"), ("T", "ms"), ("R", "mOhm"), ("Q", "J")]:
    c, s = zcols[k]
    win.append({"variable": k, "unit": unit, "median": round(c, 5), "robust_sigma": round(s, 5),
                "rule_low(med-3.5s)": round(c - Z_CUT * s, 5), "rule_high(med+3.5s)": round(c + Z_CUT * s, 5),
                "observed_safe_min": round(safe[k].min(), 5), "observed_safe_max": round(safe[k].max(), 5),
                "p0.5": round(safe[k].quantile(0.005), 5), "p99.5": round(safe[k].quantile(0.995), 5)})
win = pd.DataFrame(win)
win.to_csv(S1 / "safe_process_window.csv", index=False, encoding="utf-8-sig")
say("\n== (b) safe process window ==")
say(win.to_string(index=False))
say(f"joint condition: robust Mahalanobis d^2 on (F,I,V,T) <= {md_cut:.2f} (chi2 {MD_Q}, df=4)")

# ------------------------------------------------------------------ outputs
cols = ["uid", "date", "idx", "seq", "F", "I", "V", "T", "R", "Q", "P", "q_per_F",
        "d_F", "d_I", "d_V", "d_R", "d_Q", "step_F", "step_I", "step_V",
        "mz_F", "mz_I", "mz_V", "mz_T", "mz_R", "mz_Q", "mahal_d2", "iforest_score",
        "D1_window", "D2_mahal", "D3_iforest", "votes", "anomaly_score", "day_rank_pct", "candidate_source", "segment_id"]
cand = df.loc[df.candidate, cols].copy()
for c in ["D1_window", "D2_mahal", "D3_iforest"]:
    cand[c] = cand[c].astype(int)
cand = cand.round({"F": 2, "I": 2, "V": 3, "T": 2, "R": 5, "Q": 2, "P": 3, "q_per_F": 2, "d_F": 3, "d_I": 3, "d_V": 4, "d_R": 5, "d_Q": 2,
                   "step_F": 3, "step_I": 3, "step_V": 4, "mahal_d2": 3, "iforest_score": 4,
                   "day_rank_pct": 4})
cand.to_csv(S1 / "defect_candidates.csv", index=False, encoding="utf-8-sig")
segs.round({"F_max": 2, "F_median": 2, "V_min": 3, "V_max": 3}).to_csv(S1 / "defect_segments.csv", index=False, encoding="utf-8-sig")
say("\nwritten:", S1 / "defect_candidates.csv", cand.shape)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
# 1) one block day timeline
d = 20200325
g = df[df.date == d]
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, k, lab in zip(axs, ["F", "V", "Q"], ["force (bar)", "voltage (V)", "heat Q (J)"]):
    ax.plot(g.seq, g[k], lw=0.7, color="#555")
    for _, s in segs[segs.date == d].iterrows():
        ss = g[g.segment_id == s.segment_id].seq
        ax.axvspan(ss.min(), ss.max(), color="#ffcc80", alpha=0.5, lw=0)
    pc = g[g.point_candidate]
    ax.scatter(pc.seq, pc[k], s=14, color="#d62728", zorder=3, label="point candidate")
    dc = g[g.day_candidate]
    ax.scatter(dc.seq, dc[k], s=14, color="#9467bd", zorder=3, label="day-relative candidate")
    if k in zcols:
        c, s_ = zcols[k]
        ax.axhspan(c - Z_CUT * s_, c + Z_CUT * s_, color="#2ca02c", alpha=0.12, lw=0, label="safe window")
    ax.set_ylabel(lab)
axs[0].set_title(f"{d}: drift segment (orange), point (red) / day-relative (purple) candidates, safe window (green)")
axs[0].legend(fontsize=8, frameon=False, loc="upper left")
axs[-1].set_xlabel("weld sequence in the day")
fig.tight_layout()
fig.savefig(S1 / "fig_s1_timeline.png")
plt.close(fig)

# 2) stable-regime PCA with candidates
X = StandardScaler().fit_transform(df.loc[st, ["F", "I", "V", "T", "R", "Q"]])
emb = PCA(2).fit_transform(X)
m = df.loc[st, "point_candidate"].values
m2 = df.loc[st, "day_candidate"].values
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(emb[~(m | m2), 0], emb[~(m | m2), 1], s=4, c="#c8c8c8", label="stable & normal")
ax.scatter(emb[m2, 0], emb[m2, 1], s=16, c="#9467bd", label="day-relative candidate")
ax.scatter(emb[m, 0], emb[m, 1], s=16, c="#d62728", label="point candidate")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Stable regime (PCA): point-anomaly candidates")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(S1 / "fig_s1_pca_candidates.png")
plt.close(fig)

# 3) safe window in (I, Q) and (F, Q)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (x, y) in zip(axs, [("I", "Q"), ("F", "Q")]):
    nn = df[~df.candidate]
    ax.scatter(nn[x], nn[y], s=3, c="#c8c8c8", label="normal")
    ax.scatter(df.loc[df.in_segment, x], df.loc[df.in_segment, y], s=5, c="#ff9f40", label="drift segment")
    ax.scatter(df.loc[df.day_candidate, x], df.loc[df.day_candidate, y], s=14, c="#9467bd", label="day-relative candidate")
    ax.scatter(df.loc[df.point_candidate, x], df.loc[df.point_candidate, y], s=14, c="#d62728", label="point candidate")
    (cx, sx), (cy, sy) = zcols[x], zcols[y]
    ax.add_patch(plt.Rectangle((cx - Z_CUT * sx, cy - Z_CUT * sy), 2 * Z_CUT * sx, 2 * Z_CUT * sy,
                               fill=False, ec="#2ca02c", lw=1.5, label="safe window"))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if x == "F":
        ax.set_xscale("log")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Safe process window and stage-1 candidates")
fig.tight_layout()
fig.savefig(S1 / "fig_s1_safe_window.png")
plt.close(fig)

(S1 / "stage1_log.txt").write_text("\n".join(log), encoding="utf-8")
