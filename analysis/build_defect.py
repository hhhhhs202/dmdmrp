"""
Spot welding (Spot-01, item 65235-25800) defect-label construction & analysis.

Inputs
  data/Welding_Data_Set_01.xlsx   (sheets: Raw data / result / data set)
  data/scaled_data.csv            (utf-8 version of the provided scaled_data.csv)

Outputs
  output/Defect.csv               row-level labels (0 normal, 1 dent, 2 lack of fusion, 3 crack)
  output/metrics.txt              numbers quoted in REPORT.md
  output/fig_*.png                figures

Run:  python analysis/build_defect.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

TYPE_NAME = {0: "정상", 1: "파임불량", 2: "용접부족", 3: "크랙발생"}
TYPE_EN = {0: "normal", 1: "T1 dent", 2: "T2 lack of fusion", 3: "T3 crack"}
COLORS = {0: "#c8c8c8", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
RAW4 = ["F", "I", "V", "T"]
PHYS = ["F", "I", "V", "T", "R", "Q"]
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


# ---------------------------------------------------------------- 1. load
xl = pd.read_excel(ROOT / "data/Welding_Data_Set_01.xlsx", sheet_name=None)
raw = xl["Raw data"].copy()
raw.columns = ["idx", "Machine_Name", "Item No", "working time", "t1", "t2", "F", "I", "V", "T"]
res = xl["result"].iloc[:, :6].copy()
res.columns = ["idx", "Machine_Name", "Item No", "working time", "defect", "defect type"]
spec = xl["data set"]

say("== 1. raw data ==")
say("rows", len(raw), "| constant cols:", [c for c in raw.columns if raw[c].nunique() == 1])
say("rows per day:", raw.groupby("working time").size().to_dict())

# ---------------------------------------------------------------- 2. result sheet
say("\n== 2. result sheet (daily defect counts) ==")
pv = res.pivot_table(index="working time", columns="defect type", values="defect", aggfunc="sum")
pv = pv.reindex(sorted(raw["working time"].unique()))
say(pv.to_string())
say("total per type:", res.groupby("defect type")["defect"].sum().to_dict(), "| total", res["defect"].sum())

# ---------------------------------------------------------------- 3. preprocessing
say("\n== 3. preprocessing ==")
ranges = {"F": (1, 12), "I": (12, 18), "V": (1.5, 3.5), "T": (30, 120)}
oor = pd.concat([(raw[k] < a) | (raw[k] > b) for k, (a, b) in ranges.items()], axis=1).any(axis=1)
say("out-of-spec rows (data set sheet ranges):", int(oor.sum()), "| NaN:", int(raw[RAW4].isna().sum().sum()))
dup = raw[raw.duplicated(["working time", "idx"], keep=False)]
say("duplicated (date, idx):", len(dup))
# idx continuity per day
for d, g in raw.groupby("working time"):
    jumps = np.where(np.diff(g["idx"].values) != 1)[0]
    if len(jumps):
        say(f"  idx jump on {d.date()}:", [(int(g.idx.values[j]), int(g.idx.values[j + 1])) for j in jumps])
nonint_t = (raw["T"] % 1 != 0)
say("non-integer weld time rows:", int(nonint_t.sum()), sorted(raw.loc[nonint_t, "T"].unique())[:6])

# physical features: dynamic resistance and heat input (Joule heating Q = I^2 R t = V I t)
raw["R"] = raw["V"] / raw["I"]                  # V / kA  = mOhm
raw["Q"] = raw["V"] * raw["I"] * raw["T"]       # V * kA * ms = J
raw["date"] = raw["working time"].dt.date.astype(str)

# anomalous-process block: force outside normal operating band
normal_band = raw.loc[raw["F"] < 3, "F"]
raw["force_block"] = raw["F"] > normal_band.quantile(0.999) + 0.3
say("force-block rows by day:", raw.groupby("date")["force_block"].sum().to_dict())

# ---------------------------------------------------------------- 4. date effect
say("\n== 4. date effect ==")
day = raw.groupby("date").agg(n=("F", "size"), F_mean=("F", "mean"), F_med=("F", "median"),
                             I_med=("I", "median"), Q_med=("Q", "median"), block=("force_block", "sum"))
dd = pv.fillna(0).sum(axis=1)
dd.index = dd.index.date.astype(str)
day["defects"] = dd
day.loc["2020-03-27", "defects"] = np.nan
day["rate_%"] = day["defects"] / day["n"] * 100
say(day.round(3).to_string())
ok = day.dropna()
say("corr(defects, n) = %.2f | corr(defects, has_block) = %.2f | corr(rate, has_block) = %.2f"
    % (ok.defects.corr(ok.n), ok.defects.corr((ok.block > 0).astype(float)),
       ok["rate_%"].corr((ok.block > 0).astype(float))))

# day-to-day duplication (3-gram of rounded rows)
seqs = {d: g[RAW4].round(3).astype(str).agg("|".join, axis=1).tolist() for d, g in raw.groupby("date")}
dates = list(seqs)
ov = pd.DataFrame(index=dates, columns=dates, dtype=float)
for a in dates:
    sa = set(zip(seqs[a], seqs[a][1:], seqs[a][2:]))
    for b in dates:
        sb = set(zip(seqs[b], seqs[b][1:], seqs[b][2:]))
        ov.loc[a, b] = len(sa & sb) / min(len(sa), len(sb))
say("3-gram overlap between days (1.0 = one day is contained in the other):")
say(ov.round(2).to_string())

# ---------------------------------------------------------------- 5. labels
say("\n== 5. defect labeling ==")
# robust z-scores against the whole line (median / IQR)
z = pd.DataFrame(RobustScaler().fit_transform(raw[PHYS]), columns=PHYS, index=raw.index)


def rank01(s):
    return s.rank(pct=True)


def scores(zdf, physical=True):
    if physical:
        # T1 dent: excessive force + high heat  -> electrode sinks into the sheet
        s1 = zdf.F + zdf.Q
        # T2 lack of fusion: not enough Joule heat (low Q, low contact resistance)
        s2 = -zdf.Q - zdf.R
        # T3 crack: high heat with insufficient forging force (+ current surge)
        s3 = zdf.Q + zdf.I - zdf.F
    else:
        # same idea using only the four measured signals
        s1 = zdf.F
        s2 = -(zdf.I + zdf.V + zdf["T"])
        s3 = zdf.I + zdf.V + zdf["T"] - zdf.F
    return pd.DataFrame({1: s1, 2: s2, 3: s3})


def assign(score_df, how="physics"):
    """Per day: put exactly k_type rows in each type, maximizing total (rank-normalized) score.
    Solved as a linear assignment problem (rows x slots) so the result does not depend on type order."""
    lab = pd.Series(0, index=raw.index)
    for d, g in raw.groupby("date"):
        if d == "2020-03-27":            # no inspection record for this day
            lab[g.index] = -1
            continue
        k = pv.loc[pd.Timestamp(d)].fillna(0).astype(int)
        slots = [t for t in (1, 2, 3) for _ in range(k.get(t, 0))]
        if not slots:
            continue
        if how == "random":
            pick = RNG.choice(g.index, len(slots), replace=False)
            lab[pick] = slots
            continue
        sc = score_df.loc[g.index].apply(rank01)       # within-day percentile per type
        # prefer rows that are extreme for one type AND not for the others
        margin = sc.sub(sc.mean(axis=1), axis=0)
        cost = -np.column_stack([sc[t].values + 0.5 * margin[t].values for t in slots])
        r_i, c_i = linear_sum_assignment(cost)
        lab[g.index[r_i]] = [slots[c] for c in c_i]
    return lab


raw["label_phys"] = assign(scores(z, True))
raw["label_raw4"] = assign(scores(z, False))
raw["label_rand"] = assign(None, how="random")

for c in ["label_phys", "label_raw4", "label_rand"]:
    say(c, raw[c].value_counts().sort_index().to_dict())

# ---------------------------------------------------------------- 6. Defect.csv
defect = raw[["idx", "Machine_Name", "Item No", "working time", "t1", "t2", "F", "I", "V", "T", "R", "Q"]].copy()
defect.columns = ["idx", "Machine_Name", "Item No", "working time", "Thickness 1(mm)", "Thickness 2(mm)",
                  "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)",
                  "R_dyn(mOhm)", "Q_heat(J)"]
sc_phys = scores(z, True)
defect["score_T1"] = sc_phys[1].round(3)
defect["score_T2"] = sc_phys[2].round(3)
defect["score_T3"] = sc_phys[3].round(3)
lab = raw["label_phys"]
defect = defect.round({"weld force(bar)": 2, "weld current(kA)": 2, "weld Voltage(v)": 3, "weld time(ms)": 2,
                       "R_dyn(mOhm)": 5, "Q_heat(J)": 2})
defect["defect"] = pd.Series(np.where(lab > 0, 1, 0), index=lab.index).where(lab >= 0).astype("Int64")
defect["defect type"] = lab.where(lab >= 0).astype("Int64")
defect["defect name"] = lab.map(TYPE_NAME).fillna("미검사(결과없음)")
defect["force_block"] = raw["force_block"].astype(int)
defect["working time"] = defect["working time"].dt.date
defect.to_csv(OUT / "Defect.csv", index=False, encoding="utf-8-sig")
say("Defect.csv written:", defect.shape)

# check: reproduce the result sheet from Defect.csv
chk = defect[defect["defect"] == 1].groupby(["working time", "defect type"]).size().unstack(fill_value=0)
say("re-aggregated from Defect.csv:\n" + chk.to_string())

say("\nprofile of labeled rows (median):")
say(raw[raw.label_phys >= 0].groupby("label_phys")[PHYS + ["force_block"]].median().round(3).to_string())

# ---------------------------------------------------------------- 7. PCA / separability
say("\n== 7. PCA & separability ==")
lab_mask = raw.label_phys >= 0


raw["logF"] = np.log(raw["F"])
stable = ~raw["force_block"]          # fit scaler/PCA on the stable regime only


def pca_embed(cols):
    sc = StandardScaler().fit(raw.loc[stable, cols])
    X = sc.transform(raw[cols])
    p = PCA(n_components=min(len(cols), 4)).fit(X[stable.values])
    return p.transform(X), p


def separability(emb, labels):
    m = (labels > 0).values
    return silhouette_score(emb[m][:, :2], labels[m]) if labels[m].nunique() > 1 else np.nan


def loo_day_acc(cols, labels):
    """Train on 7 days, predict the held-out day, only defect rows (3-class).  LDA keeps it simple."""
    m = labels > 0
    X = StandardScaler().fit(raw.loc[stable, cols]).transform(raw.loc[m, cols])
    pred = cross_val_predict(LinearDiscriminantAnalysis(), X, labels[m], groups=raw.loc[m, "date"],
                             cv=LeaveOneGroupOut())
    return (pred == labels[m].values).mean()


# Q and R are deterministic functions of the measured signals: how much new information do they carry?
from sklearn.linear_model import LinearRegression
for tgt, src in [("Q", ["V", "I", "T"]), ("R", ["V", "I"])]:
    r2 = LinearRegression().fit(raw[src], raw[tgt]).score(raw[src], raw[tgt])
    say(f"linear R^2 of {tgt} from {src}: {r2:.4f}")

FEATSETS = [("raw4 (F,I,V,T)", RAW4), ("raw4 + R,Q", PHYS), ("physics (logF,Q,R)", ["logF", "Q", "R"])]
rows = []
embs = {}
for feat_name, cols in FEATSETS:
    emb, p = pca_embed(cols)
    embs[feat_name] = (emb, p)
    say(feat_name, "explained var (stable regime):", p.explained_variance_ratio_.round(3))
    say("   PC1:", dict(zip(cols, p.components_[0].round(2))), "| PC2:", dict(zip(cols, p.components_[1].round(2))))
    for lname in ["label_rand", "label_raw4", "label_phys"]:
        rows.append({"features": feat_name, "labels": lname,
                     "silhouette_PC12": separability(emb, raw[lname]),
                     "LODO_LDA_acc": loo_day_acc(cols, raw[lname])})
tbl = pd.DataFrame(rows)
say(tbl.round(3).to_string(index=False))
say("(chance accuracy for 3 balanced classes ~0.33)")

# ---------------------------------------------------------------- 8. scaled_data
say("\n== 8. scaled_data.csv ==")
sd = pd.read_csv(ROOT / "data/scaled_data.csv", index_col=0)
sd.columns = RAW4
say("shape", sd.shape, "| NaN per col", sd.isna().sum().to_dict())
say("share of rows with value < 0.01 (per col):", (sd < 0.01).mean().round(3).to_dict())
res_unit = {"F": 0.01, "I": 0.01, "V": 0.001, "T": 1}
recon = {}
for k in RAW4:
    u = np.sort(sd[k].dropna().unique())
    step = np.diff(u)[np.diff(u) > 1e-9].min()
    span = res_unit[k] / step
    q = np.round(sd[k] / step)
    off = np.nanmedian(raw[k] - q * res_unit[k])
    recon[k] = off + q * res_unit[k]
    say(f"{k}: implied MinMax range = [{off:.3f}, {off + span:.2f}]  (raw range [{raw[k].min()}, {raw[k].max()}],"
        f" spec {ranges[k]})")
recon = pd.DataFrame(recon)
rec_oor = pd.concat([(recon[k] < a) | (recon[k] > b) for k, (a, b) in ranges.items()], axis=1)
say("out-of-spec values in scaled_data (after inverse scaling):", rec_oor.sum().to_dict(),
    "| rows:", int(rec_oor.any(axis=1).sum()))
good = ~rec_oor.any(axis=1) & recon.notna().all(axis=1)
say("row-wise corr with raw (clean rows):", {k: round(np.corrcoef(recon.loc[good, k], raw.loc[good, k])[0, 1], 3)
                                            for k in RAW4})
say("examples of spikes:\n" + recon[rec_oor.any(axis=1)].head(8).to_string())

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# F1: daily defect composition
fig, ax = plt.subplots(figsize=(8, 3.6))
pvp = pv.fillna(0)
bottom = np.zeros(len(pvp))
xs = [d.strftime("%m-%d") for d in pvp.index]
for t in (1, 2, 3):
    ax.bar(xs, pvp[t], bottom=bottom, color=COLORS[t], label=TYPE_EN[t])
    bottom += pvp[t].values
ax.bar(["03-27"], [0], color="none")
ax.text(xs.index("03-27"), 0.2, "no\nrecord", ha="center", fontsize=8, color="gray")
ax2 = ax.twinx()
ax2.plot(xs, day["n"].values, "k.--", lw=1, label="welds/day")
ax2.set_ylabel("welds per day")
ax.set_ylabel("defects")
ax.set_title("Result sheet: defects per day by type (bars) vs production volume (line)")
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_daily_defects.png")
plt.close(fig)

# F2: date effect - force per day + block
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
order = sorted(raw.date.unique())
axs[0].boxplot([raw.loc[raw.date == d, "F"] for d in order], tick_labels=[d[5:] for d in order], showfliers=True,
               flierprops={"ms": 2})
axs[0].set_ylabel("weld force (bar)")
axs[0].set_title("Force by day: 4 days carry the identical high-force block")
im = axs[1].imshow(ov.values.astype(float), cmap="Greys", vmin=0, vmax=1)
axs[1].set_xticks(range(len(order)), [d[5:] for d in order], rotation=90)
axs[1].set_yticks(range(len(order)), [d[5:] for d in order])
axs[1].set_title("Sequence overlap between days (3-gram)")
fig.colorbar(im, ax=axs[1], fraction=0.046)
fig.tight_layout()
fig.savefig(OUT / "fig2_date_effect.png")
plt.close(fig)

# F3: PCA grid (features x labeling)
fig, axs = plt.subplots(3, 3, figsize=(14, 12.5))
for i, (feat_name, (emb, p)) in enumerate(embs.items()):
    for j, (lname, ltitle) in enumerate([("label_rand", "random within day"),
                                         ("label_raw4", "rule on raw F,I,V,T"),
                                         ("label_phys", "physics rule (Q,R)")]):
        ax = axs[i, j]
        L = raw[lname]
        bg = L == 0
        sub = RNG.choice(np.where(bg)[0], 3000, replace=False)
        ax.scatter(emb[sub, 0], emb[sub, 1], s=4, c=COLORS[0], label="normal (sample)")
        for t in (1, 2, 3):
            m = (L == t).values
            ax.scatter(emb[m, 0], emb[m, 1], s=40, c=COLORS[t], edgecolor="k", lw=0.4, label=TYPE_EN[t])
        r_ = tbl[(tbl.features == feat_name) & (tbl.labels == lname)].iloc[0]
        ax.set_title(f"{feat_name} | labels: {ltitle}\nsilhouette={r_.silhouette_PC12:.2f}  "
                     f"leave-one-day-out acc={r_.LODO_LDA_acc:.2f}", fontsize=9)
        ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%})")
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
axs[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_pca_grid.png")
plt.close(fig)

# F4: physical plane
fig, ax = plt.subplots(figsize=(6.5, 4.8))
L = raw.label_phys
sub = RNG.choice(np.where(L == 0)[0], 4000, replace=False)
ax.scatter(raw.F.values[sub], raw.Q.values[sub], s=4, c=COLORS[0], label="normal (sample)")
for t in (1, 2, 3):
    m = L == t
    ax.scatter(raw.F[m], raw.Q[m], s=45, c=COLORS[t], edgecolor="k", lw=0.4, label=TYPE_EN[t])
ax.set_xscale("log")
ax.set_xlabel("weld force F (bar, log)")
ax.set_ylabel("heat input Q = V·I·t (J)")
ax.set_title("Labeled defects in the physical (F, Q) plane")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig4_force_heat_plane.png")
plt.close(fig)

# F5: scaled vs proper scaling
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
axs[0].hist(sd["F"].dropna(), bins=100, color="#555")
axs[0].set_yscale("log")
axs[0].set_title("scaled_data: force (MinMax incl. spikes)\n99% of rows squeezed below 0.01", fontsize=9)
mm = MinMaxScaler().fit_transform(raw[["F"]])
axs[1].hist(mm, bins=100, color="#1f77b4")
axs[1].set_yscale("log")
axs[1].set_title("MinMax on cleaned raw force", fontsize=9)
axs[2].hist(RobustScaler().fit_transform(raw[["I"]]), bins=60, color="#2ca02c")
axs[2].set_title("RobustScaler on raw current (recommended)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig5_scaling.png")
plt.close(fig)

# F6: scaled vs raw row alignment (inverse-scaled)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
sl = slice(1800, 2200)
axs[0].plot(raw.F.values[sl], lw=1, label="raw sheet")
axs[0].plot(recon.F.values[sl], lw=1, alpha=0.7, label="scaled_data (inverse-scaled)")
axs[0].set_ylim(0, 20)
axs[0].set_title("Force, rows 1800-2200: same process, spikes/NaN injected", fontsize=9)
axs[0].legend(fontsize=8, frameon=False)
axs[1].plot(recon["T"].values, lw=0.6, color="#d62728")
axs[1].set_yscale("log")
axs[1].set_title("scaled_data weld time after inverse scaling (spec 30-120 ms)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig6_scaled_vs_raw.png")
plt.close(fig)

(OUT / "metrics.txt").write_text("\n".join(log), encoding="utf-8")
print("\ndone")
