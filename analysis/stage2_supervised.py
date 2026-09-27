"""
STAGE 2 - supervised defect-type classification on the stage-1 candidates
  1. weak labels: the result sheet only gives (date, type) -> count.  Inside each labelled day the
     k_type welds are chosen among the candidates by physics scores (Joule heat / force), solved as an
     assignment problem.  Other candidates of that day get 0 (= candidate but normal).
  2. learn the selection criteria:
       - DecisionTree (depth 4)  -> human-readable rules (the "선별 기준")
       - RandomForest            -> final classifier
     evaluated with leave-one-day-out CV; the decision threshold on P(defect) is tuned on the
     out-of-fold predictions.
  3. apply to every candidate (incl. the un-inspected day 03-27) and keep ONLY the welds whose final
     type is 1/2/3.

Output (output/stage2/)
  final_defects.csv       final defects only (candidate-but-normal rows are dropped)
  training_labels.csv     weak labels used for training (candidates of labelled days)
  selection_rules.txt     decision-tree rules + feature importance
  stage2_log.txt, fig_s2_*.png
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier, export_text

from common import COLORS, OUT, TYPE_EN, TYPE_NAME, UNLABELED_DAY, load_raw

warnings.filterwarnings("ignore")
S2 = OUT / "stage2"
S2.mkdir(parents=True, exist_ok=True)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


_, counts = load_raw()
cand = pd.read_csv(OUT / "stage1/defect_candidates.csv")
say("candidates:", cand.shape, "| by source:", cand.candidate_source.value_counts().to_dict())

# ------------------------------------------------------------------ 1. weak labels (physics + assignment)
# robust z-scores (median / robust sigma of the stable regime) come from stage 1: mz_*
PHYS_SCORE = {
    1: cand.mz_F + cand.mz_Q,               # dent: excessive force + high heat
    2: -cand.mz_Q - cand.mz_R,              # lack of fusion: not enough Joule heat
    3: cand.mz_Q + cand.mz_I - cand.mz_F,   # crack: high heat, insufficient forging force
}
score = pd.DataFrame(PHYS_SCORE)
cand["y"] = np.nan
for d, g in cand.groupby("date"):
    if d == UNLABELED_DAY or d not in counts.index:
        continue
    k = counts.loc[d]
    slots = [t for t in (1, 2, 3) for _ in range(int(k.get(t, 0)))]
    cand.loc[g.index, "y"] = 0
    if not slots:
        continue
    pct = score.loc[g.index].rank(pct=True)
    margin = pct.sub(pct.mean(axis=1), axis=0)          # favour welds extreme for ONE type only
    cost = -np.column_stack([pct[t].values + 0.5 * margin[t].values for t in slots])
    r_i, c_i = linear_sum_assignment(cost)
    cand.loc[g.index[r_i], "y"] = [slots[c] for c in c_i]

lab = cand[cand.y.notna()].copy()
lab["y"] = lab.y.astype(int)
say("weak labels (labelled days):", lab.y.value_counts().sort_index().to_dict())
lab[["uid", "date", "idx", "F", "I", "V", "T", "R", "Q", "candidate_source", "y"]].assign(
    y_name=lab.y.map(TYPE_NAME)).to_csv(S2 / "training_labels.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 2. learn the criteria
FEATS = ["F", "I", "V", "T", "R", "Q", "q_per_F", "d_F", "d_I", "d_V", "d_Q", "step_F", "step_I", "step_V"]
X, y, grp = lab[FEATS].values, lab.y.values, lab.date.values


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=-1)


def tree():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=0)


# leave-one-day-out out-of-fold probabilities
oof = {"rf": np.zeros((len(y), 4)), "tree": np.zeros((len(y), 4))}
for tr, te in LeaveOneGroupOut().split(X, y, grp):
    for name, mk in [("rf", rf), ("tree", tree)]:
        m = mk().fit(X[tr], y[tr])
        p = np.zeros((len(te), 4))
        p[:, m.classes_] = m.predict_proba(X[te])
        oof[name][te] = p


def decide(prob, tau):
    """defect if P(defect)=1-P(normal) >= tau; type = argmax over defect types."""
    pdef = 1 - prob[:, 0]
    return np.where(pdef >= tau, prob[:, 1:].argmax(axis=1) + 1, 0)


taus = np.round(np.arange(0.2, 0.95, 0.05), 2)
tau_tbl = pd.DataFrame([{"tau": t, "macroF1": f1_score(y, decide(oof["rf"], t), average="macro"),
                         "n_pred_defect": int((decide(oof["rf"], t) > 0).sum())} for t in taus])
# prevalence matching: the inspected defect rate is known (39 welds), so only thresholds whose
# out-of-fold defect count is within +-20 % of it are allowed; among them take the best macro-F1
n_true = int((y > 0).sum())
ok = tau_tbl[(tau_tbl.n_pred_defect - n_true).abs() <= 0.2 * n_true]
TAU = float((ok if len(ok) else tau_tbl).sort_values("macroF1").iloc[-1]["tau"])
say("\nthreshold search (RF, out-of-fold):\n" + tau_tbl.round(3).to_string(index=False))
say("chosen tau =", TAU, "| true defects in training:", int((y > 0).sum()))

for name in ["rf", "tree"]:
    pred = decide(oof[name], TAU) if name == "rf" else oof[name].argmax(axis=1)
    say(f"\n== leave-one-day-out: {name} ==")
    say(classification_report(y, pred, labels=[0, 1, 2, 3], target_names=[TYPE_EN[i] for i in range(4)],
                              zero_division=0))
    say("confusion (rows=true, cols=pred):\n" + pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1, 2, 3]),
                                                           index=[TYPE_EN[i] for i in range(4)],
                                                           columns=[TYPE_EN[i] for i in range(4)]).to_string())

# final models on all labelled candidates
rf_final = rf().fit(X, y)
tree_final = tree().fit(X, y)
imp = pd.Series(rf_final.feature_importances_, index=FEATS).sort_values(ascending=False)
rules = export_text(tree_final, feature_names=FEATS, class_names=[TYPE_EN[c] for c in tree_final.classes_],
                    decimals=4, show_weights=False)
(S2 / "selection_rules.txt").write_text(
    "# Decision-tree selection rules (depth 4, trained on weak labels of stage-1 candidates)\n"
    "# F bar, I kA, V V, T ms, R mOhm (=V/I), Q J (=V*I*T), q_per_F = Q/F,\n"
    "# d_X = X - rolling median(11 welds) of the same day, step_X = X - previous weld\n\n"
    + rules + "\n\n# RandomForest feature importance\n" + imp.round(4).to_string(), encoding="utf-8")
say("\nRF feature importance:\n" + imp.round(3).to_string())
say("\ndecision-tree rules:\n" + rules)

# ------------------------------------------------------------------ 3. final decision on every candidate
P = np.zeros((len(cand), 4))
P[:, rf_final.classes_] = rf_final.predict_proba(cand[FEATS].values)
cand["pred_type"] = decide(P, TAU)
for t in range(4):
    cand[f"P_{TYPE_EN[t].split()[0]}"] = P[:, t].round(3)
cand["P_defect"] = (1 - P[:, 0]).round(3)
cand["tree_rule_type"] = tree_final.predict(cand[FEATS].values)

final = cand[cand.pred_type > 0].copy()
final["defect type"] = final.pred_type.astype(int)
final["defect name"] = final["defect type"].map(TYPE_NAME)
final["weak_label"] = final.y.map(lambda v: "" if pd.isna(v) else TYPE_NAME[int(v)])
final["inspected_day"] = np.where(final.date == UNLABELED_DAY, "N (예측만)", "Y")
out_cols = ["uid", "date", "idx", "seq", "F", "I", "V", "T", "R", "Q", "candidate_source", "segment_id",
            "anomaly_score", "defect type", "defect name", "P_defect", "P_normal", "P_T1", "P_T2", "P_T3",
            "tree_rule_type", "weak_label", "inspected_day"]
final = final[out_cols].rename(columns={"F": "weld force(bar)", "I": "weld current(kA)", "V": "weld Voltage(v)",
                                        "T": "weld time(ms)", "R": "R_dyn(mOhm)", "Q": "Q_heat(J)"})
final.to_csv(S2 / "final_defects.csv", index=False, encoding="utf-8-sig")

cmp = final.groupby(["date", "defect type"]).size().unstack(fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
cmp.columns = [f"pred_T{c}" for c in cmp.columns]
res = counts.reindex(columns=[1, 2, 3], fill_value=0)
res.columns = [f"result_T{c}" for c in res.columns]
cmp = cmp.join(res, how="outer").fillna(0).astype(int)
cmp["n_candidates"] = cand.groupby("date").size()
say("\n== final defects vs result sheet ==\n" + cmp.to_string())
say("final defects:", len(final), "| candidates dropped as normal:", int((cand.pred_type == 0).sum()))
say("agreement with weak label on labelled days: %.3f" %
    (cand.loc[cand.y.notna(), "pred_type"] == cand.loc[cand.y.notna(), "y"]).mean())

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (xv, yv) in zip(axs, [("F", "Q"), ("I", "Q")]):
    nd = cand[cand.pred_type == 0]
    ax.scatter(nd[xv], nd[yv], s=5, c=COLORS[0], label="candidate → normal")
    for t in (1, 2, 3):
        m = cand.pred_type == t
        ax.scatter(cand.loc[m, xv], cand.loc[m, yv], s=36, c=COLORS[t], edgecolor="k", lw=0.4, label=TYPE_EN[t])
    ax.set_xlabel(xv)
    ax.set_ylabel(yv)
    if xv == "F":
        ax.set_xscale("log")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Stage 2: final defect types among stage-1 candidates")
fig.tight_layout()
fig.savefig(S2 / "fig_s2_final_types.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
imp.iloc[::-1].plot.barh(ax=ax, color="#1f77b4")
ax.set_title("RandomForest feature importance (selection criteria)")
fig.tight_layout()
fig.savefig(S2 / "fig_s2_importance.png")
plt.close(fig)

(S2 / "stage2_log.txt").write_text("\n".join(log), encoding="utf-8")
