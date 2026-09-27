"""
스폿용접 불량 선별 통합 파이프라인 (단일 파일)

  [1단계 · 비지도]  Raw data 시트(센서값)만 사용한다. 불량 기록(result 시트)은 절대 읽지 않는다.
      1-1 전처리 + 물리/시간 피처
      1-2 이상 공정 구간(defect 구간) 탐지
      1-3 안정 구간 이상탐지 3종 투표 (안전범위 규칙 · 강건 마할라노비스 · Isolation Forest)
      1-4 안전 공정 범위 산출
      1-5 불량 후보만 저장  → 여기서 결과를 "동결(freeze)"하고 지문(SHA-256)을 남긴다
  [2단계 · 지도]    동결된 후보 파일 + result 시트(날짜별·타입별 불량 개수)
      2-1 학습 라벨 생성 (물리 점수 + 헝가리안 할당)
      2-2 선별 기준 자동 학습 (RandomForest + DecisionTree, 날짜 단위 교차검증)
      2-3 판정 임계값 결정 → 후보 중 real defect만 저장
  [분리 검증]       (1) 1단계 산출물이 2단계 이후에도 바이트 단위로 동일한지
                    (2) result 시트를 삭제한 엑셀로 1단계를 다시 돌려도 산출물이 동일한지

실행:  python welding_defect_pipeline.py
필요:  pip install pandas numpy scipy scikit-learn openpyxl
출력:  output/pipeline/
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Welding_Data_Set_01.xlsx"
OUT = ROOT / "output" / "pipeline"

# ---------------------------------------------------------------- 1단계 하이퍼파라미터 (라벨을 보고 튜닝하지 않는 사전 고정값)
SPEC = {"F": (1.0, 12.0), "I": (12.0, 18.0), "V": (1.5, 3.5), "T": (30.0, 120.0)}   # data set 시트 수집 범위
Z_CUT = 3.5          # 수정 z-점수 컷 (Iglewicz-Hoaglin 권고값)
GAP, MIN_SEG = 5, 10 # 이상점 병합 간격 / 구간 최소 길이 (타점)
ROLL_W = 11          # 국소 기준선 창 (타점)
MD_Q = 0.999         # 마할라노비스 χ² 분위
IF_CONT = 0.02       # Isolation Forest 오염률
DAY_TOP = 0.02       # 날짜별 이상 점수 상위 비율 (현장 불량률 사전지식 상한)

# ---------------------------------------------------------------- 2단계 하이퍼파라미터
TYPE_NAME = {0: "정상", 1: "파임불량", 2: "용접부족", 3: "크랙발생"}
FEATS = ["F", "I", "V", "T", "R", "Q", "q_per_F", "d_F", "d_I", "d_V", "d_Q", "step_F", "step_I", "step_V"]
PREVALENCE_TOL = 0.2 # 임계값: OOF 예측 불량 수가 실제의 ±20% 이내인 것 중 macro-F1 최대

# 1단계 입력에 절대 있어서는 안 되는 열 (라벨 누수 방지 가드)
LABEL_COLS = {"defect", "defect type", "y", "label", "defect name"}

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mod_z_params(x):
    """수정 z-점수의 중심/척도: median, MAD/0.6745 (MAD=0이면 IQR/1.349)."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        return med, mad / 0.6745
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return med, (iqr / 1.349) if iqr > 0 else 1.0


# ==================================================================================================
# 1단계 · 비지도 학습
# ==================================================================================================
def load_sensor_data(path=DATA):
    """Raw data 시트만 읽는다. (result 시트는 이 함수에서 접근하지 않는다)"""
    raw = pd.read_excel(path, sheet_name="Raw data")
    raw.columns = ["idx", "Machine_Name", "Item No", "working time", "t1", "t2", "F", "I", "V", "T"]
    raw["date"] = raw["working time"].dt.strftime("%Y%m%d").astype(int)
    # 센서 분해능으로 반올림 (엑셀 저장 방식에 따른 1e-15 수준 부동소수 잔여 오차 제거)
    raw = raw.round({"F": 2, "I": 2, "V": 3, "T": 2})
    return raw


def stage1_preprocess(raw):
    df = raw.copy()
    for k, (lo, hi) in SPEC.items():                                  # 스펙 범위 밖 → 제거
        df.loc[(df[k] < lo) | (df[k] > hi), k] = np.nan
    n_spec = int(df[list(SPEC)].isna().any(axis=1).sum())
    df = df.dropna(subset=list(SPEC))
    dup = df.duplicated(["date", "idx"], keep="first")                # (날짜, idx) 중복 제거
    df = df[~dup].copy()
    df["seq"] = df.groupby("date").cumcount()
    df["uid"] = df["date"].astype(str) + "_" + df["idx"].astype(str)

    df["R"] = df["V"] / df["I"]                  # 동저항 [mΩ]
    df["Q"] = df["V"] * df["I"] * df["T"]        # 입열량 I²Rt = VIt [J]
    df["P"] = df["V"] * df["I"]                  # 전력 [kW]
    df["q_per_F"] = df["Q"] / df["F"]            # 가압력당 입열 [J/bar]
    for k in ["F", "I", "V", "R", "Q"]:          # 국소 기준선 편차 / 직전 대비 변화
        base = df.groupby("date")[k].transform(lambda s: s.rolling(ROLL_W, center=True, min_periods=3).median())
        df[f"d_{k}"] = df[k] - base
        df[f"step_{k}"] = df.groupby("date")[k].diff().fillna(0)
    say(f"[1-1] 전처리: 스펙 위반 {n_spec}행, (date,idx) 중복 {int(dup.sum())}행 제거 → {len(df)}행")
    return df.reset_index(drop=True)


def stage1_segments(df):
    """F 또는 V의 전역 수정 z가 3.5 초과인 점을 GAP 이내로 병합, MIN_SEG 이상이면 이상 공정 구간."""
    cF, sF = mod_z_params(df["F"])
    cV, sV = mod_z_params(df["V"])
    point = ((df["F"] - cF).abs() / sF > Z_CUT) | ((df["V"] - cV).abs() / sV > Z_CUT)
    df["segment_id"] = ""
    segs = []
    for d, g in df.groupby("date"):
        pos = g.index[point[g.index]]
        if len(pos) == 0:
            continue
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
                s = df.loc[a:b]
                segs.append({"segment_id": sid, "date": d, "idx_start": int(s.idx.iloc[0]),
                             "idx_end": int(s.idx.iloc[-1]), "n_welds": len(s), "F_max": round(s.F.max(), 2),
                             "F_median": round(s.F.median(), 2), "V_min": round(s.V.min(), 3),
                             "V_max": round(s.V.max(), 3), "Q_min": round(s.Q.min(), 1),
                             "Q_max": round(s.Q.max(), 1), "R_min": round(s.R.min(), 5)})
    df["in_segment"] = df["segment_id"] != ""
    segs = pd.DataFrame(segs)
    say(f"[1-2] 이상 공정 구간 {len(segs)}개, 구간 내 타점 {int(df.in_segment.sum())}개")
    return df, segs


def stage1_point_anomalies(df):
    """안정 구간에서만 학습하는 3개 탐지기 + 날짜별 상대 순위."""
    st = ~df["in_segment"]
    win_vars = ["F", "I", "V", "T", "R", "Q"]
    params = {}
    d1 = pd.Series(False, index=df.index)
    for k in win_vars:                                               # D1 안전범위 규칙
        c, s = mod_z_params(df.loc[st, k])
        params[k] = (c, s)
        df[f"mz_{k}"] = ((df[k] - c) / s).round(3)
        d1 |= df[f"mz_{k}"].abs() > Z_CUT
    df["D1_window"] = d1 & st

    base4 = ["F", "I", "V", "T"]                                     # D2 강건 마할라노비스
    sc = StandardScaler().fit(df.loc[st, base4])
    mcd = MinCovDet(support_fraction=0.9, random_state=0).fit(sc.transform(df.loc[st, base4]))
    df["mahal_d2"] = mcd.mahalanobis(sc.transform(df[base4]))
    md_cut = chi2.ppf(MD_Q, df=len(base4))
    df["D2_mahal"] = (df["mahal_d2"] > md_cut) & st

    if_feats = ["F", "I", "V", "T", "R", "Q", "d_F", "d_I", "d_V", "step_F", "step_I", "step_V"]
    sci = StandardScaler().fit(df.loc[st, if_feats])                  # D3 Isolation Forest
    iso = IsolationForest(n_estimators=400, contamination=IF_CONT, random_state=0).fit(sci.transform(df.loc[st, if_feats]))
    df["iforest_score"] = -iso.score_samples(sci.transform(df[if_feats]))
    df["D3_iforest"] = (iso.predict(sci.transform(df[if_feats])) == -1) & st
    df["votes"] = df[["D1_window", "D2_mahal", "D3_iforest"]].sum(axis=1)

    maxz = pd.concat([df[f"mz_{k}"].abs() for k in win_vars], axis=1).max(axis=1)
    score = (maxz[st].rank(pct=True) + df.loc[st, "mahal_d2"].rank(pct=True)
             + df.loc[st, "iforest_score"].rank(pct=True)) / 3
    df["anomaly_score"] = score.reindex(df.index).round(4)
    df["day_rank_pct"] = df[st].groupby("date")["anomaly_score"].rank(pct=True, ascending=False).reindex(df.index)

    df["point_candidate"] = st & (df["votes"] >= 2)
    df["day_candidate"] = st & ~df["point_candidate"] & (df["day_rank_pct"] <= DAY_TOP)
    df["candidate"] = df["in_segment"] | df["point_candidate"] | df["day_candidate"]
    df["candidate_source"] = np.select([df["in_segment"], df["point_candidate"], df["day_candidate"]],
                                       ["drift_segment", "point_anomaly", "day_relative"], "")
    say(f"[1-3] 탐지기 적중 D1={int(df.D1_window.sum())} D2={int(df.D2_mahal.sum())} D3={int(df.D3_iforest.sum())}"
        f" | 점 이상 {int(df.point_candidate.sum())} + 날짜별 상위 {int(df.day_candidate.sum())}"
        f" + 구간 {int(df.in_segment.sum())} = 후보 {int(df.candidate.sum())} ({df.candidate.mean():.1%})")
    return df, params, md_cut


def stage1_safe_window(df, params, md_cut):
    safe = df[~df["in_segment"] & ~df["candidate"]]
    rows = []
    for k, unit in [("F", "bar"), ("I", "kA"), ("V", "V"), ("T", "ms"), ("R", "mOhm"), ("Q", "J")]:
        c, s = params[k]
        rows.append({"variable": k, "unit": unit, "median": round(c, 5), "robust_sigma": round(s, 5),
                     "rule_low(med-3.5s)": round(c - Z_CUT * s, 5), "rule_high(med+3.5s)": round(c + Z_CUT * s, 5),
                     "observed_safe_min": round(safe[k].min(), 5), "observed_safe_max": round(safe[k].max(), 5),
                     "p0.5": round(safe[k].quantile(0.005), 5), "p99.5": round(safe[k].quantile(0.995), 5)})
    win = pd.DataFrame(rows)
    say(f"[1-4] 안전 공정 범위 계산 (정상 타점 {len(safe)}개), 결합 조건 마할라노비스 d² ≤ {md_cut:.2f}")
    return win


def run_stage1(data_path=DATA, out_dir=None):
    out_dir = out_dir or OUT
    raw = load_sensor_data(data_path)
    assert not (set(raw.columns) & LABEL_COLS), "1단계 입력에 라벨 열이 섞여 있음"
    df = stage1_preprocess(raw)
    df, segs = stage1_segments(df)
    df, params, md_cut = stage1_point_anomalies(df)
    win = stage1_safe_window(df, params, md_cut)
    assert not (set(df.columns) & LABEL_COLS), "1단계 산출물에 라벨 열이 생김"

    cols = ["uid", "date", "idx", "seq", "F", "I", "V", "T", "R", "Q", "P", "q_per_F",
            "d_F", "d_I", "d_V", "d_R", "d_Q", "step_F", "step_I", "step_V",
            "mz_F", "mz_I", "mz_V", "mz_T", "mz_R", "mz_Q", "mahal_d2", "iforest_score",
            "D1_window", "D2_mahal", "D3_iforest", "votes", "anomaly_score", "day_rank_pct",
            "candidate_source", "segment_id"]
    cand = df.loc[df.candidate, cols].copy()
    for c in ["D1_window", "D2_mahal", "D3_iforest"]:
        cand[c] = cand[c].astype(int)
    cand = cand.round({"F": 2, "I": 2, "V": 3, "T": 2, "R": 5, "Q": 2, "P": 3, "q_per_F": 2, "d_F": 3, "d_I": 3,
                       "d_V": 4, "d_R": 5, "d_Q": 2, "step_F": 3, "step_I": 3, "step_V": 4, "mahal_d2": 3,
                       "iforest_score": 4, "day_rank_pct": 4})
    paths = {"candidates": out_dir / "stage1_defect_candidates.csv",
             "segments": out_dir / "stage1_defect_segments.csv",
             "window": out_dir / "stage1_safe_process_window.csv"}
    cand.to_csv(paths["candidates"], index=False, encoding="utf-8-sig")
    segs.to_csv(paths["segments"], index=False, encoding="utf-8-sig")
    win.to_csv(paths["window"], index=False, encoding="utf-8-sig")
    # ---- 동결: 1단계 산출물의 지문을 남긴다. 2단계는 이 파일만 읽는다.
    fingerprint = {k: sha256_file(p) for k, p in paths.items()}
    say(f"[1-5] 후보 {len(cand)}행 저장 · 1단계 동결 (candidates sha256 {fingerprint['candidates'][:12]}…)")
    return paths, fingerprint


# ==================================================================================================
# 2단계 · 지도 학습
# ==================================================================================================
def load_labels():
    """result 시트(날짜별·타입별 불량 개수). 2단계에서만 호출된다."""
    res = pd.read_excel(DATA, sheet_name="result").iloc[:, :6]
    res.columns = ["idx", "Machine_Name", "Item No", "working time", "defect", "defect type"]
    res["date"] = res["working time"].dt.strftime("%Y%m%d").astype(int)
    return res.pivot_table(index="date", columns="defect type", values="defect", aggfunc="sum").fillna(0).astype(int)


def stage2_weak_labels(cand, counts):
    """검사일마다 후보 중 k_type개를 물리 점수로 골라 타입 부여 (헝가리안 할당). 나머지 후보 = 0."""
    score = pd.DataFrame({1: cand.mz_F + cand.mz_Q,                # 파임: 과가압 + 과입열
                          2: -cand.mz_Q - cand.mz_R,               # 부족: 입열 부족
                          3: cand.mz_Q + cand.mz_I - cand.mz_F})   # 크랙: 과입열 + 가압 부족
    cand["y"] = np.nan
    for d, g in cand.groupby("date"):
        if d not in counts.index:
            continue                                               # 검사 기록 없는 날 → 학습 제외
        slots = [t for t in (1, 2, 3) for _ in range(int(counts.loc[d].get(t, 0)))]
        cand.loc[g.index, "y"] = 0
        if not slots:
            continue
        pct = score.loc[g.index].rank(pct=True)
        margin = pct.sub(pct.mean(axis=1), axis=0)
        cost = -np.column_stack([pct[t].values + 0.5 * margin[t].values for t in slots])
        r_i, c_i = linear_sum_assignment(cost)
        cand.loc[g.index[r_i], "y"] = [slots[c] for c in c_i]
    lab = cand[cand.y.notna()].copy()
    lab["y"] = lab.y.astype(int)
    covered = cand.groupby("date").size().reindex(counts.index).fillna(0)
    short = list(counts.index[covered < counts.sum(axis=1)])
    say(f"[2-1] 학습 라벨 {lab.y.value_counts().sort_index().to_dict()} | 후보가 기록 불량 수보다 적은 날: {short or '없음'}")
    return cand, lab


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=-1)


def tree():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=0)


def decide(prob, tau):
    return np.where(1 - prob[:, 0] >= tau, prob[:, 1:].argmax(axis=1) + 1, 0)


def stage2_learn(lab):
    X, y, grp = lab[FEATS].values, lab.y.values, lab.date.values
    oof = np.zeros((len(y), 4))
    for tr, te in LeaveOneGroupOut().split(X, y, grp):             # 하루를 통째로 빼고 학습 → 그날 예측
        m = rf().fit(X[tr], y[tr])
        p = np.zeros((len(te), 4))
        p[:, m.classes_] = m.predict_proba(X[te])
        oof[te] = p
    taus = np.round(np.arange(0.2, 0.95, 0.05), 2)
    tbl = pd.DataFrame([{"tau": t, "macroF1": f1_score(y, decide(oof, t), average="macro"),
                         "n_pred": int((decide(oof, t) > 0).sum())} for t in taus])
    n_true = int((y > 0).sum())
    ok = tbl[(tbl.n_pred - n_true).abs() <= PREVALENCE_TOL * n_true]
    tau = float((ok if len(ok) else tbl).sort_values("macroF1").iloc[-1]["tau"])
    say(f"[2-2] 날짜 단위 교차검증 → 임계값 τ={tau} (OOF 예측 {int(tbl.set_index('tau').loc[tau, 'n_pred'])}개 / 실제 {n_true}개)")
    say(classification_report(y, decide(oof, tau), labels=[0, 1, 2, 3],
                              target_names=["normal", "T1 dent", "T2 lack", "T3 crack"], zero_division=0))
    rf_final, tree_final = rf().fit(X, y), tree().fit(X, y)
    rules = export_text(tree_final, feature_names=FEATS, class_names=[TYPE_NAME[c] for c in tree_final.classes_],
                        decimals=4)
    imp = pd.Series(rf_final.feature_importances_, index=FEATS).sort_values(ascending=False)
    (OUT / "stage2_selection_rules.txt").write_text(
        "# 학습된 선별 기준 (DecisionTree depth 4)\n\n" + rules + "\n\n# RandomForest 피처 중요도\n"
        + imp.round(4).to_string(), encoding="utf-8")
    return rf_final, tree_final, tau


def stage2_select(cand, counts, rf_final, tree_final, tau):
    P = np.zeros((len(cand), 4))
    P[:, rf_final.classes_] = rf_final.predict_proba(cand[FEATS].values)
    cand["pred_type"] = decide(P, tau)
    cand["P_defect"] = (1 - P[:, 0]).round(3)
    for t, n in enumerate(["P_normal", "P_T1", "P_T2", "P_T3"]):
        cand[n] = P[:, t].round(3)
    cand["tree_rule_type"] = tree_final.predict(cand[FEATS].values)
    cand["weak_label"] = cand.y.map(lambda v: "" if pd.isna(v) else TYPE_NAME[int(v)])
    cand["inspected_day"] = np.where(cand.date.isin(counts.index), "Y", "N (예측만)")

    # (A) 임계값 판정
    cand["final_type"] = cand["pred_type"]
    # (B) 검사 개수 맞춤: 검사일마다 타입별 기록 개수만큼, 해당 타입 확률 상위 후보를 1:1 할당
    tie = cand["anomaly_score"].fillna(1.0).values * 1e-3 - cand["seq"].values * 1e-9
    cand["matched_type"] = cand["pred_type"]
    for d, g in cand.groupby("date"):
        if d not in counts.index:
            continue
        cand.loc[g.index, "matched_type"] = 0
        slots = [t for t in (1, 2, 3) for _ in range(int(counts.loc[d].get(t, 0)))]
        if not slots:
            continue
        pos = cand.index.get_indexer(g.index)
        r_i, c_i = linear_sum_assignment(-np.column_stack([P[pos, t] + tie[pos] for t in slots]))
        cand.loc[g.index[r_i], "matched_type"] = [slots[c] for c in c_i]

    out_cols = ["uid", "date", "idx", "seq", "F", "I", "V", "T", "R", "Q", "candidate_source", "segment_id",
                "anomaly_score", "defect type", "defect name", "P_defect", "P_normal", "P_T1", "P_T2", "P_T3",
                "tree_rule_type", "weak_label", "inspected_day"]
    ren = {"F": "weld force(bar)", "I": "weld current(kA)", "V": "weld Voltage(v)", "T": "weld time(ms)",
           "R": "R_dyn(mOhm)", "Q": "Q_heat(J)"}
    for col, fname in [("final_type", "stage2_final_defects.csv"), ("matched_type", "stage2_final_defects_matched.csv")]:
        f = cand[cand[col] > 0].copy()                               # real defect만 (후보였지만 정상은 제외)
        f["defect type"] = f[col].astype(int)
        f["defect name"] = f["defect type"].map(TYPE_NAME)
        f[out_cols].rename(columns=ren).to_csv(OUT / fname, index=False, encoding="utf-8-sig")
        by_day = f.groupby(["date", "defect type"]).size().unstack(fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
        say(f"[2-3] {fname}: {len(f)}행 (검사일 {int((f.inspected_day == 'Y').sum())}개)\n{by_day.to_string()}")
    cand[["uid", "date", "idx", "F", "I", "V", "T", "R", "Q", "candidate_source", "y", "weak_label"]] \
        .dropna(subset=["y"]).to_csv(OUT / "stage2_training_labels.csv", index=False, encoding="utf-8-sig")


def run_stage2(stage1_paths):
    cand = pd.read_csv(stage1_paths["candidates"])                  # 동결된 1단계 산출물만 입력
    counts = load_labels()                                          # 라벨은 여기서 처음 읽는다
    say(f"[2-0] 입력: 1단계 후보 {len(cand)}행 + result 시트 (검사일 {len(counts)}일, 불량 {int(counts.values.sum())}개)")
    cand, lab = stage2_weak_labels(cand, counts)
    rf_final, tree_final, tau = stage2_learn(lab)
    stage2_select(cand, counts, rf_final, tree_final, tau)


# ==================================================================================================
def check_label_blind(fp_main):
    """라벨 유무만 다른 엑셀 사본 2개로 1단계를 각각 돌려 산출물을 비교한다.
      - control : 원본을 그대로 다시 저장 (result 시트 있음)
      - blind   : result 시트를 삭제하고 저장
    두 결과와 본 실행 결과가 모두 같다면 1단계는 라벨 정보에 전혀 의존하지 않는다."""
    import tempfile

    from openpyxl import load_workbook
    fps = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, drop in [("control", False), ("blind", True)]:
            wb = load_workbook(DATA)
            if drop:
                del wb["result"]
            path = tmp / f"{name}.xlsx"
            wb.save(path)
            (tmp / name).mkdir()
            _, fps[name] = run_stage1(path, tmp / name)
    return fps["control"] == fps["blind"] == fp_main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    say("=" * 30, "1단계 · 비지도 (라벨 미사용)", "=" * 30)
    paths, fp_before = run_stage1()
    say("=" * 30, "2단계 · 지도 (라벨 사용)", "=" * 30)
    run_stage2(paths)

    say("=" * 30, "분리 검증", "=" * 30)
    fp_after = {k: sha256_file(p) for k, p in paths.items()}
    same = fp_before == fp_after
    say(f"(검증 1) 1단계 산출물이 2단계 실행 전후로 동일: {same}")
    assert same, "2단계가 1단계 산출물을 변경함 (분리 위반)"
    blind = check_label_blind(fp_before)
    say(f"(검증 2) result 시트를 삭제한 사본으로 1단계를 다시 돌려도 산출물 동일: {blind}")
    assert blind, "1단계가 라벨 정보에 의존함 (분리 위반)"
    (OUT / "pipeline_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
