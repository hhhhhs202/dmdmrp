"""Shared loading / preprocessing / physical-feature code for stage 1 and stage 2."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# collection ranges from the 'data set' sheet (physical/spec limits of the sensors)
SPEC = {"F": (1.0, 12.0), "I": (12.0, 18.0), "V": (1.5, 3.5), "T": (30.0, 120.0)}
TYPE_NAME = {0: "정상", 1: "파임불량", 2: "용접부족", 3: "크랙발생"}
TYPE_EN = {0: "normal", 1: "T1 dent", 2: "T2 lack of fusion", 3: "T3 crack"}
COLORS = {0: "#c8c8c8", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
UNLABELED_DAY = 20200327            # no inspection record in the result sheet
ROLL_W = 11                           # window (welds) for local-baseline features

ORIG_COLS = {"F": "weld force(bar)", "I": "weld current(kA)", "V": "weld Voltage(v)", "T": "weld time(ms)"}


def mod_z(x, center=None, scale=None):
    """Iglewicz-Hoaglin modified z-score: 0.6745 (x - median) / MAD.
    Falls back to IQR/1.349 when MAD is 0 (discrete signals such as weld time)."""
    med = np.median(x) if center is None else center
    if scale is None:
        mad = np.median(np.abs(x - med))
        scale = mad / 0.6745 if mad > 0 else (np.subtract(*np.percentile(x, [75, 25])) / 1.349 or 1.0)
    return (x - med) / scale, med, scale


def load_raw():
    xl = pd.read_excel(ROOT / "data/Welding_Data_Set_01.xlsx", sheet_name=None)
    raw = xl["Raw data"].copy()
    raw.columns = ["idx", "Machine_Name", "Item No", "working time", "t1", "t2", "F", "I", "V", "T"]
    raw["date"] = raw["working time"].dt.strftime("%Y%m%d").astype(int)      # 20200324
    res = xl["result"].iloc[:, :6].copy()
    res.columns = ["idx", "Machine_Name", "Item No", "working time", "defect", "defect type"]
    res["date"] = res["working time"].dt.strftime("%Y%m%d").astype(int)
    counts = res.pivot_table(index="date", columns="defect type", values="defect", aggfunc="sum").fillna(0)
    counts = counts.astype(int)
    return raw, counts


def preprocess(raw):
    """1) spec-range check  2) duplicated (date, idx)  3) row order key  4) physical + local features."""
    df = raw.copy()
    for k, (lo, hi) in SPEC.items():
        df.loc[(df[k] < lo) | (df[k] > hi), k] = np.nan
    n_nan = int(df[list(SPEC)].isna().any(axis=1).sum())
    df = df.dropna(subset=list(SPEC))
    dup = df.duplicated(["date", "idx"], keep="first")
    df = df[~dup].copy()
    df["seq"] = df.groupby("date").cumcount()          # position within the day (time order)
    df["uid"] = df["date"].astype(str) + "_" + df["idx"].astype(str)

    # physical features (resistance spot welding)
    df["R"] = df["V"] / df["I"]                        # dynamic resistance  [V/kA = mOhm]
    df["Q"] = df["V"] * df["I"] * df["T"]              # Joule heat I^2 R t = V I t  [V*kA*ms = J]
    df["P"] = df["V"] * df["I"]                        # electrical power [kW]
    df["logF"] = np.log(df["F"])
    df["q_per_F"] = df["Q"] / df["F"]                  # heat per unit electrode force

    # local (temporal) features: deviation from the rolling median of the same day
    for k in ["F", "I", "V", "R", "Q"]:
        base = df.groupby("date")[k].transform(lambda s: s.rolling(ROLL_W, center=True, min_periods=3).median())
        df[f"d_{k}"] = df[k] - base
        df[f"step_{k}"] = df.groupby("date")[k].diff().fillna(0)   # weld-to-weld jump
    info = {"spec_violations": n_nan, "dup_date_idx": int(dup.sum())}
    return df.reset_index(drop=True), info
