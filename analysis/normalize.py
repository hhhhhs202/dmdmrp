"""Normalization (MinMax, 0~1) and standardization (Z-score, mean 0 / std 1) of both datasets.

Inputs
  data/Welding_Data_Set_01.xlsx   'Raw data' sheet: F, I, V, T in physical units
  data/scaled_data.csv            already MinMax-scaled, but with sensor spikes and NaN mixed in

scaled_data cannot be re-scaled as is: one spike stretches the MinMax denominator so 90~99.6 % of
normal rows sit below 0.01 (REPORT.md section 7). So it is first inverse-scaled to physical units,
out-of-spec values ('data set' sheet ranges) are set to NaN and filled from the previous weld, and only
then normalized/standardized. Scalers are fitted per dataset on all (cleaned) rows.

Outputs (output/normalized/)
  raw_normalized.csv / raw_standardized.csv
  scaled_data_normalized.csv / scaled_data_standardized.csv
  normalized_all.xlsx             same four tables as sheets + scaler parameters
  scaler_params.csv               min/max/mean/std used for every column (to inverse-transform)
  normalize_log.txt

Run:  python analysis/normalize.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import OUT, ROOT, SPEC, load_raw

DST = OUT / "normalized"
DST.mkdir(parents=True, exist_ok=True)
RAW4 = list(SPEC)                                         # F, I, V, T
KO = {"F": "용접 가압력(bar)", "I": "전류(kA)", "V": "전압(V)", "T": "통전시간(ms)"}
RES_UNIT = {"F": 0.01, "I": 0.01, "V": 0.001, "T": 1}     # sensor resolution of the raw sheet

log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


def scale(df, name):
    """Return (normalized, standardized, params) for the F/I/V/T columns of df."""
    x = df[RAW4]
    mm, ss = MinMaxScaler().fit(x), StandardScaler().fit(x)
    norm, std = df.copy(), df.copy()
    norm[RAW4] = mm.transform(x)
    std[RAW4] = ss.transform(x)
    params = pd.DataFrame({"dataset": name, "column": RAW4, "min": mm.data_min_, "max": mm.data_max_,
                           "mean": ss.mean_, "std": ss.scale_})
    say(f"\n[{name}] rows={len(df)}")
    say("  normalized   min/max :", norm[RAW4].agg(["min", "max"]).round(4).to_dict())
    say("  standardized mean/std:", std[RAW4].agg(["mean", "std"]).round(4).to_dict())
    return norm, std, params


# ---------------------------------------------------------------- 1. raw sheet
raw, _ = load_raw()
raw_df = raw[["date", "idx", *RAW4]].copy()
say("raw sheet NaN:", int(raw_df[RAW4].isna().sum().sum()),
    "| out-of-spec:", {k: int(((raw_df[k] < a) | (raw_df[k] > b)).sum()) for k, (a, b) in SPEC.items()})
raw_n, raw_s, raw_p = scale(raw_df, "raw")

# ---------------------------------------------------------------- 2. scaled_data
sd = pd.read_csv(ROOT / "data/scaled_data.csv", index_col=0)
sd.columns = RAW4
say("\nscaled_data NaN per col:", sd.isna().sum().to_dict())
say("scaled_data share < 0.01 (before cleaning):", (sd < 0.01).mean().round(3).to_dict())

# inverse MinMax: quantization step of the scaled column == one sensor resolution unit
recon = {}
for k in RAW4:
    u = np.sort(sd[k].dropna().unique())
    step = np.diff(u)[np.diff(u) > 1e-9].min()
    q = np.round(sd[k] / step)
    off = np.nanmedian(raw[k] - q * RES_UNIT[k])          # rows are aligned with the raw sheet
    recon[k] = off + q * RES_UNIT[k]
    say(f"  {k}: implied original MinMax range [{off:.3f}, {off + RES_UNIT[k] / step:.2f}]")
sd_phys = pd.DataFrame(recon).round({"F": 2, "I": 2, "V": 3, "T": 0})

oor = pd.concat([(sd_phys[k] < a) | (sd_phys[k] > b) for k, (a, b) in SPEC.items()], axis=1)
say("out-of-spec values -> NaN:", oor.sum().to_dict(), "| rows:", int(oor.any(axis=1).sum()))
sd_phys = sd_phys.mask(oor)
n_nan = int(sd_phys.isna().sum().sum())
sd_phys = sd_phys.ffill().bfill()                         # time series: carry the previous weld
say(f"filled {n_nan} NaN (original NaN + spikes) with previous value")
sd_phys.insert(0, "row", sd_phys.index)
sd_n, sd_s, sd_p = scale(sd_phys, "scaled_data")
say("scaled_data share < 0.01 (after cleaning + MinMax):", (sd_n[RAW4] < 0.01).mean().round(3).to_dict())

# ---------------------------------------------------------------- 3. save
params = pd.concat([raw_p, sd_p], ignore_index=True)
tables = {"raw_normalized": raw_n, "raw_standardized": raw_s,
          "scaled_data_normalized": sd_n, "scaled_data_standardized": sd_s}
ren = lambda d: d.rename(columns=KO)                      # noqa: E731  Korean headers for the deliverable
for name, d in tables.items():
    ren(d).round(6).to_csv(DST / f"{name}.csv", index=False, encoding="utf-8-sig")
params.to_csv(DST / "scaler_params.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(DST / "normalized_all.xlsx") as xw:
    for name, d in tables.items():
        ren(d).round(6).to_excel(xw, sheet_name=name, index=False)
    params.to_excel(xw, sheet_name="scaler_params", index=False)
say("\nscaler parameters:\n" + params.round(4).to_string(index=False))
(DST / "normalize_log.txt").write_text("\n".join(log), encoding="utf-8")
print("\nsaved to", DST)
