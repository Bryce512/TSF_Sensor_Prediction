# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
!pip install numpy pandas

# COMMAND ----------

# DBTITLE 1,Detrend, normalize, then autoregression
import os
import numpy as np
import pandas as pd

# Order of operations: subtract this truck's own straight-line trend, divide by
# this truck's own leftover spread, then ask whether lags beat predicting the mean.
DATA_DIR = os.environ.get("SCANIA_DIR", "/Volumes/workspace/bronze/raw_data/")
N_TRUCKS = 10
MAX_LAG = 8
MIN_POINTS = 40

def load_ops(name):
    print(f"Loading {name}...")
    df = pd.read_csv(DATA_DIR + name)
    for c in df.columns:
        if c == "vehicle_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c != "time_step":
            df[c] = df[c].astype(np.float32)
    return df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

def detrend_and_normalize(truck, sensor_cols):
    """Remove the fitted line versus time_step, then scale by the leftover spread."""
    values = truck[sensor_cols].to_numpy(np.float64)
    mask = np.isnan(values)
    idx = np.where(~mask, np.arange(len(truck))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    values = values[idx, np.arange(values.shape[1])[None, :]]
    time = truck["time_step"].to_numpy(np.float64)
    t = time - time.mean()
    denom = float(np.dot(t, t)) or 1.0
    # Centered values, so the slope is sum(t * value) / sum(t^2).
    centered = values - np.nanmean(values, axis=0)
    slope = (t @ np.nan_to_num(centered)) / denom
    line = np.nanmean(values, axis=0) + np.outer(time - time.mean(), slope)
    resid = values - line
    std = np.nanstd(resid, axis=0)
    z = np.full_like(resid, np.nan)
    usable = std > 1e-8
    z[:, usable] = resid[:, usable] / std[usable]
    return z

def fit_ar(z):
    """AIC chooses the lag count. Explained is the variance the lags add past the mean."""
    finite = np.isfinite(z)
    if finite.sum() < MIN_POINTS:
        return None
    series = z[finite]
    base = float(np.mean((series - series.mean()) ** 2))
    if base <= 1e-8:
        return None
    best = None
    for k in range(1, MAX_LAG + 1):
        y = series[k:]
        x = np.column_stack([series[k - lag : len(series) - lag] for lag in range(1, k + 1)])
        x = np.column_stack([np.ones(len(y)), x])
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        mse = float(np.mean(resid ** 2))
        aic = len(y) * np.log(max(mse, 1e-12)) + 2 * (k + 1)
        dof = max(len(y) - x.shape[1], 1)
        sigma2 = float(np.sum(resid ** 2) / dof)
        try:
            se = np.sqrt(np.maximum(np.diag(np.linalg.inv(x.T @ x)) * sigma2, 0))
        except np.linalg.LinAlgError:
            se = np.full(x.shape[1], np.nan)
        tstat = coef / np.where(se > 0, se, np.nan)
        record = (aic, k, mse, tstat)
        if best is None or aic < best[0]:
            best = record
    aic, k, mse, tstat = best
    explained = max(0.0, 1.0 - mse / base)
    lag_t = tstat[1:]
    strongest = int(np.nanargmax(np.abs(lag_t))) + 1 if np.isfinite(lag_t).any() else 0
    strong_lags = [i + 1 for i, t in enumerate(lag_t) if np.isfinite(t) and abs(t) >= 2]
    return {
        "lags": k,
        "explained": explained,
        "strongest": strongest,
        "beats_mean": explained >= 0.02 and len(strong_lags) > 0,
    }

def choose_trucks(last):
    """Ten class-4 trucks and ten healthy trucks with a similar number of visits."""
    rem = last["length_of_study_time_step"] - last["max"]
    failed = last[(last["in_study_repair"] == 1) & (rem <= 6) & (last["count"] >= 100)]
    failed = failed.sort_values("count", ascending=False).head(N_TRUCKS)
    lo, hi = float(failed["count"].min()), float(failed["count"].max())
    healthy_pool = last[(last["in_study_repair"] != 1) & (last["count"] >= lo) & (last["count"] <= hi)]
    rng = np.random.default_rng(42)
    if len(healthy_pool) < N_TRUCKS:
        healthy_pool = last[last["in_study_repair"] != 1].sort_values("count", ascending=False).head(200)
    chosen = rng.choice(healthy_pool["vehicle_id"].to_numpy(), size=N_TRUCKS, replace=False)
    healthy = last[last["vehicle_id"].isin(chosen)]
    failed = failed.copy()
    healthy = healthy.copy()
    failed["group"] = "failed"
    healthy["group"] = "healthy"
    return pd.concat([failed, healthy], ignore_index=True)

print("Choosing trucks...")
tte = pd.read_csv(DATA_DIR + "train_tte.csv")
ops_id = pd.read_csv(DATA_DIR + "train_operational_readouts.csv", usecols=["vehicle_id", "time_step"])
last = ops_id.groupby("vehicle_id", sort=False)["time_step"].agg(["max", "count"]).reset_index()
last = last.merge(tte, on="vehicle_id", how="left")
pick = choose_trucks(last)
print(pick.groupby("group")["count"].agg(["min", "median", "max"]).to_string())
keep = set(pick["vehicle_id"].tolist())
group_of = dict(zip(pick["vehicle_id"], pick["group"]))

ops = load_ops("train_operational_readouts.csv")
ops = ops[ops["vehicle_id"].isin(keep)]
sensor_cols = [c for c in ops.columns if c not in ("vehicle_id", "time_step")]

rows = []
for vid, truck in ops.groupby("vehicle_id", sort=False):
    z = detrend_and_normalize(truck, sensor_cols)
    print(f"{group_of[vid]} truck {int(vid)}: {len(truck)} visits")
    for j, sensor in enumerate(sensor_cols):
        fit = fit_ar(z[:, j])
        if fit is None:
            continue
        rows.append({"group": group_of[vid], "vehicle_id": int(vid), "sensor": sensor, **fit})

fits = pd.DataFrame(rows)

def report(part, title):
    wins = part[part["beats_mean"]]
    print()
    print(title)
    print(f"  series fit: {len(part)}")
    print(f"  lags beat the mean: {len(wins)} of {len(part)} ({len(wins) / max(len(part), 1):.0%})")
    if wins.empty:
        return
    print(f"  median variance beyond the mean: {wins['explained'].median():.0%}")
    print("  lag count kept:")
    print(wins["lags"].value_counts().sort_index().to_string().replace("\n", "\n  "))
    print("  strongest lag:")
    print(wins["strongest"].value_counts().sort_index().to_string().replace("\n", "\n  "))

report(fits[fits["group"] == "failed"], "Failed trucks, after detrend and normalize")
report(fits[fits["group"] == "healthy"], "Healthy trucks, after detrend and normalize")

# A sensor supports the theory when its lags beat the mean on failed trucks more often than on healthy ones.
rows = []
for sensor, part in fits.groupby("sensor"):
    failed = part[part["group"] == "failed"]
    healthy = part[part["group"] == "healthy"]
    rows.append({
        "sensor": sensor,
        "failed_wins": int(failed["beats_mean"].sum()),
        "healthy_wins": int(healthy["beats_mean"].sum()),
        "failed_explained": float(failed["explained"].median()),
        "healthy_explained": float(healthy["explained"].median()),
        "failed_lags": float(failed.loc[failed["beats_mean"], "lags"].median()) if failed["beats_mean"].any() else 0,
    })
summary = pd.DataFrame(rows)
summary["gap"] = summary["failed_wins"] - summary["healthy_wins"]
summary = summary.sort_values(["gap", "failed_explained"], ascending=False)
print()
print("Sensors where failed trucks use lags more often than healthy trucks")
print(f"{'sensor':<10} {'failed':>8} {'healthy':>8} {'failed var':>11} {'healthy var':>12} {'failed lags':>12}")
shown = summary.head(12)
for rec in shown.to_dict("records"):
    print(
        f"{rec['sensor']:<10} {rec['failed_wins']:>5}/{N_TRUCKS:<2} {rec['healthy_wins']:>5}/{N_TRUCKS:<2} "
        f"{rec['failed_explained']:>10.0%} {rec['healthy_explained']:>11.0%} {rec['failed_lags']:>12.0f}"
    )
print()
print(
    f"Median wins per sensor: failed {summary['failed_wins'].median():.0f}/{N_TRUCKS}, "
    f"healthy {summary['healthy_wins'].median():.0f}/{N_TRUCKS}"
)