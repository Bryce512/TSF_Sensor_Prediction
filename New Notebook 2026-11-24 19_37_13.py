# Databricks notebook source
!pip install numpy pandas lightgbm scikit-learn

# COMMAND ----------

# DBTITLE 1,Last-20 summaries and a small class-2 list
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# One row per truck, so this uses every truck and still trains in minutes.
DATA_DIR = os.environ.get("SCANIA_DIR", "/Volumes/workspace/bronze/raw_data/")
K = 20
COST = np.array([
    [0, 500, 1000, 2000, 4000],
    [500, 0, 500, 1000, 2000],
    [1000, 500, 0, 500, 1000],
    [2000, 1000, 500, 0, 500],
    [4000, 2000, 1000, 500, 0],
], dtype=np.float64)

def total_cost(y, pred):
    return float(COST[np.asarray(y, int), np.asarray(pred, int)].sum())

def load_ops(name):
    print(f"Loading {name}...")
    df = pd.read_csv(DATA_DIR + name)
    for c in df.columns:
        if c == "vehicle_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # time_step decides the class boundary. float32 moves a few trucks across it.
        if c != "time_step":
            df[c] = df[c].astype(np.float32)
    return df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

def class_from_rem(rem, failed):
    """Same windows as the validation file: 0 healthy, 4 means 6 steps or fewer."""
    y = np.zeros(len(rem), dtype=np.int8)
    m = failed & np.isfinite(rem)
    y[m & (rem <= 48) & (rem > 24)] = 1
    y[m & (rem <= 24) & (rem > 12)] = 2
    y[m & (rem <= 12) & (rem > 6)] = 3
    y[m & (rem <= 6)] = 4
    return y

def summarize(ops):
    """Turn each truck's last 20 visits into a few numbers per sensor.

    last: the newest reading
    mean / std: the typical value and how much it jumped around
    slope: the tilt across those visits (positive means the sensor is climbing)
    slope5: the tilt of only the last 5 visits
    delta: newest reading minus the oldest of the 20
    """
    sensor_cols = [c for c in ops.columns if c not in ("vehicle_id", "time_step")]
    vid = ops["vehicle_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(vid[1:] != vid[:-1]) + 1]
    ends = np.r_[starts[1:], len(vid)]
    values = ops[sensor_cols].to_numpy(np.float32, copy=True)
    for a, b in zip(starts, ends):
        seg = values[a:b]
        valid = ~np.isnan(seg)
        idx = np.where(valid, np.arange(b - a)[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        filled = seg[idx, np.arange(seg.shape[1])[None, :]]
        filled[~np.isfinite(filled)] = np.nan
        values[a:b] = filled

    n, f = len(starts), len(sensor_cols)
    blocks = {
        name: np.full((n, f), np.nan, np.float32)
        for name in ("last", "mean", "std", "slope", "slope5", "delta")
    }
    for i, (a, b) in enumerate(zip(starts, ends)):
        seg = values[max(a, b - K):b]
        blocks["last"][i] = seg[-1]
        with np.errstate(all="ignore"):
            blocks["mean"][i] = np.nanmean(seg, axis=0)
            blocks["std"][i] = np.nanstd(seg, axis=0)
        blocks["delta"][i] = seg[-1] - seg[0]
        t = np.arange(len(seg), dtype=np.float32)
        t -= t.mean()
        denom = float(np.dot(t, t)) or 1.0
        blocks["slope"][i] = (t @ np.nan_to_num(seg)) / denom
        seg5 = seg[-5:]
        t5 = np.arange(len(seg5), dtype=np.float32)
        t5 -= t5.mean()
        denom5 = float(np.dot(t5, t5)) or 1.0
        blocks["slope5"][i] = (t5 @ np.nan_to_num(seg5)) / denom5
        if i and i % 5000 == 0:
            print(f"  summarized {i:,} trucks")

    frame = pd.DataFrame({"vehicle_id": vid[ends - 1].astype(np.int64)})
    for name, block in blocks.items():
        block[~np.isfinite(block)] = np.nan
        extra = pd.DataFrame(block, columns=[f"{c}_{name}" for c in sensor_cols])
        frame = pd.concat([frame, extra], axis=1)
    print(f"  {frame.shape[0]:,} trucks, {frame.shape[1] - 1} sensor summaries")
    return frame

def choose_fraction(prob, y, target_counts):
    """Pick how many trucks to flag, and whether the flag is class 1 or class 2.

    Held-out training trucks are about 10% failures. The validation snapshot is
    about 97% healthy. Each draw keeps every healthy held-out truck and a
    failure sample at the validation rate, then we keep the flag rate whose
    worst draw is cheapest.
    """
    y = np.asarray(y, int)
    rng = np.random.default_rng(0)
    pool = {c: np.flatnonzero(y == c) for c in range(5)}
    n0 = len(pool[0])
    keep_n = {0: n0}
    for c in range(1, 5):
        keep_n[c] = min(len(pool[c]), max(1, int(round(n0 * target_counts[c] / target_counts[0]))))
    fracs = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015]
    best = None
    for frac in fracs:
        for cls in (1, 2):
            costs = []
            for _draw in range(12):
                chosen = [pool[0]]
                for c in range(1, 5):
                    chosen.append(rng.choice(pool[c], size=keep_n[c], replace=False))
                idx = np.concatenate(chosen)
                k = max(1, int(round(frac * len(idx))))
                pred = np.zeros(len(idx), int)
                pred[np.argsort(prob[idx])[::-1][:k]] = cls
                costs.append(total_cost(y[idx], pred) * (target_counts[0] / n0))
            worst, avg = float(np.max(costs)), float(np.mean(costs))
            key = (worst, avg, frac)
            if best is None or key < best[0]:
                best = (key, frac, cls, avg, worst)
    return best

print("Building one row per truck from the last 20 visits...")
train_ops = load_ops("train_operational_readouts.csv")
Xtr = summarize(train_ops)
last_step = train_ops.groupby("vehicle_id", sort=False)["time_step"].max()
del train_ops

tte = pd.read_csv(DATA_DIR + "train_tte.csv")
lab = Xtr[["vehicle_id"]].merge(tte, on="vehicle_id", how="left")
lab["time_step"] = lab["vehicle_id"].map(last_step)
y = class_from_rem(
    (lab["length_of_study_time_step"] - lab["time_step"]).to_numpy(),
    lab["in_study_repair"].to_numpy() == 1,
)
Xtr = Xtr.merge(pd.read_csv(DATA_DIR + "train_specifications.csv"), on="vehicle_id", how="left")
print("Train class counts", np.bincount(y, minlength=5).tolist())

val_ops = load_ops("validation_operational_readouts.csv")
Xva = summarize(val_ops)
del val_ops
Xva = Xva.merge(pd.read_csv(DATA_DIR + "validation_specifications.csv"), on="vehicle_id", how="left")
y_val = (
    Xva[["vehicle_id"]]
    .merge(pd.read_csv(DATA_DIR + "validation_labels.csv"), on="vehicle_id")["class_label"]
    .to_numpy()
    .astype(int)
)
target_counts = np.bincount(y_val, minlength=5).astype(np.float64)
print("Validation class counts", target_counts.astype(int).tolist())

spec_cols = [c for c in Xtr.columns if str(c).startswith("Spec_")]
feats = [c for c in Xtr.columns if c != "vehicle_id"]
for c in spec_cols:
    cats = pd.Index(Xtr[c].astype(str).fillna("NA").unique())
    Xtr[c] = pd.Categorical(Xtr[c].astype(str).fillna("NA"), categories=cats)
    Xva[c] = pd.Categorical(Xva[c].astype(str).fillna("NA"), categories=cats)
X = Xtr[feats]
Xv = Xva[feats]

# The tree answers "is this truck inside a failure window?" One row is one truck.
oof = np.zeros(len(y))
val_prob = np.zeros(len(y_val))
yy = (y > 0).astype(np.int8)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, te) in enumerate(skf.split(X, yy), 1):
    print(f"Fold {fold}")
    pos = int(yy[tr].sum())
    neg = len(tr) - pos
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        colsample_bytree=0.5,
        scale_pos_weight=neg / max(pos, 1),
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X.iloc[tr], yy[tr])
    oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    val_prob += model.predict_proba(Xv)[:, 1] / 5

print(
    "Held-out score, class 4 median/90th "
    f"{np.median(oof[y == 4]):.3f}/{np.quantile(oof[y == 4], 0.9):.3f}  "
    "healthy "
    f"{np.median(oof[y == 0]):.3f}/{np.quantile(oof[y == 0], 0.9):.3f}"
)

(_key, frac, cls, avg, worst) = choose_fraction(oof, y, target_counts)
k = max(1, int(round(frac * len(val_prob))))
pred = np.zeros(len(val_prob), dtype=int)
pred[np.argsort(val_prob)[::-1][:k]] = cls
flagged = y_val[pred > 0]

always = total_cost(y_val, np.zeros(len(y_val), dtype=int))
cost = total_cost(y_val, pred)
print()
print("=== Validation Total Cost (Lower is Better) ===")
print(f"Always healthy cost : {always:,.0f}")
print(f"Flag top {frac:.2%} as class {cls} ({k} trucks)")
print(f"Held-out worst draw : {worst:,.0f}   average draw {avg:,.0f}")
print(f"Model cost          : {cost:,.0f}")
print(f"Prediction counts   : {np.bincount(pred, minlength=5).tolist()}")
print(f"True classes in the flagged trucks : {np.bincount(flagged, minlength=5).tolist()}")
print(f"Savings versus doing nothing : {always - cost:,.0f}")