import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import io
from math import sqrt

# for HTML/JS injection (kept; now used for Priority box too, though priority text is short)
import streamlit.components.v1 as components

# =========================================================
# IMPORTANT COMPAT PATCH
# =========================================================
def sparse_to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


# =========================================================
# CRITICAL PATCH (fix OneHotEncoder isnan crash on object categories)
# =========================================================
from sklearn.utils import _encode as _sk_encode

_original_check_unknown = _sk_encode._check_unknown

def _check_unknown_safe(Xi, known_values, return_mask=False):
    try:
        return _original_check_unknown(Xi, known_values, return_mask=return_mask)
    except TypeError:
        known = np.array(known_values, dtype=object)
        known_no_nan = known[~pd.isnull(known)]
        Xi_arr = np.array(Xi, dtype=object)
        Xi_no_nan = Xi_arr[~pd.isnull(Xi_arr)]
        diff = np.setdiff1d(np.unique(Xi_no_nan), known_no_nan)
        if return_mask:
            valid_mask = np.isin(Xi_arr, known_no_nan) | pd.isnull(Xi_arr)
            return diff, valid_mask
        return diff

_sk_encode._check_unknown = _check_unknown_safe


# =========================================================
# Config
# =========================================================
st.set_page_config(
    page_title="Roche Lab — Delay Risk Predictor",
    page_icon="🧪",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

MODEL_PATH     = PROJECT_ROOT / "models" / "lab_delay_model_v3.pkl"
WORKFLOW_PATH  = PROJECT_ROOT / "data" / "enriched" / "workflow_logs_priority_queue.csv"
TELEMETRY_PATH = PROJECT_ROOT / "data" / "raw" / "telemetry_logs.csv"
REAGENT_PATH   = PROJECT_ROOT / "data" / "raw" / "reagent_logs.csv"
THR_PATH       = PROJECT_ROOT / "models" / "thresholds.csv"

TARGET_DELAY_MIN_THRESHOLD_DEFAULT = 30  # minutes

# Display cap to avoid absurd-looking outputs
QUEUE_SLA_CAP_MIN = 24 * 60  # 24h cap for display
QUEUE_BASELINE_MIN = 5.0     # small setup/hand-off baseline


# =========================================================
# CSS
# NOTE: "Instrument" metric card has been replaced by "Priority"
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.app-header {
  font-size: 42px; font-weight: 900; margin: 0 0 4px 0;
  background: linear-gradient(90deg, #0b3b8c 0%, #1a73e8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-subtitle { font-size: 16px; color: #6b7280; margin: 0 0 24px 0; }

.metric-card {
  background: #ffffff; border-radius: 16px; padding: 18px 18px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;
  text-align: center;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.metric-label {
  font-size: 12px; font-weight: 800; color: #6b7280;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-value {
  font-size: 28px; font-weight: 900; margin: 6px 0 0 0;
  color: #1f2937;
  line-height: 1.1;
}
.metric-value-text {
  font-size: 20px !important;
  line-height: 1.15 !important;
  word-break: keep-all;
}

/* shrink-to-fit box (kept; now used for Priority box too) */
.metric-value-fit {
  width: 100%;
  height: 44px;
  padding: 0 6px;
  box-sizing: border-box;

  display: flex;
  align-items: center;
  justify-content: center;

  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  line-height: 1.05;
  font-weight: 900;
  color: #1f2937;
  font-size: 28px;
}

.pred-card {
  background: #ffffff; border-radius: 18px; padding: 28px 26px;
  border-left: 8px solid #f5a623;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.pred-action { font-size: 16px; color: #374151; margin-top: 10px; line-height: 1.5; }

.insight-box {
  background: linear-gradient(135deg, #eef4ff 0%, #f0f7ff 100%);
  border-radius: 16px; padding: 20px;
  border: 1px solid #c7d9f5;
}
.insight-title { font-size: 22px; font-weight: 900; margin: 0 0 12px 0; color: #0b3b8c; }
.insight-item-title { font-size: 16px; font-weight: 800; color: #0b3b8c; margin: 12px 0 4px 0; }
.insight-bullet { font-size: 14px; color: #1e4fa0; margin: 0 0 0 10px; line-height: 1.5; }

.rec-card {
  background: #ffffff; border-radius: 14px; padding: 18px 20px;
  border: 1px solid #e5e7eb; margin-bottom: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.rec-title { font-size: 17px; font-weight: 800; color: #1f2937; margin: 0 0 6px 0; }
.rec-rationale { font-size: 14px; color: #4b5563; margin: 0 0 4px 0; }
.rec-impact { font-size: 14px; font-weight: 700; margin: 0; }

.small-muted { color: #6b7280; font-size: 13px; }

div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-size: 16px; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# JS auto-fit: shrink any .metric-value-fit until it fits
components.html(
    """
<script>
(function() {
  function fitAll() {
    const doc = window.parent.document;
    const els = doc.querySelectorAll('.metric-value-fit');

    els.forEach(el => {
      el.style.fontSize = '28px';
      void el.offsetWidth;

      const maxW = el.clientWidth;
      const maxH = el.clientHeight;

      let size = 28;
      while (size > 12 && (el.scrollWidth > maxW || el.scrollHeight > maxH)) {
        size -= 1;
        el.style.fontSize = size + 'px';
        void el.offsetWidth;
      }
    });
  }

  window.addEventListener('load', fitAll);
  window.addEventListener('resize', fitAll);

  let t = 0;
  const timer = setInterval(() => {
    fitAll();
    t += 1;
    if (t > 12) clearInterval(timer);
  }, 250);
})();
</script>
""",
    height=0,
)

# =========================================================
# Utilities
# =========================================================
def safe_slider(label: str, lo: float, hi: float, mid: float, step: float):
    lo = float(lo) if np.isfinite(lo) else 0.0
    hi = float(hi) if np.isfinite(hi) else 1.0
    mid = float(mid) if np.isfinite(mid) else (lo + hi) / 2.0
    if lo >= hi:
        lo, hi = mid - 1.0, mid + 1.0
    span = hi - lo
    if not np.isfinite(step) or step <= 0:
        step = max(0.01, span / 100.0)
    mid = float(np.clip(mid, lo, hi))
    return st.slider(label, lo, hi, mid, step)

def fmt_min_to_h(mins: float) -> str:
    mins = float(mins)
    if mins < 60:
        return f"{mins:.0f} min"
    return f"{mins/60.0:.1f} h"


# =========================================================
# Load model + data
# =========================================================
for path, label in [
    (MODEL_PATH, "model_v3"),
    (WORKFLOW_PATH, "workflow_logs_priority_queue"),
    (TELEMETRY_PATH, "telemetry_logs"),
    (REAGENT_PATH, "reagent_logs"),
]:
    if not path.exists():
        st.error(f"Missing {label}: {path}")
        st.stop()

model = joblib.load(MODEL_PATH)
df_work = pd.read_csv(WORKFLOW_PATH)
df_tel = pd.read_csv(TELEMETRY_PATH)
df_rea = pd.read_csv(REAGENT_PATH)


# =========================================================
# Threshold
# =========================================================
delay_min_threshold = TARGET_DELAY_MIN_THRESHOLD_DEFAULT
if THR_PATH.exists():
    try:
        tdf = pd.read_csv(THR_PATH)
        if "delay_min_threshold" in tdf.columns:
            delay_min_threshold = float(tdf.iloc[0]["delay_min_threshold"])
    except Exception:
        pass


# =========================================================
# Patch imputers (v3 compatibility)
# =========================================================
def patch_all_simple_imputers(est, _seen=None):
    if _seen is None:
        _seen = set()
    oid = id(est)
    if oid in _seen:
        return est
    _seen.add(oid)

    if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
        stats = getattr(est, "statistics_", None)
        strat = getattr(est, "strategy", "")
        if isinstance(stats, np.ndarray):
            est._fill_dtype = np.dtype("object") if stats.dtype == object else stats.dtype
        else:
            est._fill_dtype = np.dtype("object") if strat in ("most_frequent", "constant") else np.dtype("float64")

    if isinstance(est, Pipeline):
        for _, step in est.steps:
            patch_all_simple_imputers(step, _seen)

    if isinstance(est, ColumnTransformer):
        trs = getattr(est, "transformers_", None) or getattr(est, "transformers", None) or []
        for _, tr, _ in trs:
            patch_all_simple_imputers(tr, _seen)

    if hasattr(est, "estimator"):
        try:
            patch_all_simple_imputers(est.estimator, _seen)
        except Exception:
            pass

    if hasattr(est, "calibrated_classifiers_"):
        try:
            for cc in est.calibrated_classifiers_:
                if hasattr(cc, "estimator"):
                    patch_all_simple_imputers(cc.estimator, _seen)
        except Exception:
            pass

    return est

model = patch_all_simple_imputers(model)


# =========================================================
# Telemetry aggregation (mean/max/std)
# =========================================================
df_tel["timestamp"] = pd.to_datetime(df_tel.get("timestamp", pd.NaT), errors="coerce")
df_tel["ambient_temp"] = pd.to_numeric(df_tel.get("ambient_temp", np.nan), errors="coerce")

tel_agg = df_tel.groupby("experiment_id").agg(
    ambient_temp=("ambient_temp", "mean"),
    ambient_temp_max=("ambient_temp", "max"),
    ambient_temp_std=("ambient_temp", "std"),
    telemetry_records=("timestamp", "count"),
    tel_time_span_sec=("timestamp", lambda s: (s.max() - s.min()).total_seconds() if s.notna().any() else 0.0),
).reset_index()
tel_agg["ambient_temp_std"] = pd.to_numeric(tel_agg["ambient_temp_std"], errors="coerce").fillna(0.0)

df = df_work.merge(tel_agg, on="experiment_id", how="left")
for c in ["ambient_temp", "ambient_temp_max", "ambient_temp_std", "telemetry_records", "tel_time_span_sec"]:
    df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)

# Reagent merge
df = df.merge(
    df_rea[["experiment_id", "reagent_batch_id"]].drop_duplicates("experiment_id"),
    on="experiment_id", how="left"
)
df["reagent_batch_id"] = df["reagent_batch_id"].fillna("UNKNOWN")


# =========================================================
# Feature engineering
# =========================================================
df["booking_time"] = pd.to_datetime(df.get("booking_time", pd.NaT), errors="coerce")
df["hour_of_day"] = df["booking_time"].dt.hour.fillna(9).astype(int)
df["day_of_week"] = df["booking_time"].dt.dayofweek.fillna(0).astype(int)
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

bt0 = df["booking_time"].min()
df["days_since_start"] = 0.0 if pd.isna(bt0) else ((df["booking_time"] - bt0).dt.total_seconds() / 86400.0).fillna(0.0)

df["scientist_workload"] = pd.to_numeric(df.get("scientist_workload", 0.0), errors="coerce").fillna(0.0)
df["lab_occupancy_level"] = pd.to_numeric(df.get("lab_occupancy_level", 0.0), errors="coerce").fillna(0.0)
df["stress_index"] = df["scientist_workload"] * df["lab_occupancy_level"]

# ensure priority/queue exist
if "priority" not in df.columns:
    df["priority"] = "Normal"
df["priority"] = df["priority"].astype(str).fillna("Normal")

if "queue_length" not in df.columns:
    df["queue_length"] = 0
df["queue_length"] = pd.to_numeric(df["queue_length"], errors="coerce").fillna(0).astype(int)

if "queue_wait_min" not in df.columns:
    df["queue_wait_min"] = 0.0
df["queue_wait_min"] = pd.to_numeric(df["queue_wait_min"], errors="coerce").fillna(0.0)

# instrument_id still needed for model + queue wait calc
if "instrument_id" not in df.columns:
    df["instrument_id"] = "UNKNOWN"
df["instrument_id"] = df["instrument_id"].astype(str).fillna("UNKNOWN")

# Duration-to-queue ratio
df["expected_duration"] = pd.to_numeric(df.get("expected_duration", 0), errors="coerce").fillna(0.0)
df["duration_to_queue_ratio"] = df["expected_duration"] / (df["queue_length"] + 1)

# Instrument cumulative hours (degradation proxy)
df = df.sort_values("booking_time").reset_index(drop=True)
df["instrument_cumulative_hours"] = df.groupby("instrument_id")["expected_duration"].cumsum() / 60.0

# Instrument recent failure rate (rolling 30-day window)
df["machine_failure"] = pd.to_numeric(df.get("machine_failure", 0), errors="coerce").fillna(0).astype(int)

def _rolling_failure_rate(group, window_days=30):
    group = group.sort_values("booking_time")
    times = group["booking_time"].values
    fails = group["machine_failure"].values
    n = len(group)
    rates = np.zeros(n, dtype=float)
    window_ns = np.timedelta64(window_days, 'D').astype('int64')
    left = 0; cum_fail = 0
    for i in range(n):
        cum_fail += fails[i]
        curr_ns = times[i].astype('int64') if not pd.isna(times[i]) else 0
        while left < i and (curr_ns - (times[left].astype('int64') if not pd.isna(times[left]) else 0)) > window_ns:
            cum_fail -= fails[left]; left += 1
        count = i - left + 1
        rates[i] = cum_fail / count if count > 0 else 0.0
    return pd.Series(rates, index=group.index)

df["instrument_recent_failure_rate"] = (
    df.groupby("instrument_id", group_keys=False)
    .apply(_rolling_failure_rate)
)


# =========================================================
# Priority-aware queue (realistic, instrument-based, P50/P90)
# =========================================================
PRIORITIES = ["Normal", "High", "Critical"]
PRIORITY_RANK = {"Normal": 0, "High": 1, "Critical": 2}
Z_P90 = 1.281551565545  # 90th percentile of standard normal

def _safe_priority(p: str) -> str:
    p = str(p) if p is not None else "Normal"
    return p if p in PRIORITY_RANK else "Normal"

@st.cache_data(show_spinner=False)
def build_instrument_service_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    tmp = df_all.copy()
    tmp["instrument_id"] = tmp.get("instrument_id", "UNKNOWN").astype(str)
    tmp["expected_duration"] = pd.to_numeric(tmp.get("expected_duration", np.nan), errors="coerce")

    g = tmp.groupby("instrument_id")["expected_duration"]
    stats = g.agg(["mean", "std"]).reset_index().rename(columns={"mean": "dur_mean", "std": "dur_std"})

    q = tmp.groupby("instrument_id")["expected_duration"].quantile([0.5, 0.9]).unstack().reset_index()
    q = q.rename(columns={0.5: "dur_p50", 0.9: "dur_p90"})

    out = stats.merge(q, on="instrument_id", how="left")

    out["dur_mean"] = out["dur_mean"].fillna(60.0)
    out["dur_std"] = out["dur_std"].fillna(15.0).replace(0.0, 10.0)
    out["dur_p50"] = out["dur_p50"].fillna(out["dur_mean"])
    out["dur_p90"] = out["dur_p90"].fillna(out["dur_mean"] + Z_P90 * out["dur_std"])
    return out

@st.cache_data(show_spinner=False)
def build_priority_mix_by_instrument(df_all: pd.DataFrame) -> pd.DataFrame:
    tmp = df_all.copy()
    tmp["instrument_id"] = tmp.get("instrument_id", "UNKNOWN").astype(str)
    tmp["priority"] = tmp.get("priority", "Normal").astype(str).map(_safe_priority)

    ct = tmp.groupby(["instrument_id", "priority"]).size().rename("n").reset_index()
    tot = ct.groupby("instrument_id")["n"].sum().rename("tot").reset_index()
    ct = ct.merge(tot, on="instrument_id", how="left")
    ct["p"] = ct["n"] / ct["tot"].clip(lower=1)

    piv = ct.pivot_table(index="instrument_id", columns="priority", values="p", fill_value=0.0).reset_index()
    for col in PRIORITIES:
        if col not in piv.columns:
            piv[col] = 0.0
    piv = piv.rename(columns={"Normal": "p_normal", "High": "p_high", "Critical": "p_critical"})

    global_mix = tmp["priority"].value_counts(normalize=True)
    g_norm = float(global_mix.get("Normal", 0.7))
    g_high = float(global_mix.get("High", 0.2))
    g_crit = float(global_mix.get("Critical", 0.1))

    piv["p_normal"] = piv["p_normal"].replace(0.0, np.nan).fillna(g_norm)
    piv["p_high"] = piv["p_high"].replace(0.0, np.nan).fillna(g_high)
    piv["p_critical"] = piv["p_critical"].replace(0.0, np.nan).fillna(g_crit)

    s = (piv["p_normal"] + piv["p_high"] + piv["p_critical"]).replace(0.0, 1.0)
    piv["p_normal"] = piv["p_normal"] / s
    piv["p_high"] = piv["p_high"] / s
    piv["p_critical"] = piv["p_critical"] / s
    return piv

def congestion_multiplier_from_stress(stress_index: float, df_all: pd.DataFrame) -> float:
    s = pd.to_numeric(df_all.get("stress_index", 0.0), errors="coerce").fillna(0.0)
    if len(s) < 100:
        return 1.0
    p50 = float(s.quantile(0.50))
    p90 = float(s.quantile(0.90))
    p10 = float(s.quantile(0.10))
    denom = max(1e-6, (p90 - p10))
    z = (float(stress_index) - p50) / denom
    mult = 1.0 + 0.9 * np.tanh(1.5 * z) * 0.25
    return float(np.clip(mult, 0.85, 1.45))

def compute_queue_wait_realistic(
    instrument_id: str,
    user_priority: str,
    queue_length: int,
    stress_index: float,
    instr_stats: pd.DataFrame,
    pr_mix: pd.DataFrame,
    df_all: pd.DataFrame,
    sla_cap_min: float = QUEUE_SLA_CAP_MIN,
    baseline_min: float = QUEUE_BASELINE_MIN,
):
    instrument_id = str(instrument_id) if instrument_id is not None else "UNKNOWN"
    user_priority = _safe_priority(user_priority)
    q = int(max(0, queue_length))

    row_s = instr_stats.loc[instr_stats["instrument_id"] == instrument_id]
    if len(row_s) == 0:
        dur_mean, dur_std = 60.0, 15.0
    else:
        dur_mean = float(row_s.iloc[0]["dur_mean"])
        dur_std = float(row_s.iloc[0]["dur_std"])

    row_p = pr_mix.loc[pr_mix["instrument_id"] == instrument_id]
    if len(row_p) == 0:
        mix = {"Normal": 0.7, "High": 0.2, "Critical": 0.1}
    else:
        mix = {
            "Normal": float(row_p.iloc[0]["p_normal"]),
            "High": float(row_p.iloc[0]["p_high"]),
            "Critical": float(row_p.iloc[0]["p_critical"]),
        }

    exp_counts = {k: q * mix[k] for k in PRIORITIES}
    ur = PRIORITY_RANK[user_priority]
    higher = sum(exp_counts[p] for p in PRIORITIES if PRIORITY_RANK[p] > ur)
    equal = exp_counts[user_priority]
    effective_jobs_ahead = higher + 0.5 * equal

    cong = congestion_multiplier_from_stress(stress_index, df_all)

    mean_wait = baseline_min + cong * effective_jobs_ahead * dur_mean
    eff_n = max(1e-6, effective_jobs_ahead)
    std_wait = cong * sqrt(eff_n) * dur_std

    p50 = mean_wait
    p90 = mean_wait + Z_P90 * std_wait

    p50_cap = float(np.clip(p50, 0.0, sla_cap_min))
    p90_cap = float(np.clip(p90, 0.0, sla_cap_min))
    over = bool(p90 > sla_cap_min)

    components = {
        "instrument_id": instrument_id,
        "priority": user_priority,
        "queue_length": q,
        "mix": mix,
        "higher_jobs_exp": float(higher),
        "equal_jobs_exp": float(equal),
        "effective_jobs_ahead": float(effective_jobs_ahead),
        "dur_mean": float(dur_mean),
        "dur_std": float(dur_std),
        "congestion_multiplier": float(cong),
        "baseline_min": float(baseline_min),
        "mean_wait_raw_min": float(p50),
        "p90_wait_raw_min": float(p90),
        "capped": over,
    }
    return p50_cap, p90_cap, float(p50), float(p90), over, components

instr_stats = build_instrument_service_stats(df)
pr_mix = build_priority_mix_by_instrument(df)


# =========================================================
# Infer required columns + numeric/categorical from model
# =========================================================
def find_preprocess(est):
    if hasattr(est, "named_steps") and "preprocess" in est.named_steps:
        return est.named_steps["preprocess"]
    if hasattr(est, "steps"):
        for _, step in est.steps:
            if isinstance(step, ColumnTransformer):
                return step
    if isinstance(est, ColumnTransformer):
        return est
    return None

def pipeline_has_numeric_imputer(trans):
    if isinstance(trans, Pipeline):
        for _, s in trans.steps:
            if isinstance(s, SimpleImputer) and getattr(s, "strategy", None) in ("mean", "median"):
                return True
    if isinstance(trans, SimpleImputer) and getattr(trans, "strategy", None) in ("mean", "median"):
        return True
    return False

def pipeline_has_cat_imputer(trans):
    if isinstance(trans, Pipeline):
        for _, s in trans.steps:
            if isinstance(s, SimpleImputer) and getattr(s, "strategy", None) in ("most_frequent", "constant"):
                return True
    if isinstance(trans, SimpleImputer) and getattr(trans, "strategy", None) in ("most_frequent", "constant"):
        return True
    return False

def infer_numeric_categorical(preprocess_ct: ColumnTransformer):
    num_cols, cat_cols = set(), set()
    if preprocess_ct is None:
        return num_cols, cat_cols
    transformers = getattr(preprocess_ct, "transformers_", None) or getattr(preprocess_ct, "transformers", [])
    for _, trans, cols in transformers:
        if cols is None or cols == "drop":
            continue
        try:
            col_list = list(cols) if not isinstance(cols, slice) else []
        except Exception:
            col_list = []
        if not col_list:
            continue
        if pipeline_has_numeric_imputer(trans):
            num_cols.update(col_list)
        elif pipeline_has_cat_imputer(trans):
            cat_cols.update(col_list)
    return num_cols, cat_cols

def get_required_cols_from_model(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    prep = find_preprocess(est)
    if prep is not None and hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    return None

preprocess = find_preprocess(model)
REQUIRED_COLS = get_required_cols_from_model(model)
NUM_EXPECTED, CAT_EXPECTED = infer_numeric_categorical(preprocess)

if not REQUIRED_COLS:
    EXCLUDE = {"experiment_id", "delay", "actual_duration", "is_delayed"}
    REQUIRED_COLS = [c for c in df.columns if c not in EXCLUDE]

if not NUM_EXPECTED and not CAT_EXPECTED:
    for c in REQUIRED_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            NUM_EXPECTED.add(c)
        else:
            CAT_EXPECTED.add(c)


# =========================================================
# Sanitize numeric-like columns (avoid median+MISSING crash)
# =========================================================
def sanitize_numeric_like_columns(X: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in required_cols:
        if c not in X.columns:
            continue
        s = X[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        s_str = s.astype(str)
        missing_mask = s.isna() | s_str.eq("MISSING") | s_str.eq("None") | s_str.eq("nan") | s_str.eq("NaN")
        s_num = pd.to_numeric(s_str.where(~missing_mask, np.nan), errors="coerce")
        non_missing_n = int((~missing_mask).sum())
        if non_missing_n == 0:
            continue
        ok_ratio = float(s_num.notna().sum()) / float(non_missing_n)
        if ok_ratio >= 0.90:
            X[c] = s_num
    return X

def ensure_required_features(X_in: pd.DataFrame) -> pd.DataFrame:
    X = X_in.copy()

    def is_numeric_col(c: str) -> bool:
        if c in NUM_EXPECTED:
            return True
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return True
        return False

    for c in REQUIRED_COLS:
        if c not in X.columns:
            X[c] = np.nan if is_numeric_col(c) else "MISSING"

    for c in REQUIRED_COLS:
        if is_numeric_col(c):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].astype("string").fillna("MISSING").astype(str)

    X_out = X[REQUIRED_COLS].copy()
    X_out = sanitize_numeric_like_columns(X_out, REQUIRED_COLS)
    return X_out


# =========================================================
# Predict helpers
# =========================================================
def predict_risk(m, X_row: pd.DataFrame) -> float:
    Xf = ensure_required_features(X_row)
    if hasattr(m, "predict_proba"):
        return float(m.predict_proba(Xf)[:, 1][0])
    if hasattr(m, "decision_function"):
        z = float(m.decision_function(Xf)[0])
        return float(1.0 / (1.0 + np.exp(-z)))
    yhat = float(pd.to_numeric(pd.Series(m.predict(Xf)), errors="coerce").fillna(0.0).iloc[0])
    return float(np.clip(yhat, 0.0, 1.0))

def predict_risk_batch(m, X_batch: pd.DataFrame) -> np.ndarray:
    Xf = ensure_required_features(X_batch)
    if hasattr(m, "predict_proba"):
        return m.predict_proba(Xf)[:, 1].astype(float)
    if hasattr(m, "decision_function"):
        z = m.decision_function(Xf).astype(float)
        return 1.0 / (1.0 + np.exp(-z))
    yhat = pd.to_numeric(pd.Series(m.predict(Xf)), errors="coerce").fillna(0.0).values
    return np.clip(yhat.astype(float), 0.0, 1.0)

def expected_delay_minutes_from_empirical(risk_prob: float) -> float:
    if "delay" not in df_work.columns:
        return 20.0 * (0.5 + 1.5 * risk_prob)
    d = pd.to_numeric(df_work["delay"], errors="coerce").fillna(0.0)
    delayed = d[d >= float(delay_min_threshold)]
    med = float(delayed.median()) if len(delayed) else max(10.0, float(delay_min_threshold))
    return float(risk_prob * med)


# =========================================================
# Families
# =========================================================
FAMILY_RULES = {
    "Device Reliability": ["instrument_type", "instrument_id", "ambient_temp", "telemetry", "tel_", "instrument_health", "instrument_cumulative", "instrument_recent"],
    "Queue & Scheduling": ["occupancy", "workload", "hour_of_day", "stress_index", "booking_time", "queue_", "priority", "day_of_week", "is_weekend", "days_since_start", "duration_to_queue"],
    "Workflow Complexity": ["experiment_type", "expected_duration", "experience_level"],
    "Reagents & Supply": ["reagent", "batch", "stock", "inventory", "reagent_batch_id"],
}
FAMILY_COLORS = {
    "Device Reliability": "#e74c3c",
    "Queue & Scheduling": "#f39c12",
    "Workflow Complexity": "#3498db",
    "Reagents & Supply": "#2ecc71",
    "Other": "#95a5a6",
}

def assign_family(feature: str) -> str:
    f = feature.lower()
    for fam, keys in FAMILY_RULES.items():
        if any(k in f for k in keys):
            return fam
    return "Other"


# =========================================================
# Importance (safe fallback)
# =========================================================
def choose_importance_scoring():
    X_test = ensure_required_features(df.head(200).copy())
    yhat = model.predict(X_test)
    yhat = pd.to_numeric(pd.Series(yhat), errors="coerce").fillna(0.0).values
    uniq = np.unique(np.round(yhat, 6))
    if set(uniq).issubset({0.0, 1.0}) and len(uniq) <= 2:
        return "accuracy"
    return "neg_mean_absolute_error"

IMPORTANCE_SCORING = choose_importance_scoring()

@st.cache_data(show_spinner=False)
def compute_perm_importance(sample_n=1500):
    X_full = ensure_required_features(df.copy())
    if "delay" in df_work.columns:
        y = (pd.to_numeric(df_work["delay"], errors="coerce").fillna(0.0) >= float(delay_min_threshold)).astype(int)
    else:
        y = pd.Series(np.zeros(len(X_full), dtype=int))
    n = min(sample_n, len(X_full))
    Xs = X_full.sample(n=n, random_state=42)
    ys = y.loc[Xs.index]
    r = permutation_importance(model, Xs, ys, n_repeats=3, random_state=42, scoring=IMPORTANCE_SCORING)
    imp = pd.DataFrame({"feature": REQUIRED_COLS, "importance": r.importances_mean})
    imp["family"] = imp["feature"].apply(assign_family)
    return imp.sort_values("importance", ascending=False)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### Operational Controls")
    thr = st.slider("Risk threshold", 0.0, 1.0, 0.15, 0.01)
    amber = st.slider("Amber upper bound", thr, 1.0, min(0.30, 1.0), 0.01)
    top_k = st.slider("Top drivers to show", 3, 20, 8, 1)

    st.divider()
    st.markdown("### Instrument / Priority / Queue")

    inst_opts = sorted(df["instrument_id"].astype(str).unique().tolist())
    instrument_user = st.selectbox("instrument_id", inst_opts, index=0)

    priority_user = st.selectbox("priority", ["Normal", "High", "Critical"], index=0)

    qmax = int(pd.to_numeric(df["queue_length"], errors="coerce").fillna(0).max()) if "queue_length" in df.columns else 0
    upper = max(200, qmax)
    queue_length_user = st.slider("queue_length (what-if)", 0, upper, min(20, upper), 1)

    st.divider()
    st.markdown("### Scientist Inputs (always on)")
    sw = pd.to_numeric(df.get("scientist_workload", 0.0), errors="coerce").dropna()
    oc = pd.to_numeric(df.get("lab_occupancy_level", 0.0), errors="coerce").dropna()
    sw_lo, sw_mid, sw_hi = (float(sw.quantile(0.05)), float(sw.median()), float(sw.quantile(0.95))) if len(sw) else (0.0, 0.0, 1.0)
    oc_lo, oc_mid, oc_hi = (float(oc.quantile(0.05)), float(oc.median()), float(oc.quantile(0.95))) if len(oc) else (0.0, 0.0, 1.0)
    if np.isclose(sw_lo, sw_hi): sw_lo, sw_hi = sw_mid - 1.0, sw_mid + 1.0
    if np.isclose(oc_lo, oc_hi): oc_lo, oc_hi = oc_mid - 1.0, oc_mid + 1.0

    scientist_workload_user = safe_slider("scientist_workload", sw_lo, sw_hi, sw_mid, max(0.01, (sw_hi - sw_lo)/100.0))
    lab_occupancy_user = safe_slider("lab_occupancy_level", oc_lo, oc_hi, oc_mid, max(0.01, (oc_hi - oc_lo)/100.0))

    st.divider()
    refresh = st.button("Refresh Importance", key="refresh_importance", use_container_width=True)

    if refresh or ("imp_df" not in st.session_state):
        try:
            with st.spinner("Computing permutation importance..."):
                st.session_state["imp_df"] = compute_perm_importance()
        except Exception as e:
            st.warning(f"Importance unavailable (fallback): {e}")
            st.session_state["imp_df"] = pd.DataFrame(
                {"feature": REQUIRED_COLS, "importance": np.zeros(len(REQUIRED_COLS)), "family": ["Other"] * len(REQUIRED_COLS)}
            )

    imp_df = st.session_state["imp_df"].copy()
    top_feats = imp_df["feature"].head(12).tolist() if "feature" in imp_df.columns else []

    st.divider()
    st.markdown("### Scenario Inputs")
    st.caption("Top model-driving features shown first (so risk changes).")

    inputs = {}
    HIDDEN = {
        "stress_index", "days_since_start", "hour_of_day",
        "day_of_week", "is_weekend",
        "duration_to_queue_ratio", "instrument_cumulative_hours",
        "instrument_recent_failure_rate",
        "priority", "queue_length", "queue_wait_min",
        "scientist_workload", "lab_occupancy_level",
        "instrument_id"
    }

    def qstats(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            return 0.0, 0.0, 1.0
        return float(s.quantile(0.05)), float(s.median()), float(s.quantile(0.95))

    for c in top_feats:
        if c in HIDDEN:
            continue
        is_num = (c in NUM_EXPECTED) or (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))
        if is_num:
            if c in df.columns:
                lo, mid, hi = qstats(c)
                if np.isclose(lo, hi):
                    lo, hi = mid - 1.0, mid + 1.0
                step = float(max(0.01, (hi - lo) / 100.0))
                inputs[c] = safe_slider(c, lo, hi, mid, step)
            else:
                inputs[c] = safe_slider(c, 0.0, 1.0, 0.5, 0.01)
        else:
            if c in df.columns:
                opts = sorted(df[c].dropna().astype(str).unique().tolist())
                if not opts:
                    opts = ["MISSING"]
                default = df[c].dropna().astype(str).value_counts().index[0] if df[c].dropna().shape[0] else opts[0]
                inputs[c] = st.selectbox(c, opts, index=opts.index(default) if default in opts else 0)
            else:
                inputs[c] = st.selectbox(c, ["MISSING"], index=0)

    st.caption("Advanced inputs (all remaining required features).")
    for c in REQUIRED_COLS:
        if c in inputs or c in HIDDEN:
            continue
        is_num = (c in NUM_EXPECTED) or (c in df.columns and pd.api.types.is_numeric_dtype(df[c]))
        if is_num:
            if c in df.columns:
                lo, mid, hi = qstats(c)
                if np.isclose(lo, hi):
                    lo, hi = mid - 1.0, mid + 1.0
                step = float(max(0.01, (hi - lo) / 100.0))
                inputs[c] = safe_slider(c, lo, hi, mid, step)
            else:
                inputs[c] = safe_slider(c, 0.0, 1.0, 0.5, 0.01)
        else:
            if c in df.columns:
                opts = sorted(df[c].dropna().astype(str).unique().tolist())
                if not opts:
                    opts = ["MISSING"]
                default = df[c].dropna().astype(str).value_counts().index[0] if df[c].dropna().shape[0] else opts[0]
                inputs[c] = st.selectbox(c, opts, index=opts.index(default) if default in opts else 0)
            else:
                inputs[c] = st.selectbox(c, ["MISSING"], index=0)


# =========================================================
# Build scenario row
# =========================================================
row = {k: v for k, v in inputs.items()}

row["scientist_workload"] = float(scientist_workload_user)
row["lab_occupancy_level"] = float(lab_occupancy_user)
row["stress_index"] = float(row["scientist_workload"] * row["lab_occupancy_level"])

row["instrument_id"] = str(instrument_user)  # still needed for queue model + features
row["priority"] = str(priority_user)
row["queue_length"] = int(queue_length_user)

queue_wait_p50_min = 0.0
queue_wait_p90_min = 0.0
queue_wait_raw_p50_min = 0.0
queue_wait_raw_p90_min = 0.0
queue_over = False
q_components = {}

try:
    p50_cap, p90_cap, p50_raw, p90_raw, queue_over, q_components = compute_queue_wait_realistic(
        instrument_id=row["instrument_id"],
        user_priority=row["priority"],
        queue_length=row["queue_length"],
        stress_index=row["stress_index"],
        instr_stats=instr_stats,
        pr_mix=pr_mix,
        df_all=df,
        sla_cap_min=QUEUE_SLA_CAP_MIN,
        baseline_min=QUEUE_BASELINE_MIN,
    )
    queue_wait_p50_min = float(p50_cap)
    queue_wait_p90_min = float(p90_cap)
    queue_wait_raw_p50_min = float(p50_raw)
    queue_wait_raw_p90_min = float(p90_raw)
except Exception as e:
    queue_wait_p50_min = float(np.clip(row["queue_length"] * 60.0 + QUEUE_BASELINE_MIN, 0, QUEUE_SLA_CAP_MIN))
    queue_wait_p90_min = queue_wait_p50_min
    queue_over = False
    q_components = {"error": str(e)}

row["queue_wait_min"] = float(queue_wait_p50_min)

X_user = pd.DataFrame([row])

if "hour_of_day" in REQUIRED_COLS and "hour_of_day" not in X_user.columns and "hour_of_day" in df.columns:
    X_user["hour_of_day"] = int(pd.to_numeric(df["hour_of_day"], errors="coerce").median())
if "days_since_start" in REQUIRED_COLS and "days_since_start" not in X_user.columns and "days_since_start" in df.columns:
    X_user["days_since_start"] = float(pd.to_numeric(df["days_since_start"], errors="coerce").median())

for c in ["ambient_temp", "ambient_temp_max", "ambient_temp_std", "telemetry_records", "tel_time_span_sec"]:
    if c in REQUIRED_COLS and c not in X_user.columns:
        X_user[c] = float(pd.to_numeric(df[c], errors="coerce").median()) if c in df.columns else 0.0

X_user_final = ensure_required_features(X_user)


# =========================================================
# Predict
# =========================================================
risk = predict_risk(model, X_user_final)
expected_delay = expected_delay_minutes_from_empirical(risk)

if risk < thr:
    tier, tier_color, border_color = "LOW RISK", "#2E7D32", "#2E7D32"
    action_text = "Proceed. Low likelihood of delay; monitor standard constraints."
elif risk < amber:
    tier, tier_color, border_color = "MODERATE RISK", "#F5A623", "#F5A623"
    action_text = "Monitor closely. Potential minor delays expected; consider 1 mitigation."
else:
    tier, tier_color, border_color = "HIGH RISK", "#B71C1C", "#B71C1C"
    action_text = "Mitigate now. Consider rescheduling/reallocating resources to reduce delay risk."


# =========================================================
# Local drivers
# =========================================================
imp_map = imp_df.set_index("feature")["importance"].to_dict() if "feature" in imp_df.columns else {}
dataset_median = {
    c: float(pd.to_numeric(df[c], errors="coerce").median())
    for c in REQUIRED_COLS
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
}

drivers = []
for f in REQUIRED_COLS:
    gi = float(imp_map.get(f, 0.0))
    if f in dataset_median and ((f in NUM_EXPECTED) or (f in df.columns and pd.api.types.is_numeric_dtype(df[f]))):
        dev = abs(float(pd.to_numeric(X_user_final.iloc[0][f], errors="coerce")) - dataset_median[f])
        score = gi * dev
    else:
        score = gi
    drivers.append({"feature": f, "family": assign_family(f), "driver_score": score, "current_value": str(X_user_final.iloc[0][f])})

drivers_df = pd.DataFrame(drivers).sort_values("driver_score", ascending=False)
local_family = drivers_df.groupby("family")["driver_score"].sum().sort_values(ascending=False).reset_index()


# =========================================================
# Manager insight
# =========================================================
def build_manager_insight():
    insights = []
    s = float(row["stress_index"])
    if s > 0:
        insights.append(("Stress Index", "Higher workload × occupancy increases congestion pressure and amplifies queueing delays."))
    insights.append(("Priority-aware Queue", "Higher-priority jobs are expected to move ahead; same-priority jobs are partially ahead on average."))
    insights.append(("Instrument Service Rate", "Queue wait depends on instrument-specific typical durations; selecting healthier/faster instruments can reduce delay exposure."))
    return insights[:3]

manager_insights = build_manager_insight()


# =========================================================
# Agent recommendations
# =========================================================
def simulate_change(updates: dict) -> float:
    tmp = X_user_final.copy()
    for k, v in updates.items():
        if k in tmp.columns:
            tmp.loc[tmp.index[0], k] = v
    return predict_risk(model, tmp)

def agent_recs():
    base = risk
    recs = []

    if risk < thr:
        recs.append({"action": "Proceed as planned", "rationale": "Risk is below operational threshold.", "impact": "No mitigation required; monitor near execution.", "delta": 0.0})
    elif risk < amber:
        recs.append({"action": "Proceed with mitigations", "rationale": "Moderate risk detected.", "impact": "Apply 1–2 mitigations below to de-risk.", "delta": 0.0})
    else:
        recs.append({"action": "Reschedule / reallocate", "rationale": "High risk detected.", "impact": "Adjust schedule/resources before execution.", "delta": 0.0})

    for feat in REQUIRED_COLS:
        if feat in {"queue_wait_min"}:
            continue
        if feat in df.columns and ((feat in NUM_EXPECTED) or pd.api.types.is_numeric_dtype(df[feat])):
            target = float(pd.to_numeric(df[feat], errors="coerce").quantile(0.25))
            current = float(pd.to_numeric(X_user_final.iloc[0][feat], errors="coerce"))
            if np.isclose(target, current):
                continue
            new_r = simulate_change({feat: target})
            delta = new_r - base
            if delta < -0.001:
                recs.append({
                    "action": f"Optimize {feat}",
                    "rationale": f"Adjusting {feat} from {current:.2f} to {target:.2f} reduces risk.",
                    "impact": f"{base:.1%} -> {new_r:.1%} ({delta:+.1%})",
                    "delta": delta,
                })

    header = recs[:1]
    mitigations = sorted(recs[1:], key=lambda r: r["delta"])
    return header + mitigations[:8]


# =========================================================
# Plotly helpers
# =========================================================
def make_risk_gauge(risk_val, thr_val, amber_val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_val * 100,
        number={"suffix": "%", "font": {"size": 48, "color": tier_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e5e7eb", "dtick": 25},
            "bar": {"color": tier_color, "thickness": 0.3},
            "bgcolor": "#f9fafb",
            "borderwidth": 0,
            "steps": [
                {"range": [0, thr_val * 100], "color": "#e8f5e9"},
                {"range": [thr_val * 100, amber_val * 100], "color": "#fff8e1"},
                {"range": [amber_val * 100, 100], "color": "#ffebee"},
            ],
            "threshold": {"line": {"color": "#374151", "width": 3}, "thickness": 0.8, "value": risk_val * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig

def make_local_drivers_bar(loc_fam):
    loc_sorted = loc_fam.iloc[::-1]
    colors = ["#95a5a6"] * len(loc_sorted)
    for i, fam in enumerate(loc_sorted["family"].astype(str).tolist()):
        colors[i] = FAMILY_COLORS.get(fam, "#95a5a6")

    fig = go.Figure(go.Bar(
        x=loc_sorted["driver_score"], y=loc_sorted["family"],
        orientation="h", marker_color=colors,
        text=loc_sorted["driver_score"].round(4), textposition="outside",
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Driver Score", yaxis_title="",
        font={"family": "Inter", "size": 12},
    )
    return fig


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="app-header">Roche Lab Delay Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">v3 model + realistic priority-aware queue (instrument-based)</div>', unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================
tab_predict, tab_analytics, tab_batch = st.tabs(["Predict", "Analytics", "Batch Predict"])


# =========================== TAB 1: PREDICT ===========================
with tab_predict:
    m1, m2, m3, m4, m5, m6 = st.columns(6, gap="medium")

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value" style="color:{tier_color}">{risk:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Risk Tier</div>
            <div class="metric-value metric-value-text" style="color:{tier_color}">{tier}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Expected Delay</div>
            <div class="metric-value" style="color:#1f2937">{expected_delay:.0f} min</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Stress Index</div>
            <div class="metric-value" style="color:#1f2937">{row["stress_index"]:.0f}</div>
        </div>""", unsafe_allow_html=True)

    # ✅ CHANGED: this card is now Priority (replaces Instrument)
    with m5:
        pr = str(row["priority"])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Priority</div>
            <div class="metric-value-fit" title="{pr}">{pr}</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Queue Wait (P50 / P90)</div>
            <div class="metric-value" style="color:#1f2937">
                {fmt_min_to_h(queue_wait_p50_min)} / {fmt_min_to_h(queue_wait_p90_min)}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    colA, colB = st.columns([1.1, 0.9], gap="large")

    with colA:
        st.plotly_chart(make_risk_gauge(risk, thr, amber), use_container_width=True)

        if queue_over:
            st.warning(
                f"Queue wait P90 exceeds SLA cap. Raw P50≈{fmt_min_to_h(queue_wait_raw_p50_min)}, "
                f"Raw P90≈{fmt_min_to_h(queue_wait_raw_p90_min)}. Display capped at 24h."
            )

        st.markdown(f"""
<div class="pred-card" style="border-left-color:{border_color}">
  <div class="pred-action"><b>Recommended Action:</b> {action_text}</div>
</div>
<p class="small-muted" style="margin-top:8px;">
Thresholds — Green: &lt;{thr:.0%} · Amber: {thr:.0%}–{amber:.0%} · Red: &gt;{amber:.0%}
</p>
""", unsafe_allow_html=True)

    with colB:
        st.markdown("#### Manager's Insight")
        items_html = '<div class="insight-box"><div class="insight-title">Operational Drivers</div>'
        for i, (title, bullet) in enumerate(manager_insights, 1):
            items_html += f'<div class="insight-item-title">{i}. {title}</div><div class="insight-bullet">{bullet}</div>'
        items_html += "</div>"
        st.markdown(items_html, unsafe_allow_html=True)

    st.divider()

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("#### Local Drivers (This Scenario)")
        st.dataframe(
            drivers_df.head(top_k)[["feature", "family", "driver_score", "current_value"]],
            use_container_width=True, hide_index=True,
        )
    with right:
        st.markdown("#### Drivers by Family")
        st.plotly_chart(make_local_drivers_bar(local_family), use_container_width=True)

    st.divider()

    st.markdown("#### AI Agent — Mitigation Recommendations")
    recs = agent_recs()
    for i, rec in enumerate(recs):
        delta = float(rec.get("delta", 0.0))
        delta_str = ""
        if delta < 0:
            delta_str = f'<span style="color:#2E7D32;font-weight:700">{delta:+.1%} risk</span>'
        elif delta > 0:
            delta_str = f'<span style="color:#B71C1C;font-weight:700">{delta:+.1%} risk</span>'

        st.markdown(f"""<div class="rec-card">
            <div class="rec-title">{i+1}. {rec["action"]}</div>
            <div class="rec-rationale">{rec["rationale"]}</div>
            <div class="rec-impact">{rec["impact"]} {delta_str}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("#### Tools")
    tool_col1, tool_col2 = st.columns(2, gap="medium")

    with tool_col1:
        st.markdown("**Export Current Prediction**")
        export_data = {
            "risk_score": [risk],
            "risk_tier": [tier],
            "expected_delay_min": [expected_delay],
            "instrument_id": [row["instrument_id"]],  # still export
            "priority": [row["priority"]],
            "queue_length": [int(row["queue_length"])],
            "queue_wait_p50_min_capped": [float(queue_wait_p50_min)],
            "queue_wait_p90_min_capped": [float(queue_wait_p90_min)],
            "queue_wait_p50_min_raw": [float(queue_wait_raw_p50_min)],
            "queue_wait_p90_min_raw": [float(queue_wait_raw_p90_min)],
            "scientist_workload": [float(row["scientist_workload"])],
            "lab_occupancy_level": [float(row["lab_occupancy_level"])],
            "stress_index": [float(row["stress_index"])],
            **{f: [X_user_final.iloc[0][f]] for f in REQUIRED_COLS},
        }
        export_df = pd.DataFrame(export_data)
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Prediction as CSV",
            csv_buf.getvalue(), "prediction_export.csv", "text/csv",
            use_container_width=True,
        )

    with tool_col2:
        with st.expander("Debug: queue model + input snapshot"):
            st.write("Queue components:", q_components)
            overlap = sorted(set(row.keys()).intersection(set(REQUIRED_COLS)))
            st.write("Inputs ∩ REQUIRED_COLS:", overlap)
            missing_cnt = int((X_user_final.iloc[0].astype(str) == "MISSING").sum())
            nan_cnt = int(pd.isna(X_user_final.iloc[0]).sum())
            st.write("MISSING:", missing_cnt, "| NaN:", nan_cnt)
            st.dataframe(X_user_final.T.head(120), use_container_width=True)


# =========================== TAB 2: ANALYTICS ===========================
with tab_analytics:
    st.markdown("#### Global Feature Importance (Permutation)")
    st.caption(f"Scoring = {IMPORTANCE_SCORING}")
    show_imp = imp_df.head(30).copy()
    show_imp["importance"] = pd.to_numeric(show_imp["importance"], errors="coerce").fillna(0.0).round(6)
    st.dataframe(show_imp[["feature", "importance", "family"]], use_container_width=True, hide_index=True)


# =========================== TAB 3: BATCH PREDICT ===========================
with tab_batch:
    st.markdown("#### Batch Prediction")
    st.markdown("Upload a CSV with experiment parameters to get risk predictions for all rows.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_df)} rows with {len(batch_df.columns)} columns")

            with st.spinner("Running batch predictions..."):
                if "booking_time" in batch_df.columns:
                    batch_df["booking_time"] = pd.to_datetime(batch_df["booking_time"], errors="coerce")
                    batch_df["hour_of_day"] = batch_df["booking_time"].dt.hour.fillna(9).astype(int)

                if "scientist_workload" in batch_df.columns and "lab_occupancy_level" in batch_df.columns:
                    batch_df["scientist_workload"] = pd.to_numeric(batch_df["scientist_workload"], errors="coerce").fillna(0.0)
                    batch_df["lab_occupancy_level"] = pd.to_numeric(batch_df["lab_occupancy_level"], errors="coerce").fillna(0.0)
                    batch_df["stress_index"] = batch_df["scientist_workload"] * batch_df["lab_occupancy_level"]

                if "experiment_id" in batch_df.columns:
                    batch_df = batch_df.merge(tel_agg, on="experiment_id", how="left", suffixes=("", "_tel"))
                    for c in ["ambient_temp", "ambient_temp_max", "ambient_temp_std", "telemetry_records", "tel_time_span_sec"]:
                        if c not in batch_df.columns and f"{c}_tel" in batch_df.columns:
                            batch_df[c] = batch_df[f"{c}_tel"]
                        if c in batch_df.columns:
                            batch_df[c] = pd.to_numeric(batch_df[c], errors="coerce").fillna(0.0)

                    # Reagent merge for batch predictions
                    batch_df = batch_df.merge(
                        df_rea[["experiment_id", "reagent_batch_id"]].drop_duplicates("experiment_id"),
                        on="experiment_id", how="left"
                    )
                if "reagent_batch_id" not in batch_df.columns:
                    batch_df["reagent_batch_id"] = "UNKNOWN"
                batch_df["reagent_batch_id"] = batch_df["reagent_batch_id"].fillna("UNKNOWN")

                if "priority" not in batch_df.columns:
                    batch_df["priority"] = "Normal"
                if "queue_length" not in batch_df.columns:
                    batch_df["queue_length"] = 0
                if "instrument_id" not in batch_df.columns:
                    batch_df["instrument_id"] = "UNKNOWN"

                q_p50_list = []
                q_p90_list = []
                for _, r in batch_df.iterrows():
                    stress = float(pd.to_numeric(r.get("stress_index", 0.0), errors="coerce") or 0.0)
                    p50_cap, p90_cap, *_ = compute_queue_wait_realistic(
                        instrument_id=str(r.get("instrument_id", "UNKNOWN")),
                        user_priority=str(r.get("priority", "Normal")),
                        queue_length=int(pd.to_numeric(r.get("queue_length", 0), errors="coerce") or 0),
                        stress_index=stress,
                        instr_stats=instr_stats,
                        pr_mix=pr_mix,
                        df_all=df,
                        sla_cap_min=QUEUE_SLA_CAP_MIN,
                        baseline_min=QUEUE_BASELINE_MIN,
                    )
                    q_p50_list.append(p50_cap)
                    q_p90_list.append(p90_cap)

                batch_df["queue_wait_p50_min"] = q_p50_list
                batch_df["queue_wait_p90_min"] = q_p90_list
                if "queue_wait_min" not in batch_df.columns:
                    batch_df["queue_wait_min"] = batch_df["queue_wait_p50_min"]

                risks = predict_risk_batch(model, batch_df)
                batch_df["predicted_risk"] = risks
                batch_df["risk_tier"] = pd.cut(risks, bins=[-0.01, thr, amber, 1.01], labels=["LOW", "MODERATE", "HIGH"])
                batch_df["expected_delay_min"] = [expected_delay_minutes_from_empirical(r) for r in risks]

            show_cols = [
                "predicted_risk", "risk_tier", "expected_delay_min",
                "instrument_id", "priority", "queue_length",
                "queue_wait_p50_min", "queue_wait_p90_min"
            ]
            st.dataframe(batch_df[show_cols].head(200), use_container_width=True, hide_index=True)

            out_buf = io.StringIO()
            batch_df.to_csv(out_buf, index=False)
            st.download_button("Download Results as CSV", out_buf.getvalue(), "batch_predictions.csv", "text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file to get started.")

st.markdown("---")
st.caption("Roche Capstone — Delay Risk Predictor | Streamlit & Plotly | v3 model + realistic priority-aware queue")