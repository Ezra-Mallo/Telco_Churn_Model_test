# =========================================================================
# STEP 73: WRITE THE SCORING MODULE — telco_serving.py
# =========================================================================
# ══ telco_serving.py · the production scoring path for the Telco churn model ══
# Imported by the batch job, the API (app.py) and the Phase 6 monitor.
# Repeats every Phase 2 step that is NOT inside the pickled pipeline.
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ADDON_NOT_APPLICABLE = {"No internet service": "No", "No phone service": "No"}


# ── Phase 2a · Steps 15, 20, 21, 22 (stateless cleaning) ───────────────────
def clean_raw(raw, config):
    data = pd.DataFrame(raw).copy()
    for col in ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")             # Step 15
    for col in config["zero_impossible_cols"]:
        data[col] = data[col].replace(0, np.nan)                            # Step 20
    new_customer = (data["tenure"] == 0) & data["TotalCharges"].isna()
    data.loc[new_customer, "TotalCharges"] = 0.0                            # Step 21
    for col in config["service_addons"]:
        data[col] = data[col].replace(ADDON_NOT_APPLICABLE)                 # Step 22
    return data


# ── Phase 2b · Steps 36–38 (feature engineering) ───────────────────────────
def add_features(data, inner_bin_edges):
    data = data.copy()
    data["charges_per_month"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["monthly_to_total"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1)
    data["tenure_per_charge"] = data["tenure"] / (data["MonthlyCharges"] + 1)
    fiber = (data["InternetService"] == "Fiber optic").astype(int)
    paperless = (data["PaperlessBilling"] == "Yes").astype(int)
    senior = data["SeniorCitizen"].astype(int)
    data["fiber_x_monthly"] = fiber * data["MonthlyCharges"]
    data["paperless_x_monthly"] = paperless * data["MonthlyCharges"]
    data["senior_x_tenure"] = senior * data["tenure"]
    edges = [-np.inf] + list(inner_bin_edges) + [np.inf]                   # TRAIN edges, never recomputed
    data["tenure_bin"] = pd.cut(data["tenure"], bins=edges, labels=False)
    return data


class ChurnScorer:
    """Loads one deployment folder and scores raw customer records."""

    def __init__(self, folder, log_path=None):
        self.folder = folder
        with open(os.path.join(folder, "model_config.json")) as f:
            self.config = json.load(f)
        self.model = joblib.load(os.path.join(folder, self.config["model_file"]))
        self.threshold = self.config["threshold"]
        self.version = self.config["model_version"]
        self.features = self.config["raw_features"]
        self.log_path = log_path

    # ── input contract (Phase 2 Step 17, applied to every request) ──────────
    def validate(self, raw):
        missing_cols = [c for c in self.features if c not in raw.columns]
        if missing_cols:
            raise ValueError(f"Request is missing columns: {missing_cols}")
        n = len(raw)
        errors = [[] for _ in range(n)]
        warnings = [[] for _ in range(n)]
        for col, (low, high) in self.config["numeric_limits"].items():
            text = raw[col].astype(str).str.strip()
            values = pd.to_numeric(raw[col], errors="coerce")
            empty = raw[col].isna() | (text == "") | (text.str.lower() == "nan")
            not_number = values.isna() & ~empty
            out_of_range = values.notna() & ((values < low) | (values > high))
            for i in range(n):
                if not_number.iloc[i]:
                    errors[i].append(f"{col} is not a number ({raw[col].iloc[i]!r})")
                elif out_of_range.iloc[i]:
                    errors[i].append(f"{col}={values.iloc[i]} outside [{low}, {high}]")
                elif empty.iloc[i] and col in self.config["required_numeric"]:
                    errors[i].append(f"{col} is missing")
                elif empty.iloc[i] and col == "TotalCharges":
                    tenure = pd.to_numeric(raw["tenure"].iloc[i], errors="coerce")
                    if tenure != 0:
                        warnings[i].append("TotalCharges blank for an existing customer — imputed")
                elif empty.iloc[i]:
                    warnings[i].append(f"{col} missing — imputed")
        for col, allowed in self.config["allowed_categories"].items():
            for i, value in enumerate(raw[col]):
                if pd.isna(value):
                    warnings[i].append(f"{col} missing — imputed")
                elif value not in allowed:
                    warnings[i].append(f"{col}={value!r} never seen in training")
        return errors, warnings

    def _prepare(self, raw):
        cleaned = clean_raw(raw[self.features], self.config)
        return add_features(cleaned, self.config["tenure_bin_inner_edges"])

    # ── score: one row out per row in ──────────────────────────────────────
    def score(self, raw, log=True):
        raw = pd.DataFrame(raw).reset_index(drop=True)
        errors, warnings = self.validate(raw)
        ok = np.array([len(e) == 0 for e in errors])
        scores = np.full(len(raw), np.nan)
        if ok.any():
            features = self._prepare(raw[ok])
            scores[ok] = self.model.predict_proba(features)[:, 1]
        result = pd.DataFrame({
            "customerID": raw["customerID"] if "customerID" in raw.columns else range(len(raw)),
            "churn_score": np.round(scores, 6),
            "flagged": ok & (np.nan_to_num(scores) >= self.threshold),
            "status": np.where(ok, "scored", "rejected"),
            "message": ["; ".join(e + w) for e, w in zip(errors, warnings)],
            "model_version": self.version,
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if log and self.log_path:
            self._log(raw, result)
        return result

    # ── reason codes: which inputs push this customer's score up? ─────────
    def explain(self, raw, top_n=3):
        raw = pd.DataFrame(raw).reset_index(drop=True)
        cleaned = clean_raw(raw[self.features], self.config)
        base = self.model.predict_proba(add_features(cleaned, self.config["tenure_bin_inner_edges"]))[:, 1]
        effects = {}
        for col, typical in self.config["reference_values"].items():
            changed = cleaned.copy()
            changed[col] = typical                                   # replace with a typical customer's value
            new = self.model.predict_proba(add_features(changed, self.config["tenure_bin_inner_edges"]))[:, 1]
            effects[col] = base - new                                # > 0 → this input raises the risk
        effects = pd.DataFrame(effects)
        reasons = []
        for i in range(len(raw)):
            top = effects.iloc[i].sort_values(ascending=False).head(top_n)
            parts = []
            for col, eff in top.items():
                if eff > 0.005:
                    shown = str(raw[col].iloc[i]).strip() or "(blank)"
                    parts.append(f"{col}={shown} (+{eff:.2f})")
            reasons.append("; ".join(parts) if parts else "no single input stands out")
        return reasons

    # ── batch policy: rank, cap at capacity, attach reasons ───────────────
    def call_list(self, raw, max_flag_rate):
        raw = pd.DataFrame(raw).reset_index(drop=True)
        scored = self.score(raw)
        n_scored = int((scored["status"] == "scored").sum())
        capacity = int(np.floor(max_flag_rate * n_scored))
        flagged = scored[scored["flagged"]].sort_values("churn_score", ascending=False)
        capped = flagged.head(capacity).copy()
        capped["rank"] = np.arange(1, len(capped) + 1)
        capped["reasons"] = self.explain(raw.loc[capped.index]) if len(capped) else []
        summary = {"customers": len(raw), "scored": n_scored, "rejected": len(raw) - n_scored,
                   "above_threshold": len(flagged), "capacity": capacity, "on_call_list": len(capped),
                   "cut_by_capacity": max(0, len(flagged) - capacity)}
        return capped, summary

    # ── prediction log (Phase 6 reads this) ───────────────────────────────
    def _log(self, raw, result):
        record = pd.concat([result.reset_index(drop=True),
                            raw[self.features].reset_index(drop=True)], axis=1)
        new_file = not os.path.exists(self.log_path)
        record.to_csv(self.log_path, mode="a", header=new_file, index=False)
