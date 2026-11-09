# ══ telco_helpers.py · shared custom objects for the Telco pipeline ══════
# Imported by Phase 2 (this notebook) and Phase 3 (training).
# Lives in a real .py file so the saved pipeline's pickle can resolve
# these objects by their module path instead of __main__.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── From Step 15 · coerce_to_numeric ─────────────────────────────────────
def coerce_to_numeric(X):
    """Convert text-looking columns to float; unparseable cells become NaN."""
    X_df = pd.DataFrame(X).copy()
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    return X_df


# ── From Step 28 · OutlierClipper ────────────────────────────────────────
# Custom sklearn transformer that clips numeric features to IQR bounds.
# Bounds are learned during fit() and stored in self.bounds_.
class OutlierClipper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            q1, q3 = X_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (
                q1 - self.factor * iqr,
                q3 + self.factor * iqr,
            )
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            X_df[col] = X_df[col].clip(low, high)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.bounds_.keys()), dtype=object)
