# ══ app.py · real-time churn scoring API ════════════════════════════════
# Run from the folder that contains deployment/:   uvicorn app:app --reload
# Then open http://127.0.0.1:8000/docs for the interactive test page.
import json
import sys
from typing import List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "deployment")              # telco_helpers + telco_serving travel with the model
from telco_serving import ChurnScorer

scorer = ChurnScorer("deployment", log_path="deployment/prediction_log.csv")
app = FastAPI(title="Telco Churn Scoring API", version=scorer.version)


class Customer(BaseModel):
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Union[float, str]          # the billing system sends " " for new customers


# ── Diagnostic endpoints ───────────────────────────────────────────────
def health():
    return {"status": "ok", "model_version": scorer.version}

def model_info():
    c = scorer.config
    return {"model_version": c["model_version"], "algorithm": c["algorithm"],
            "threshold": c["threshold"], "evaluation": c["evaluation"],
            "conditions": c["conditions"]}

# ── Scoring endpoint ──────────────────────────────────────────────────
def predict(customers: List[Customer]):
    rows = []
    for c in customers:
        rows.append(c.model_dump() if hasattr(c, "model_dump") else c.dict())
    raw = pd.DataFrame(rows)
    try:
        result = scorer.score(raw)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    result["reasons"] = ""
    ok = result["status"] == "scored"
    if ok.any():
        result.loc[ok, "reasons"] = scorer.explain(raw[ok.to_numpy()])
    return json.loads(result.to_json(orient="records"))      # NaN → null, numpy → plain JSON
