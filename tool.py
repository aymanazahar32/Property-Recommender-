import pandas as pd
import json, json5, glob, os




"""
 building a comp-selection system from a messy real-estate dataset.  wrote a Python loader that flattens each appraisal into one row per subject-candidate pair, keeps every missing value as NaN, and now robustly parses any file shape—JSON list, ND-JSON, or back-to-back multi-line objects—via a bracket-counting stream that falls back to json5 for un-quoted keys; athe next milestone is to engineer delta features and handle NaNs (LightGBM/XGBoost can ingest them directly, with optional missing-value indicators). The loader should end with something like Shape: (≈9 800, ≈120) and a non-empty DataFrame; from here the task is to add feature engineering, train a LambdaMART ranker, and plug in SHAP for explanations.
"""

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}_"))  #Flatten the nested dict keys recursively
        else:
            out[f"{prefix}{k}"] = v
    return out

def stream_concat(fp):
    buf, depth, in_str, esc = "", 0, False, False
    while True:
        ch = fp.read(1)
        if not ch:
            break
        buf += ch
        if in_str:
            esc = (not esc and ch == "\\") or (esc and ch != "\\")
            if ch == '"' and not esc:
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(buf)
                    except json.JSONDecodeError:
                        yield json5.loads(buf)
                    buf = ""

rows = []

path = "../appraisals_dataset.json"
with open(path) as f:
    first = f.read(1)
    f.seek(0)

    records = None
    try:
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    except Exception:
        pass

    if records is None:            # either quick load failed or produced nothing
        f.seek(0)
        records = list(stream_concat(f))

    for j in records:
        subj = flatten(j["subject"], "S_")      #flatten subject columns
        for cand in j["candidates"]:
            rows.append({
                "AppraisalID":  j["appraisal_id"],
                "CandidateID":  cand["id"],
                "chosen":       int(cand["is_comp"]),
                **subj,
                **flatten(cand, "C_")           #flatten candidate columns
            })

df = pd.DataFrame(rows)
print(df.shape)
print(df.head())
