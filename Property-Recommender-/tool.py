import pandas as pd
import json, glob

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}_"))  #Flatten the nested dict keys recursively
        else:
            out[f"{prefix}{k}"] = v
    return out

rows = []
for fp in glob.glob("data/appraisals_dataset.json"):
    with open(fp) as f:
        raw = json.load(f)
    for j in raw:
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
