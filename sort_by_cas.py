import pandas as pd
from datetime import datetime
import computecas as cc
from tqdm import tqdm
import json
from pathlib import Path

file_path = "all_prs.json"
with open(file_path, "r", encoding="utf-8") as f:
    pr_dict = json.load(f)

# Compute CAS for each WCA ID
results = {}
for comp in tqdm(pr_dict, desc="Calculating CAS", unit="competitor"):
    # print(pr_dict[comp])
    try:
        cas_value = cc.cc(pr_dict[comp])
        # print(cas_value)
        results[comp] = cas_value
        # print(results)
            # print(f"{comp}: {cas_value:.3f}")
    except Exception as e:
        print(f"Error computing CAS for {comp}: {e}")

df = pd.DataFrame([
    {"WCA_ID": k, **{f"Event_{i+1}": v for i, v in enumerate(scores)}, "CAS_Avg": scores[-1]}
    for k, scores in results.items() if isinstance(scores, list) and len(scores) > 0
])
df = df.sort_values(by="CAS_Avg", ascending=False).reset_index(drop=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"CAS_Results_{timestamp}.csv"
df.to_csv(outfile, index=False)
print(f"\nSaved CAS rankings to {outfile}")
