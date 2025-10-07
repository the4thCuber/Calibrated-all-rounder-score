# compute-cas.py
import os
import subprocess
import pandas as pd
import requests, json
import time
import random

"""def fetch_prs(wca_id):
    url = f"https://www.worldcubeassociation.org/api/v0/persons/{wca_id}/personal_records"
    r = requests.get(url)
    # print(f"Fetching {wca_id}: {r.status_code}")

    if r.status_code != 200:
        return {}

    try:
        data = r.json()
    except Exception as e:
        print(f"Error parsing JSON for {wca_id}: {e}")
        return {}

    # If it's already a dict (older API), just return it
    if isinstance(data, dict):
        return data

    # Otherwise it's the new list format — convert to nested dict structure
    records = {}
    for rec in data:
        if rec.get("type") != "average":
            continue
        event = rec.get("eventId")
        best = rec.get("best")
        if event and best and best > 0:
            records[event] = {"average": {"best": best}}

    return records
"""

def compute_S(T_Er, T_Ea, T_El, T_Ep, Smin, Smax):
    R_max = 1
    R_min = (T_Er + T_Ea) / (T_El + T_Ea)
    R_p = (T_Er + T_Ea) / (T_Ep + T_Ea)
    return (R_p - R_min) / (R_max - R_min) * (Smax - Smin) + Smin

def cc(prs):
    if not os.path.exists("CAS_Results.csv"):
        subprocess.run(["python", "scrape-results.py"])

    df = pd.read_csv("CAS_Results.csv")
    event_names = df["Event"].tolist()

    scores = []
    for _, row in df.iterrows():
        event = row["Event"]
        if event in prs:
            best = prs[event]
            S = compute_S(row["WR"], row["Anchor"], row["Worst"], best / 100.0, row["Smin"], row["Smax"])
        else:
            S = row["Smin"]
        scores.append(round(S, 4))
    scores.append(round(sum(scores)/17,4))

    if scores:
        # print(f"CAS = {sum(scores)/len(scores):.4f}")
        return scores
    else:
        print("No valid PRs to calculate CAS for {wca_id}")
