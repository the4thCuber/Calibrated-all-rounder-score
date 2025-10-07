import pandas as pd
import zipfile
from pathlib import Path
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle

EXPORT_DIR = Path("~/wca_tsv")
AVERAGE_RANKS = EXPORT_DIR / "WCA_export_RanksAverage.tsv"

def load_avg_ranks():
    """Extract AverageRanks.tsv into a DataFrame."""
    df = pd.read_csv(AVERAGE_RANKS, sep='\t', low_memory=False)
    print(df.columns.tolist())
    return df

def build_pr_dict(df):
    """Build dict of {wca_id: {event_id: average_in_centiseconds}}."""
    prs = {}
    for _, row in tqdm(df.iterrows(), desc=" ", unit="IDs"):
        pid = row["personId"]
        event = row["eventId"]
        avg = row["best"]
        # only keep valid averages (>0)
        if pid in prs and pd.notna(avg) and avg > 0:
            prs[pid][event] = avg
        if pid not in prs:
            prs[pid] = {}
        prs[pid][event] = avg
    return prs

if __name__ == "__main__":
    print("Loading AverageRanks...")
    df = load_avg_ranks()
    print(f"Loaded {len(df)} rows.")

    print("Building PR dictionary...")
    pr_dict = build_pr_dict(df)

    # Save to file
    pickle = EXPORT_DIR / "allprs.pkl"
    with open(pickle, "wb") as f:
        pickle.dump(pr_dict, f)

    with open("allprs.json", "w") as f:
        json.dump(pr_dict, f)

    # Optional: also make a flat CSV
    pd.DataFrame([
        {"personId": pid, "eventId": e, "average": avg}
        for pid, events in pr_dict.items()
        for e, avg in events.items()
    ]).to_csv("allprs.csv", index=False)

    print("Saved allprs.json and allprs.csv")
