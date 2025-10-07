"""
End-to-end WCA Lasso pipeline:
1. Download WCA export TSV (latest).
2. Extract competitors ranked <=400 in any event.
3. Pivot into wide competitor x event matrix.
4. Run Lasso regression predicting chosen target event.
"""

import requests, zipfile, io, os, math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

EXPORT_INDEX_URL = "https://www.worldcubeassociation.org/api/v0/export/public"
EXTRACT_DIR = "wca_tsv"

def get_latest_export_url():
    r = requests.get(EXPORT_INDEX_URL, timeout=30)
    r.raise_for_status()
    latest = r.json() # most recent export
    return latest["tsv_url"], latest["export_date"]

def download_and_extract(tsv_url, outdir=EXTRACT_DIR):
   
    if os.path.exists(outdir) and os.listdir(outdir):
        print("Using cached export in", outdir)
        return outdir

    print(f"Downloading {tsv_url} ...")
    r = requests.get(tsv_url, stream=True, timeout=120)
    r.raise_for_status()
    buf = io.BytesIO()
    total = int(r.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(8192):
            buf.write(chunk)
            pbar.update(len(chunk))
    buf.seek(0)

    print("Extracting TSVs...")
    with zipfile.ZipFile(buf) as z:
        members = z.namelist()
        needed = [m for m in members if any(x in m for x in (
            "RanksSingle.tsv", "RanksAverage.tsv", "Results.tsv", "Persons.tsv"
        ))]
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        for name in needed:
            z.extract(name, outdir)
    return outdir

def find_file(root, name):
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(name):
                return os.path.join(r, f)
    raise FileNotFoundError(name)

# We thought way too hard about this
def decode_mbld(value):
    if pd.isna(value):
        return np.nan

    try:
        v = int(value)
    except Exception:
        return np.nan

    if v <= 0:
        return np.nan

    # time is stored as 5 digits in seconds in both encodings
    # We'll extract by integer math so we tolerate missing leading digits.
    # Last 5 digits => timeInSeconds when dividing appropriately.
    if v >= 1_000_000_000:
        # old format: 1 S S A A TTTTT
        time_sec = v % 100000
        aa = (v // 100000) % 100         # attempted (AA)
        ss = (v // 10000000) % 100       # SS encodes 99 - solved
        solved = 99 - ss
        attempted = aa
    else:
        # new format: 0 D D T T T T T M M  (0 may have been dropped)
        mm = v % 100                     # missed cubes (MM)
        t = (v // 100) % 100000   # TTTTT (seconds)
        time_sec = 0.0 if math.isnan(t) else t
        dd = v // 10000000               # DD (difference field)
        difference = 99 - dd
        missed = mm
        solved = difference + missed
        attempted = solved + missed

    # guard against nonsense values
    if solved < 0 or attempted < 0:
        return np.nan

    # points used for ranking: 2*solved - attempted
    points = 2 * solved - attempted

    # handle 'unknown' time sentinel
    if time_sec == 99999:
        time_sec = np.nan
    else:
        time_sec = float(time_sec)

    # print("Solved:" + str(solved) + " Attempted:" + str(attempted) + " Time:" + str(time_sec))
    # print(points + solved/1000 - time_sec/1e6)
    return points + attempted/1000 - time_sec/1e6

def get_ids(ranks, n):
    print(f"Writing list of top {n} WCA IDs...")
    counts = ranks["personId"].value_counts()
    top_ids = counts.head(n).index.tolist()

    with open(str(n)+"WCAIDs.txt", "w") as f:
        f.write("\n".join(top_ids))

    print(f"Saved {len(top_ids)} IDs to {str(n)+"WCAIDs.txt"}")
    return top_ids

if __name__ == "__main__":
    if input("Custom arguments? [y/n]") == "y":
        ancrank = int(input("Rank to use for anchor result (Default=100): "))
        pct = float(input("Percentile cutoff for worst solve (~99): "))
        logb = np.log(2**float(input("Value of 2^b for S_min scaling (Default=.333): ")))
        alpha = float(input("Alpha value for lasso regression (Default=0.01): "))
    else:
        ancrank = 100
        pct = 99
        b = 0.125
        logb = np.log(2 ** b)
        alpha = 0.01
    print(f"Using {ancrank} as anchor, {pct} as percentile cutoff, b={b:.4f}, alpha={alpha:.4f}")
    usesingle = ["333mbf"]
    if input("Use mean for FMC? [y/n]") != "y":
        usesingle.append("333fm")
    if input("Use mean for 4BLD? [y/n]") != "y":
        usesingle.append("444bf")
    if input("Use mean for 5BLD? [y/n]") != "y":
        usesingle.append("555bf")

    print("Finding last export...")
    url, date = get_latest_export_url()
    print("Latest export date:", date)

    export_dir = download_and_extract(url)
    print("Building matrix...")
    ranks_single = pd.read_csv(find_file(export_dir,"RanksSingle.tsv"), sep="\t", dtype={"personId": str}, low_memory=False)
    print("Parsed RanksSingle!")
    ranks_avg = pd.read_csv(find_file(export_dir,"RanksAverage.tsv"), sep="\t", dtype={"personId": str}, low_memory=False)
    print("Parsed RanksAverage!")
    results = pd.read_csv(find_file(export_dir,"Results.tsv"), sep="\t", low_memory=False)
    print("Parsed Results!")
    persons = pd.read_csv(find_file(export_dir,"Persons.tsv"), sep="\t")
    print("Parsed Persons!")

    if input("Build list of WCA IDs? [y/n]") == "y":
        num = int(input("Number of IDs: "))
        top_ids = get_ids(ranks_avg, num)

    # Filter top 400 competitors
    print("Filtering top 400...")
    top_ids = set(ranks_single.loc[ranks_single["worldRank"]<=400, "personId"]) | \
              set(ranks_avg.loc[ranks_avg["worldRank"]<=400, "personId"])
    topresults = results[results["personId"].isin(top_ids)]

    # Pivot: competitor × event
    print("Pivoting and formatting")
    wide = topresults.pivot_table(index="personId", columns="eventId", values="best", aggfunc="min")

    # Attach names
    people = persons[["id","name"]].rename(columns={"id": "personId"})
    wide = wide.merge(people, on="personId", how="left")

    # Format MBLD
    if "333mbf" in wide.columns:
        wide["333mbf"] = pd.to_numeric(wide["333mbf"], errors="coerce")
        wide["333mbf"] = wide["333mbf"].apply(decode_mbld)
    
    summary_rows = []

    df = results.copy()
    df["best"] = pd.to_numeric(df["best"], errors="coerce")
    total = df["personId"].nunique()

    for e in ["333","222","444","555","666","777","333bf","333fm","333oh","clock","minx","pyram","skewb","sq1","444bf","555bf","333mbf"]:

        # Slice average rankings and store wr and average
        # Except in 4BLD, 5BLD, MBLD, and FMC
        if e in usesingle:
            ranks_src = ranks_single
        else:
            ranks_src = ranks_avg
        ranks_event = ranks_src[ranks_src["eventId"] == e].copy()
        ranks_event = ranks_event[ranks_event["best"] > 0].sort_values("best", ascending=True)
        wr = ranks_event["best"].iloc[0] / 100.0
        anchor = ranks_event["best"].iloc[min(ancrank-1, len(ranks_event)-1)] / 100.0

        # Drop competitors missing target
        wide = wide.dropna(subset=[e])

        # Slice DataFrame
        sub = df[df["eventId"] == e] 

        # Drop old events and raw MBLD
        X = wide.drop(columns=["personId","name","wcaId","333ft","333mbo","magic","mmagic",e], errors="ignore")
        y = wide[e]

        # Fill missing features (mean imputation for simplicity)
        X = X.fillna(X.mean())

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )
    
        # Scale features
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
    
        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_test_s  = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()
        
        # Train
        model = Lasso(alpha, max_iter=10000)
        model.fit(X_train_scaled, y_train_s)
        # Predict
        y_pred = model.predict(X_test_scaled)
    
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_test_s, y_pred)) 

        valid = sub[sub["best"].notna() & sub["best"] != 0]
        finished = valid["personId"].nunique()
        success_pct = (finished / total * 100) if total > 0 else np.nan

        if not valid.empty:
            p99 = np.percentile(valid["best"], pct)
        else:
            p99 = np.nan

        print(f"\n\nEvent: {e}\n % with result: {100 * finished / total}\nLasso results:\nRMSE: {rmse}\n")
        for feat, coef in zip(X.columns, model.coef_):
            if abs(coef) > 1e-6:
                print(f" {feat} coef: {coef:.4f}")

        print(f"Total: {total}, Finished: {finished}, Ratio: {total/finished}, Smin: {np.log(total/finished)/logb}")
        summary_rows.append({
            "Event": e,
            "Total Competitors": total,
            "Successful": finished,
            "Percent": round(success_pct, 5),
            "WR": wr,
            "Anchor": anchor,
            "Worst": round(p99,5)/100.0 if not math.isnan(p99) else None,
            "Smin": np.log(total/finished)/logb,
            "Smax": 100 * rmse
            })

    pd.DataFrame(summary_rows).to_csv("CAS_Results.csv", index=False)
    print("===FINAL CSV DUMP===")
    os.system("cat CAS_Results.csv")
