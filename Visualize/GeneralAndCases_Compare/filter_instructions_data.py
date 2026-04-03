import pandas as pd
import json

# Config
EXCEL_PATH = "Compare_IC_vs_Retrieval.xlsx"  # Change if needed
SHEET_NAME = "Sheet1"
OUT_DIR = "Cases_JSON"

import os

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Ensure numeric
df["F1_ret"] = pd.to_numeric(df["F1_ret"], errors="coerce")
df["ROUGE-L_ret"] = pd.to_numeric(df["ROUGE-L_ret"], errors="coerce")

# Scale if needed
for col in ["F1_ret", "ROUGE-L_ret"]:
    if df[col].dropna().quantile(0.95) <= 1.2:
        df[col] = df[col] * 100

# Define cases
cases = {
    "f1_high_rougel_low": lambda f1, rouge: (f1 >= 90) & (rouge <= 50),
    "f1_high_rougel_high": lambda f1, rouge: (f1 >= 90) & (rouge > 50),
    "f1_low_rougel_low": lambda f1, rouge: (f1 <= 10) & (rouge <= 50),
    "f1_low_rougel_high": lambda f1, rouge: (f1 <= 10) & (rouge > 50),
}

# Ensure numeric for IC columns as well
df["F1_ic"] = pd.to_numeric(df["F1_ic"], errors="coerce")
df["ROUGE-L_ic"] = pd.to_numeric(df["ROUGE-L_ic"], errors="coerce")
for col in ["F1_ic", "ROUGE-L_ic"]:
    if df[col].dropna().quantile(0.95) <= 1.2:
        df[col] = df[col] * 100

# Extract and save records for each case where BOTH IC and RET meet the requirement
for case_name, case_fn in cases.items():
    mask_ic = case_fn(df["F1_ic"], df["ROUGE-L_ic"])
    mask_ret = case_fn(df["F1_ret"], df["ROUGE-L_ret"])
    mask = mask_ic & mask_ret  # Only records where both IC and RET meet the case
    records = df.loc[mask, ["image_id", "pair_id"]]
    records_list = records.to_dict(orient="records")
    out_path = os.path.join(OUT_DIR, f"{case_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records_list)} records to {out_path}")
