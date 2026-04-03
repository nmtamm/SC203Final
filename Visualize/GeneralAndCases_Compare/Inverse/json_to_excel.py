import json
import glob
from pathlib import Path
import pandas as pd

JSON_GLOB = "Metric/*.json"
OUT_EXCEL = "Metric_IC.xlsx"

rows = []

for fp in sorted(glob.glob(JSON_GLOB)):
    run = Path(fp).stem  # json filename without .json

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    for r in data.get("results", []):
        rows.append({
            "run": run,
            "prefix": r.get("prefix"),      # image id
            "pair": r.get("pair"),          # includes recipe id
            "IoU": r.get("iou"),
            "F1": r.get("f1"),
            "ROUGE-L": r.get("rougeL"),
            "SacreBLEU": r.get("sacrebleu"),
        })

df = pd.DataFrame(rows)

# average across recipes per image
metric_cols = ["IoU", "F1", "ROUGE-L", "SacreBLEU"]
df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

# If you want per-image averages *within each JSON file/run*:
df_img_avg = (
    df.groupby(["run", "prefix"], as_index=False)[metric_cols]
      .mean()
)

# Optional: also keep how many recipes each image had
df_counts = df.groupby(["run", "prefix"], as_index=False).size().rename(columns={"size": "num_recipes"})
df_img_avg = df_img_avg.merge(df_counts, on=["run", "prefix"], how="left")

with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="per_recipe_rows", index=False)
    df_img_avg.to_excel(writer, sheet_name="per_image_avg", index=False)

print("Wrote:", OUT_EXCEL)
print("per_recipe_rows:", len(df), "rows")
print("per_image_avg:", len(df_img_avg), "rows")
