import json
import pandas as pd


def build_compare_table(
    inverse_xlsx: str,
    retrieval_xlsx: str,
    mapping_json: str,
    output_xlsx: str,
    inverse_sheet: str = "per_image_avg",
    retrieval_sheet: str = "Sheet1",
) -> None:
    # 1) Read mapping JSON
    with open(mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    map_df = pd.DataFrame(mapping).rename(
        columns={
            "image id": "image_id",
            "pair id for retrieval": "pair_id",
        }
    )

    keep_cols = [c for c in ["number", "image_id", "pair_id"] if c in map_df.columns]
    map_df = map_df[keep_cols].copy()

    # 2) Read Excel sheets
    inv = pd.read_excel(inverse_xlsx, sheet_name=inverse_sheet)
    ret = pd.read_excel(retrieval_xlsx, sheet_name=retrieval_sheet)

    # 3) Inverse Cooking: average metrics by prefix (image)
    inv_required = ["prefix", "IoU", "F1", "ROUGE-L", "SacreBLEU"]
    missing_inv = [c for c in inv_required if c not in inv.columns]
    if missing_inv:
        raise ValueError(f"Inverse sheet missing columns: {missing_inv}")

    inv_grouped = (
        inv.groupby("prefix", as_index=False)[["IoU", "F1", "ROUGE-L", "SacreBLEU"]]
        .mean()
        .rename(
            columns={
                "prefix": "image_id",
                "IoU": "IoU_ic",
                "F1": "F1_ic",
                "ROUGE-L": "ROUGE-L_ic",
                "SacreBLEU": "SacreBLEU_ic",
            }
        )
    )

    # 4) Retrieval: metrics by pair_id
    ret_required = ["pair_id", "IoU", "F1", "ROUGE-L", "SacreBLEU"]
    missing_ret = [c for c in ret_required if c not in ret.columns]
    if missing_ret:
        raise ValueError(f"Retrieval sheet missing columns: {missing_ret}")

    ret_clean = ret.rename(
        columns={
            "IoU": "IoU_ret",
            "F1": "F1_ret",
            "ROUGE-L": "ROUGE-L_ret",
            "SacreBLEU": "SacreBLEU_ret",
        }
    )

    # Average duplicates safely
    ret_clean = (
        ret_clean.groupby("pair_id", as_index=False)[
            ["IoU_ret", "F1_ret", "ROUGE-L_ret", "SacreBLEU_ret"]
        ]
        .mean()
    )

    # 5) Merge: mapping -> inverse -> retrieval
    out = map_df.merge(inv_grouped, on="image_id", how="left")
    out = out.merge(ret_clean, on="pair_id", how="left")

    # 6) Convert IoU/F1/ROUGE-L to percentage (assumes they are in [0,1])
    percent_cols = [
        "IoU_ic", "F1_ic", "ROUGE-L_ic",
        "IoU_ret", "F1_ret", "ROUGE-L_ret",
    ]
    for c in percent_cols:
        if c in out.columns:
            out[c] = out[c] * 100

    # 7) RET buckets (use strict thresholds as requested)
    f1 = out["F1_ret"].fillna(-1)
    out["RET_FAIL"] = (f1 < 30).astype(int)
    out["RET_MEDIUM"] = ((f1 >= 30) & (f1 <= 70)).astype(int)
    out["RET_SUCCESS"] = (f1 > 70).astype(int)

    # Optional: keep ordering by mapping number
    if "number" in out.columns:
        out = out.sort_values("number", ascending=True)

    # Final column order
    final_cols = [
        "image_id",
        "pair_id",
        "IoU_ic",
        "F1_ic",
        "ROUGE-L_ic",
        "SacreBLEU_ic",
        "IoU_ret",
        "F1_ret",
        "ROUGE-L_ret",
        "SacreBLEU_ret",
        "RET_SUCCESS",
        "RET_FAIL",
        "RET_MEDIUM",
    ]
    if "number" in out.columns:
        final_cols = ["number"] + final_cols

    final_cols = [c for c in final_cols if c in out.columns]
    out = out[final_cols]

    # 8) Write Excel
    out.to_excel(output_xlsx, index=False)
    print(f"✅ Wrote {len(out)} rows to {output_xlsx}")


if __name__ == "__main__":
    build_compare_table(
        inverse_xlsx="Inverse/Metric_IC.xlsx",
        retrieval_xlsx="Retrieval/Metric_RET.xlsx",
        mapping_json="filtered_sorted_idx_numbered.json",
        output_xlsx="Compare_IC_vs_Retrieval.xlsx",
        inverse_sheet="per_image_avg",
        retrieval_sheet="Sheet1",
    )
