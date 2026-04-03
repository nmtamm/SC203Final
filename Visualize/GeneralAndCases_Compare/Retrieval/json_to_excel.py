import os
import re
import json
from glob import glob
from typing import Any, Dict, List, Optional

import pandas as pd


PAIR_PREFIX_RE = re.compile(r"^Pair(\d+)$", re.IGNORECASE)
FILENAME_RE = re.compile(r"Pair(\d+)To(\d+)\.json$", re.IGNORECASE)


def _to_float(x: Any) -> Optional[float]:
    """Convert to float if possible, else None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def build_pair_level_excel(metric_folder: str, output_xlsx: str) -> None:
    """
    For each file named PairXToY.json, read JSON and export one row per entry in `results`.
    Columns: pair_id, IoU, F1, ROUGE-L, SacreBLEU
    Sorting: ascending by numeric X in pair_id (PairX).
    """
    rows: List[Dict[str, Any]] = []

    json_files = glob(os.path.join(metric_folder, "Pair*To*.json"))

    for path in json_files:
        fname = os.path.basename(path)
        if not FILENAME_RE.search(fname):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        if not isinstance(results, list):
            continue

        for item in results:
            if not isinstance(item, dict):
                continue

            pair_id = item.get("prefix")  # expected like "Pair123"
            if not isinstance(pair_id, str):
                continue

            m = PAIR_PREFIX_RE.match(pair_id.strip())
            if not m:
                # If prefix isn't exactly PairX, skip (or you can keep it)
                continue

            x = int(m.group(1))

            row = {
                "pair_id": f"Pair{x}",
                "_x": x,  # helper sort key
                "IoU": _to_float(item.get("iou")),
                "F1": _to_float(item.get("f1")),
                "ROUGE-L": _to_float(item.get("rougeL")),
                "SacreBLEU": _to_float(item.get("sacrebleu")),
            }
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No Pair-level results found. Check folder and JSON structure: {metric_folder}"
        )

    df = pd.DataFrame(rows)

    # If the same PairX appears multiple times across files, keep the first occurrence.
    # If you prefer averaging duplicates, tell me and I’ll modify it.
    df = df.sort_values(by=["_x"], ascending=True).drop_duplicates(subset=["pair_id"], keep="first")

    df = df.drop(columns=["_x"])

    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    df.to_excel(output_xlsx, index=False)

    print(f"✅ Wrote {len(df)} Pair rows to: {output_xlsx}")


if __name__ == "__main__":
    METRIC_FOLDER = "Retrieval/Metric"  # change to your real path if needed
    OUTPUT_XLSX = "Retrieval/Metric_RET.xlsx"
    build_pair_level_excel(METRIC_FOLDER, OUTPUT_XLSX)
