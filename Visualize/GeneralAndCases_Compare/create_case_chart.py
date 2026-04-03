import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
EXCEL_PATH = "Compare_IC_vs_Retrieval.xlsx"   # change if needed
SHEET_NAME = "Sheet1"               # or set your sheet name
OUT_DIR = "Charts"
os.makedirs(OUT_DIR, exist_ok=True)

# Colors (match your style)
color_ic  = "#637bef"  # Blue
color_ret = "#ef6363"  # Red

# Your column names
metric_map = {
    "IoU": ("IoU_ic", "IoU_ret", True),               # (ic_col, ret_col, scale_to_pct?)
    "F1": ("F1_ic", "F1_ret", True),
    "ROUGE-L": ("ROUGE-L_ic", "ROUGE-L_ret", True),
    "SacreBLEU": ("SacreBLEU_ic", "SacreBLEU_ret", False),
}

# 3 cases (each will become ONE image)
regimes = [
    ("RET_SUCCESS", "RET_SUCCESS"),
    ("RET_FAIL", "RET_FAIL"),
    ("RET_MEDIUM", "RET_MEDIUM"),
]

def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def maybe_scale_percent(df: pd.DataFrame, col: str):
    x = df[col].dropna()
    if len(x) == 0:
        return
    # if looks like 0..1, scale to 0..100
    if np.nanpercentile(x, 95) <= 1.2:
        df[col] = df[col] * 100

def plot_hist(ax, data_ic, data_ret, bins, title):
    counts_ic, _  = np.histogram(data_ic, bins=bins)
    counts_ret, _ = np.histogram(data_ret, bins=bins)

    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    ax.bar(centers - width/2, counts_ic,  width=width, color=color_ic,  alpha=0.8, label="Inverse Cooking")
    ax.bar(centers + width/2, counts_ret, width=width, color=color_ret, alpha=0.8, label="Retrieval")

    ax.set_title(title)
    ax.set_xlabel("F1 (%)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.legend()

# Load single sheet (has F1_ic, F1_ret, RET_SUCCESS/FAIL/MEDIUM)
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Numeric conversions
df["F1_ic"]  = safe_to_numeric(df["F1_ic"])
df["F1_ret"] = safe_to_numeric(df["F1_ret"])

# Convert F1 to percentage if needed
maybe_scale_percent(df, "F1_ic")
maybe_scale_percent(df, "F1_ret")

# Regime flags
for _, flag_col in regimes:
    df[flag_col] = safe_to_numeric(df[flag_col]).fillna(0).astype(int)

# Create ONE image with 3 subplots (SUCCESS / FAIL / MEDIUM), F1 only
fig = plt.figure(figsize=(18, 5))

for i, (regime_name, flag_col) in enumerate(regimes, start=1):
    ax = fig.add_subplot(1, 3, i)

    df_r = df[df[flag_col] == 1]
    if len(df_r) == 0:
        ax.set_title(f"F1 Comparison ({regime_name}) - no data")
        ax.axis("off")
        continue

    data_ic  = df_r["F1_ic"].dropna()
    data_ret = df_r["F1_ret"].dropna()

    if len(data_ic) == 0 or len(data_ret) == 0:
        ax.set_title(f"F1 Comparison ({regime_name}) - no data")
        ax.axis("off")
        continue

    bins = np.linspace(0, 100, 50)
    plot_hist(ax, data_ic, data_ret, bins=bins, title=regime_name)

# Leave space for note under the figure
fig.tight_layout(rect=[0, 0.08, 1, 1])

out_path = os.path.join(OUT_DIR, "F1_3cases_comparison.png")
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")