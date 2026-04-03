import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
EXCEL_PATH = "Compare_IC_vs_Retrieval.xlsx"  # change if needed
SHEET_NAME = "Sheet1"
OUT_DIR = "Charts"
os.makedirs(OUT_DIR, exist_ok=True)

color_ic = "#637bef"
color_ret = "#ef6363"


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def maybe_scale_percent(df: pd.DataFrame, col: str):
    x = df[col].dropna()
    if len(x) == 0:
        return
    if np.nanpercentile(x, 95) <= 1.2:
        df[col] = df[col] * 100


def plot_hist(ax, data_ic, data_ret, bins, title):
    counts_ic, _ = np.histogram(data_ic, bins=bins)
    counts_ret, _ = np.histogram(data_ret, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4
    ax.bar(
        centers - width / 2,
        counts_ic,
        width=width,
        color=color_ic,
        alpha=0.8,
        label="Inverse Cooking",
    )
    ax.bar(
        centers + width / 2,
        counts_ret,
        width=width,
        color=color_ret,
        alpha=0.8,
        label="Retrieval",
    )
    ax.set_title(title)
    ax.set_xlabel("F1 (%)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.legend()


# Load data
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df["F1_ic"] = safe_to_numeric(df["F1_ic"])
df["F1_ret"] = safe_to_numeric(df["F1_ret"])
df["ROUGE-L_ic"] = safe_to_numeric(df["ROUGE-L_ic"])
df["ROUGE-L_ret"] = safe_to_numeric(df["ROUGE-L_ret"])

maybe_scale_percent(df, "F1_ic")
maybe_scale_percent(df, "F1_ret")
maybe_scale_percent(df, "ROUGE-L_ic")
maybe_scale_percent(df, "ROUGE-L_ret")

# Define cases
cases = [
    ("F1 High & ROUGE-L Low", lambda f1, rouge: (f1 >= 90) & (rouge <= 50)),
    ("F1 High & ROUGE-L High", lambda f1, rouge: (f1 >= 90) & (rouge > 50)),
    ("F1 Low & ROUGE-L Low", lambda f1, rouge: (f1 <= 10) & (rouge <= 50)),
    ("F1 Low & ROUGE-L High", lambda f1, rouge: (f1 <= 10) & (rouge > 50)),
]

fig = plt.figure(figsize=(20, 5))

for i, (case_name, case_fn) in enumerate(cases, start=1):
    ax = fig.add_subplot(1, 4, i)
    mask_ic = case_fn(df["F1_ic"], df["ROUGE-L_ic"])
    mask_ret = case_fn(df["F1_ret"], df["ROUGE-L_ret"])
    data_ic = df.loc[mask_ic, "F1_ic"].dropna()
    data_ret = df.loc[mask_ret, "F1_ret"].dropna()
    if len(data_ic) == 0 and len(data_ret) == 0:
        ax.set_title(f"{case_name}\n(no data)")
        ax.axis("off")
        continue
    bins = np.linspace(0, 100, 50)
    plot_hist(ax, data_ic, data_ret, bins=bins, title=case_name)

fig.tight_layout(rect=[0, 0.08, 1, 1])
out_path = os.path.join(OUT_DIR, "F1_ROUGE_cases_comparison.pdf")
plt.savefig(out_path, dpi=200)
plt.show()
print(f"Saved: {out_path}")
