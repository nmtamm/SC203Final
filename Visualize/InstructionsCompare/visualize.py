import json
import os
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.graphics.shapes import Drawing, Rect, String

from rouge_score import rouge_scorer  # <-- Add this import

# ═══════════════════════════════════════════════════════════════════
#  Colours
# ═══════════════════════════════════════════════════════════════════
C_TITLE = HexColor("#37474F")
C_HDR = HexColor("#263238")
C_GT_L = HexColor("#C5CAE9")
C_GT = HexColor("#F0F1FA")
C_GEN_L = HexColor("#A5D6A7")
C_GEN = HexColor("#F0FAF0")
C_RET_L = HexColor("#FFCC80")
C_RET = HexColor("#FFFAF0")
C_IMG_BG = HexColor("#ECEFF1")
C_GRID = HexColor("#CFD8DC")
C_BOX = HexColor("#78909C")
HL = "#FFD54F"
A_GT, A_GEN, A_RET = "#283593", "#2E7D32", "#BF360C"


# ═══════════════════════════════════════════════════════════════════
#  Image helper  (load real image or draw grey placeholder)
# ═══════════════════════════════════════════════════════════════════
def get_img(path, max_w, max_h):
    if os.path.isfile(path):
        try:
            from PIL import Image as P

            im = P.open(path)
            iw, ih = im.size
            r = min(max_w / iw, max_h / ih)
            return RLImage(path, width=iw * r, height=ih * r)
        except Exception:
            pass
    d = Drawing(int(max_w), int(max_h))
    d.add(
        Rect(
            0,
            0,
            int(max_w),
            int(max_h),
            fillColor=C_IMG_BG,
            strokeColor=HexColor("#B0BEC5"),
            strokeWidth=0.5,
        )
    )
    d.add(
        String(
            max_w / 2,
            max_h / 2 - 3,
            "Image not found",
            fontSize=7,
            fillColor=HexColor("#90A4AE"),
            textAnchor="middle",
        )
    )
    return d


# ═══════════════════════════════════════════════════════════════════
#  Paragraph styles
# ═══════════════════════════════════════════════════════════════════
ST = dict(
    t=ParagraphStyle(
        "t",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=white,
        alignment=TA_CENTER,
        leading=15,
    ),
    h=ParagraphStyle(
        "h",
        fontName="Helvetica-Bold",
        fontSize=38,
        textColor=white,
        alignment=TA_CENTER,
        leading=48,
    ),
    l=ParagraphStyle(
        "l", fontName="Helvetica-Bold", fontSize=35, alignment=TA_CENTER, leading=45
    ),
    x=ParagraphStyle(
        "x",
        fontName="Helvetica",
        fontSize=34,
        textColor=HexColor("#333"),
        alignment=TA_LEFT,
        leading=44,
        leftIndent=1.5 * mm,
        rightIndent=1 * mm,
    ),
)


def lcs_positions_rouge(gt, pred):
    """
    Returns the set of indices in pred that are part of the LCS with gt,
    matching ROUGE-L's logic.
    """
    # Tokenize as ROUGE-L does (split on whitespace)
    gt_tokens = " ".join(gt).split() if isinstance(gt, list) else gt.split()
    pred_tokens = " ".join(pred).split() if isinstance(pred, list) else pred.split()
    m, n = len(gt_tokens), len(pred_tokens)
    # Build LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if gt_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    # Backtrack to find LCS indices in pred
    lcs_pred_indices = set()
    i, j = m, n
    while i > 0 and j > 0:
        if gt_tokens[i - 1] == pred_tokens[j - 1]:
            lcs_pred_indices.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs_pred_indices


def fmt_rouge(steps, matched):
    out, off = [], 0
    for i, s in enumerate(steps, 1):
        parts = []
        for j, w in enumerate(s.split()):
            ew = escape(w)
            if (off + j) in matched:
                parts.append(f'<font backColor="{HL}" color="#333"><b>{ew}</b></font>')
            else:
                parts.append(ew)
        out.append(f'<font color="#999">{i}.</font> {" ".join(parts)}')
        off += len(s.split())
    return "<br/>".join(out)


def split_into_two_columns(text, max_width=40):
    """
    Splits a long text into two subcolumns (lists of lines), balancing the content.
    Returns a tuple: (left_lines, right_lines)
    """
    words = text.split()
    half = len(words) // 2
    # Try to split at a step boundary if possible
    left = []
    right = []
    left_len = 0
    for i, w in enumerate(words):
        if left_len < half:
            left.append(w)
            left_len += 1
        else:
            right.append(w)
    return " ".join(left), " ".join(right)


def fmt_rouge_two_columns(steps, matched, max_chars_per_line=60):
    """
    Splits steps into two subcolumns, keeping each step intact,
    and finds the split that balances the estimated number of rendered lines.
    """

    def estimate_lines(step):
        return max(1, (len(step) // max_chars_per_line) + 1)

    step_lines = [estimate_lines(s) for s in steps]
    total_steps = len(steps)
    total_lines = sum(step_lines)

    # Find the split index that minimizes the difference in lines
    best_split = 1
    min_diff = float("inf")
    for split in range(1, total_steps):
        left_lines = sum(step_lines[:split])
        right_lines = sum(step_lines[split:])
        diff = abs(left_lines - right_lines)
        if diff < min_diff:
            min_diff = diff
            best_split = split

    # Assign steps to columns
    left, right = [], []
    off = 0
    step_offsets = []
    for s in steps:
        step_offsets.append(off)
        off += len(s.split())

    for idx in range(best_split):
        left.append((idx, steps[idx], step_offsets[idx]))
    for idx in range(best_split, total_steps):
        right.append((idx, steps[idx], step_offsets[idx]))

    def format_column(col_steps):
        out = []
        for idx, s, off in col_steps:
            parts = []
            for j, w in enumerate(s.split()):
                ew = escape(w)
                if (off + j) in matched:
                    parts.append(
                        f'<font backColor="{HL}" color="#333"><b>{ew}</b></font>'
                    )
                else:
                    parts.append(ew)
            out.append(f'<font color="#999">{idx+1}.</font> {" ".join(parts)}')
        return "<br/>".join(out)

    left_text = format_column(left)
    right_text = format_column(right)
    t = Table(
        [[Paragraph(left_text, ST["x"]), Paragraph(right_text, ST["x"])]], colWidths="*"
    )
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                # You can add more styles if needed
            ]
        )
    )
    return t


def build(data, out="rouge_l_visualization.pdf"):

    pw = 1920
    margin = 0
    usable = pw
    gap = 0
    cw = usable / 2
    img_h = 96 * mm  # max image height

    # ── per-column cells ──
    hr, ir = [], []  # headers, images
    gl, gt = [], []  # GT label, GT text
    ql, qt = [], []  # Gen label, Gen text
    rl, rt = [], []  # Ret label, Ret text

    for e in data:
        gI = e["gt_instructions"]
        nI = e["generated_instructions"]
        rI = e["retrieved_instructions"]

        # Use ROUGE-L LCS for highlighting
        m_gen = lcs_positions_rouge(gI, nI)
        m_ret = lcs_positions_rouge(gI, rI)

        # header
        hr.append(
            Paragraph(
                f'<font size="16">#{e["number"]}</font> &nbsp;'
                f'{escape(e["pair id for retrieval"])} &nbsp;|&nbsp; '
                f'GT {escape(e["ground truth recipe id"])}',
                ST["h"],
            )
        )
        # image
        ir.append(get_img(e["image_path"], cw, img_h))
        # ground truth
        gl.append(Paragraph(f'<font color="{A_GT}">GROUND TRUTH</font>', ST["l"]))
        gt.append(fmt_rouge_two_columns(gI, set()))
        # generated
        ql.append(
            Paragraph(
                f'<font color="{A_GEN}">GENERATED</font>',
                ST["l"],
            )
        )
        qt.append(fmt_rouge_two_columns(nI, m_gen))
        # retrieved
        rl.append(
            Paragraph(
                f'<font color="{A_RET}">RETRIEVED</font>',
                ST["l"],
            )
        )
        rt.append(fmt_rouge_two_columns(rI, m_ret))

    # ── assemble table (remove title and sample headers) ──
    rows = [
        ir,  # images
        gl,
        gt,  # GT label   4  GT text
        ql,
        qt,  # 5  Gen label  6  Gen text
        rl,
        rt,  # 7  Ret label  8  Ret text
    ]

    tbl = Table(rows, colWidths=[cw] * 2)
    tbl.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("GRID", (0, 0), (-1, -1), 0.4, C_GRID),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0, white),
                ("BACKGROUND", (0, 0), (-1, 0), C_IMG_BG),
                ("BACKGROUND", (0, 1), (-1, 1), C_GT_L),
                ("BACKGROUND", (0, 2), (-1, 2), C_GT),
                ("BACKGROUND", (0, 3), (-1, 3), C_GEN_L),
                ("BACKGROUND", (0, 4), (-1, 4), C_GEN),
                ("BACKGROUND", (0, 5), (-1, 5), C_RET_L),
                ("BACKGROUND", (0, 6), (-1, 6), C_RET),
            ]
        )
    )

    # Measure table size
    w, h = tbl.wrap(0, 0)

    # Create document with dynamic height
    doc = SimpleDocTemplate(
        out,
        pagesize=(pw, h * 1.025),
        leftMargin=0,
        rightMargin=0,
        topMargin=0,
        bottomMargin=0,
    )

    doc.build([tbl])
    print(f"✓  PDF → {out}")


# with open("data/f1_high_rougel_high.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
# build(data, "results/f1_high_rougel_high.pdf")

# Traverse all JSON files in the directory and generate PDFs
input_dir = "data"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            print(f"⚠️  Skipping {filename} (empty data)")
            continue
        output_path = os.path.join(output_dir, filename.replace(".json", ".pdf"))
        build(data, output_path)