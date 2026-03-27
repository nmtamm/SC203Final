import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import json
import os
import matplotlib


def split_ingredients_if_short(
    ingredients, ax, indices=None, gt_indices=None, fontsize=26, threshold=0.45
):
    if not ingredients:
        return False  # Nothing to draw

    renderer = plt.gcf().canvas.get_renderer()
    widths = []
    for word in ingredients:
        text = ax.text(0.5, 0.5, word, fontsize=fontsize, ha="center", va="center")
        plt.draw()
        bbox = text.get_window_extent(renderer=renderer)
        ax_bbox = ax.get_window_extent(renderer=renderer)
        text_width_frac = bbox.width / ax_bbox.width
        widths.append(text_width_frac)
        text.remove()

    group2_indices = [i for i, w in enumerate(widths) if w <= threshold]
    group1_indices = [i for i, w in enumerate(widths) if w > threshold]

    group2 = [ingredients[i] for i in group2_indices]
    group2_idx = [indices[i] if indices else None for i in group2_indices]
    group1 = [ingredients[i] for i in group1_indices]
    group1_idx = [indices[i] if indices else None for i in group1_indices]

    if not group2 and not group1:
        return False

    # Calculate total number of rows needed
    n_rows_2col = (len(group2) + 1) // 2
    n_rows_1col = len(group1)
    total_rows = max(n_rows_2col, (len(group2) // 2)) + n_rows_1col
    total_rows = (
        max(n_rows_2col, len(group1))
        if group2 and not group1
        else n_rows_2col + n_rows_1col
    )
    if group2 and group1:
        total_rows = n_rows_2col + n_rows_1col
    elif group2:
        total_rows = n_rows_2col
    elif group1:
        total_rows = n_rows_1col

    step = 1.0 / (total_rows + 1)
    y = 1.0 - step / 2

    # Draw two-column group first
    if group2:
        mid = (len(group2) + 1) // 2
        col1 = group2[:mid]
        col2 = group2[mid:]
        idx1 = group2_idx[:mid]
        idx2 = group2_idx[mid:]
        y1 = y
        y2 = y
        for i in range(max(len(col1), len(col2))):
            if i < len(col1):
                word = col1[i]
                idx = idx1[i]
                color = (
                    "blue"
                    if gt_indices and idx in gt_indices
                    else "red" if idx is not None else "black"
                )
                ax.text(
                    0.25,
                    y1,
                    word,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    color=color,
                    transform=ax.transAxes,
                )
            if i < len(col2):
                word = col2[i]
                idx = idx2[i]
                color = (
                    "blue"
                    if gt_indices and idx in gt_indices
                    else "red" if idx is not None else "black"
                )
                ax.text(
                    0.75,
                    y2,
                    word,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    color=color,
                    transform=ax.transAxes,
                )
            y1 -= step
            y2 -= step
        y = min(y1, y2)

    # Draw one-column group below
    if group1:
        y1 = y
        for word, idx in zip(group1, group1_idx):
            color = (
                "blue"
                if gt_indices and idx in gt_indices
                else "red" if idx is not None else "black"
            )
            ax.text(
                0.5,
                y1,
                word,
                ha="center",
                va="top",
                fontsize=fontsize,
                color=color,
                transform=ax.transAxes,
            )
            y1 -= step

    return True


def plot_ingredient_table_grid(
    items, n_rows=None, image_list=None  # Optional: list of images, or None
):
    if n_rows is None:
        n_rows = len(items)
    max_items_per_row = [
        max(
            len(item.get("gt_ingredients", [])),
            len(item.get("pred_ingredients", [])),
            1,
        )
        for item in items
    ]
    total_height = sum(
        [max(4, n * 0.5) for n in max_items_per_row]
    )  # 0.5 per ingredient, min 4 per row

    fig, ax = plt.subplots(n_rows, 4, figsize=(20, total_height))
    columns = ["Image", "Inverse Cooking", "Retrieved", "Ground Truth"]

    if n_rows == 1:
        ax = [ax]  # Ensure ax is always a list of rows

    for row, item in enumerate(items):
        gt_ingredients = item.get("gt_ingredients", [])
        gt_indices = item.get("gt_indices", [])
        retrieval_ingredients = item.get("pred_ingredients", [])
        retrieval_indices = item.get("pred_indices", [])
        ours_ingredients = item.get("gen_ingredients_names", [])
        ours_indices = item.get("gen_indices", [])

        # 1. Image column
        ax[row][0].axis("off")
        image_path = item.get("image_path")
        if image_path:
            image_path = image_path.replace(
                "\\", "/"
            )  # Replace backslashes with forward slashes
            print("Trying to load:", image_path, "Exists:", os.path.exists(image_path))
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax[row][0].imshow(img)
        else:
            ax[row][0].text(0.5, 0.5, "Image", ha="center", va="center", fontsize=12)

        # Helper to draw a single box around all text in a column
        def draw_column_box(axx):
            rect = Rectangle(
                (0.03, 0.01),
                0.94,
                0.98,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
                transform=axx.transAxes,
                zorder=0,
            )
            axx.add_patch(rect)

        # Calculate the max number of items in this row (across all three columns)
        max_items = max(
            len(ours_ingredients), len(retrieval_ingredients), len(gt_ingredients), 1
        )
        step = 1.0 / (max_items + 1)
        y_start = 1.0 - step / 2

        # 2. Ours column
        ax[row][1].axis("off")
        # Try to split into two columns if possible
        if not split_ingredients_if_short(
            ours_ingredients,
            ax[row][1],
            indices=ours_indices,
            gt_indices=gt_indices,
            fontsize=26,
            threshold=0.45,
        ):
            ours_step = (
                1.0 / (len(ours_ingredients) + 1) if len(ours_ingredients) > 0 else 1.0
            )
            ours_y = 1.0 - ours_step / 2
            for word, idx in zip(ours_ingredients, ours_indices):
                color = "blue" if idx in gt_indices else "red"
                ax[row][1].text(
                    0.5,
                    ours_y,
                    word,
                    ha="center",
                    va="top",
                    fontsize=26,
                    color=color,
                    transform=ax[row][1].transAxes,
                )
                ours_y -= ours_step
        draw_column_box(ax[row][1])
        if row == 0:
            ax[row][1].set_title(columns[1], fontsize=28, fontweight="bold")

        # 3. Retrieval column
        ax[row][2].axis("off")
        if not split_ingredients_if_short(
            retrieval_ingredients,
            ax[row][2],
            indices=retrieval_indices,
            gt_indices=gt_indices,
            fontsize=26,
            threshold=0.45,
        ):
            retrieval_step = (
                1.0 / (len(retrieval_ingredients) + 1)
                if len(retrieval_ingredients) > 0
                else 1.0
            )
            retrieval_y = 1.0 - retrieval_step / 2
            for word, idx in zip(retrieval_ingredients, retrieval_indices):
                color = "blue" if idx in gt_indices else "red"
                ax[row][2].text(
                    0.5,
                    retrieval_y,
                    word,
                    ha="center",
                    va="top",
                    fontsize=26,
                    color=color,
                    transform=ax[row][2].transAxes,
                )
                retrieval_y -= retrieval_step
        draw_column_box(ax[row][2])
        if row == 0:
            ax[row][2].set_title(columns[2], fontsize=28, fontweight="bold")

        # 4. Ground Truth column
        ax[row][3].axis("off")
        if not split_ingredients_if_short(
            gt_ingredients, ax[row][3], fontsize=26, threshold=0.45
        ):
            gt_step = (
                1.0 / (len(gt_ingredients) + 1) if len(gt_ingredients) > 0 else 1.0
            )
            gt_y = 1.0 - gt_step / 2
            for word in gt_ingredients:
                ax[row][3].text(
                    0.5,
                    gt_y,
                    word,
                    ha="center",
                    va="top",
                    fontsize=26,
                    color="black",
                    transform=ax[row][3].transAxes,
                )
                gt_y -= gt_step
        draw_column_box(ax[row][3])
        if row == 0:
            ax[row][3].set_title(columns[3], fontsize=28, fontweight="bold")

    plt.tight_layout()
    plt.savefig("all_items_grid.pdf", dpi=200)
    plt.show()


# Usage
JSON_path = "path to your combined JSON file including indices and ingredients list for ground truth, retrieval, and generative, with image ids and image paths"
with open(JSON_path, "r", encoding="utf-8") as f:
    data = json.load(f)

plot_ingredient_table_grid(data["results"])
