import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import json
import os
import matplotlib
import textwrap
from matplotlib.font_manager import FontProperties


def draw_two_paragraphs(
    ax, first_text, second_text, fontsize=26, color1="blue", color2="red", margin_px=10
):
    # Estimate number of lines for each paragraph
    def count_lines(text):
        return text.count("\n") + 1 if text else 0

    n1 = count_lines(first_text)
    n2 = count_lines(second_text)

    # Line height in axes coordinates (approximate)
    bbox = ax.get_window_extent()
    axes_height_px = bbox.height
    line_height_px = fontsize * 1.5  # 1.2 is a line spacing factor
    total_height_px = (n1 + n2) * line_height_px
    # Center the block of text in the axes
    y_center = 0.5
    if n1 > 0 and n2 > 0:
        # If both exist, stack them with a small gap
        y1 = y_center + (n2 * line_height_px) / (2 * axes_height_px)
        y2 = y_center - (n1 * line_height_px) / (2 * axes_height_px)
    elif n1 > 0:
        y1 = y_center
        y2 = None
    elif n2 > 0:
        y1 = None
        y2 = y_center
    else:
        y1 = y2 = y_center

    if n1 > 0:
        ax.text(
            0.5,
            y1,
            first_text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color1,
            transform=ax.transAxes,
            wrap=True,
        )
    if n2 > 0:
        ax.text(
            0.5,
            y2,
            second_text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color2,
            transform=ax.transAxes,
            wrap=True,
        )


def draw_ingredients_paragraph(
    ax, ingredients, gt_ingredients, color="black", fontsize=26, margin_px=10
):
    if not ingredients:
        return
    text = " ".join(ingredients)

    # Estimate how many characters fit in the box width (minus margin)
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    box_width_px = bbox.width * 0.94  # box width is 94% of axes width
    usable_width_px = max(10, box_width_px - 2 * margin_px)

    avg_char_px = fontsize * 0.6
    max_chars_per_line = max(10, int(usable_width_px // avg_char_px))

    correct_ingredients = []
    wrong_ingredients = []

    for ingredient in ingredients:
        if ingredient in gt_ingredients:
            correct_ingredients.append(ingredient)
        else:
            wrong_ingredients.append(ingredient)

    correct_ingredients_text = " ".join(correct_ingredients)
    wrong_ingredients_text = " ".join(wrong_ingredients)

    correct_wrapped = "\n".join(
        textwrap.wrap(correct_ingredients_text, width=max_chars_per_line)
    )
    wrong_wrapped = "\n".join(
        textwrap.wrap(wrong_ingredients_text, width=max_chars_per_line)
    )

    wrapped_text = "\n".join(textwrap.wrap(text, width=max_chars_per_line))

    draw_two_paragraphs(ax, correct_wrapped, wrong_wrapped, fontsize=26)

    if not gt_ingredients:
        ax.text(
            0.5,
            0.5,
            wrapped_text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            transform=ax.transAxes,
            wrap=True,
        )


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
    columns = ["Image", "Inverse Cooking", "via Retrieval", "Ground Truth"]

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
            image_path = image_path.replace("\\", "/")
            print("Trying to load:", image_path, "Exists:", os.path.exists(image_path))
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax[row][0].imshow(img)
        else:
            ax[row][0].text(0.5, 0.5, "Image", ha="center", va="center", fontsize=12)

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

        # 2. Ours column
        ax[row][1].axis("off")
        draw_ingredients_paragraph(
            ax[row][1], ours_ingredients, gt_ingredients, color="black", fontsize=26
        )
        draw_column_box(ax[row][1])
        if row == 0:
            ax[row][1].set_title(columns[1], fontsize=28, fontweight="bold")

        # 3. Retrieval column
        ax[row][2].axis("off")
        draw_ingredients_paragraph(
            ax[row][2],
            retrieval_ingredients,
            gt_ingredients,
            color="black",
            fontsize=26,
        )
        draw_column_box(ax[row][2])
        if row == 0:
            ax[row][2].set_title(columns[2], fontsize=28, fontweight="bold")

        # 4. Ground Truth column
        ax[row][3].axis("off")
        empty = []
        draw_ingredients_paragraph(
            ax[row][3], gt_ingredients, empty, color="black", fontsize=26
        )
        draw_column_box(ax[row][3])
        if row == 0:
            ax[row][3].set_title(columns[3], fontsize=28, fontweight="bold")

    plt.tight_layout()
    plt.savefig("all_items_grid", dpi=plt.gcf().dpi)
    plt.show()


# ...existing code...
# Usage
JSON_path = "path to your combined JSON file including indices and ingredients list for ground truth, retrieval, and generative, with image ids and image paths"
with open(JSON_path, "r", encoding="utf-8") as f:
    data = json.load(f)

plot_ingredient_table_grid(data["results"])
