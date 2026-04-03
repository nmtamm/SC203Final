import os
import json

# Folder containing your JSON files
json_folder = "data"
data_folder = "your path to the data folder containing images"


def find_image_path(image_id, root_folder):
    for root, dirs, files in os.walk(root_folder):
        if image_id in files:
            return os.path.join(root, image_id)
    return None


for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue
    file_path = os.path.join(json_folder, json_file)
    with open(file_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)

    for item in current_data:
        image_id = item.get("image id")
        if image_id:
            image_id_search = image_id[5:] if image_id.startswith("Image") else image_id
            img_path = find_image_path(image_id_search, data_folder)
            item["image_path"] = img_path if img_path else None

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, ensure_ascii=False)
