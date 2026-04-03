import json

"""
This file is used to map the images that Inverse Cooking used, the pair IDs for retrieval and ground truth recipe IDs.
The output JSON file will contain the following fields:
- "number": the new order of the pairs after filtering out the not found ones
- "pair id for retrieval": the original pair ID in the format "Pair{original_index}"
- "ground truth recipe id": the ground truth recipe ID corresponding to the pair
- "image id": the image ID corresponding to the pair (with "Image" prefix and .jpg extension)
"""

not_found_path = "data/retrieval_pair_not_found_idxs.txt"
image_path = "data/filtered_image_names_inverse.txt"
output_path = "results/mapping_retrieval_pairs_with_images.json"
retrieval_ground_truth_recipe_id_path = "../Retrieval/retrieval_pairs.json"

# Read not found indices (1-based)
not_found_pairs_indices = set()
with open(not_found_path, "r") as f:
    for line in f:
        idx = line.strip()
        if idx.isdigit():
            not_found_pairs_indices.add(int(idx))

# Read sorted indices
sorted_indices = []
with open(retrieval_ground_truth_recipe_id_path, "r") as f:
    for line in f:
        idx = line.strip()
        sorted_indices.append(idx)

# Read image names (keep .jpg and add 'Image' prefix)
image_names = []
with open(image_path, "r") as f:
    for line in f:
        name = line.strip().replace('"', "")
        if name:
            if not name.startswith("Image"):
                name = "Image" + name
            image_names.append(name)

# Prepare output: only keep indices NOT in not_found_pairs_indices (convert to 1-based for comparison)
output_list = []
img_idx = 0
for i, idx in enumerate(sorted_indices):
    original_index = i + 1  # 1-based original index
    if original_index not in not_found_pairs_indices:
        image_id = image_names[img_idx] if img_idx < len(image_names) else None
        output_list.append(
            {
                "number": len(output_list) + 1,  # new order
                "pair id for retrieval": f"Pair{original_index}",
                "ground truth recipe id": idx,
                "image id": image_id,
            }
        )
        img_idx += 1

# Write to JSON file
with open(output_path, "w") as f:
    json.dump(output_list, f, indent=2)
