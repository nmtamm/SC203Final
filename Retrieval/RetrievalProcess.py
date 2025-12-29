import json
import shutil
import os
import math

"""
This divide the total retrieval pairs into smaller batches and saves them into separate folders.
You can adjust the batch_size variable as needed.
"""

data_path = (
    "path to JSON files contanining retrieval pairs extracted using recall@1 metric"
)
with open(
    data_path,
    "r",
    encoding="utf-8",
) as jf:
    pairs = json.load(jf)

dictionary_dir = "path to JSON files contanining recipe information"
output_dir = "path to output directory for saving the JSON files for retrieval pairs"
batch_size = 571

not_found_log = os.path.join(output_dir, "not_found_pairs.txt")

with open(not_found_log, "w", encoding="utf-8") as logf:
    total_pairs = len(pairs)
    num_batches = math.ceil(total_pairs / batch_size)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_pairs)
        batch_pairs = pairs[start_idx:end_idx]
        batch_folder = os.path.join(output_dir, f"Pair{start_idx+1}To{end_idx}")
        os.makedirs(batch_folder, exist_ok=True)
        for pair_idx, (pred_id, gt_id) in enumerate(batch_pairs, start_idx + 1):
            gt_json_path = f"{dictionary_dir}{gt_id}.json"
            pred_json_path = f"{dictionary_dir}{pred_id}.json"
            output_gt_path = os.path.join(
                batch_folder, f"Pair{pair_idx}_GroundTruth.json"
            )
            output_pred_path = os.path.join(
                batch_folder, f"Pair{pair_idx}_Predicted.json"
            )

            gt_exists = os.path.isfile(gt_json_path)
            pred_exists = os.path.isfile(pred_json_path)

            if gt_exists and pred_exists:
                print(
                    f"Processing Batch {batch_idx+1} Pair {pair_idx}: GT ID = {gt_id}, Pred ID = {pred_id}"
                )
                shutil.copyfile(gt_json_path, output_gt_path)
                shutil.copyfile(pred_json_path, output_pred_path)
            else:
                print(
                    f"Skipping Batch {batch_idx+1} Pair {pair_idx}: GT ID = {gt_id}, Pred ID = {pred_id} (file not found)"
                )
                logf.write(f"{pair_idx}\t{pred_id}\t{gt_id}\n")
