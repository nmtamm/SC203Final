import json

pair = [2119, 2682, 4459, 5092, 8905, 9978, 10418, 12894, 28628, 31164, 32356, 34520]


def filter_retrieval_pairs(pair):
    # This function used to find the ground truth and retrieved recipes based on the given pair ID list
    input_path = "../Retrieval/retrieval_pairs.json"
    output_path = "results/selected_retrieval_pairs_for_instructions.json"

    # Load pairs from JSON file
    with open(input_path, "r") as f:
        all_pairs = json.load(f)

    # Convert to 0-based indices
    zero_based_indices = [i - 1 for i in pair]

    # Extract the specified pairs
    selected_pairs = [all_pairs[i] for i in zero_based_indices]

    # Print or save the selected pairs
    print(selected_pairs)

    # Save to a new JSON file
    with open(output_path, "w") as f:
        json.dump(selected_pairs, f, indent=4)


def add_detail_to_filtered_pairs():
    detailed_path = "data/mapping_ic_retrieval_groundtruth.json"
    output_path = "results/selected_detailed_pairs_for_instructions.json"

    # Load the JSON file with detailed items
    with open(detailed_path, "r") as f:
        all_items = json.load(f)

    # Build the set of target pair ids
    target_pair_ids = {f"Pair{ipair}" for ipair in pair}

    # Filter items with matching "pair id for retrieval"
    filtered_items = [
        item
        for item in all_items
        if item.get("pair id for retrieval") in target_pair_ids
    ]

    # Save to a new JSON file
    with open(output_path, "w") as f:
        json.dump(filtered_items, f, indent=2)

    print(f"Found {len(filtered_items)} matching items.")


# Usage
filter_retrieval_pairs(pair)
add_detail_to_filtered_pairs()
