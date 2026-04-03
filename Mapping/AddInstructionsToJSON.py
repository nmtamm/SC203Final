import os
import json


def add_retrieved_instructions(data_folder, retrieval_folder):
    # Build a mapping from prefix to (gt_instructions, predicted_instructions)
    retrieval_map = {}

    for ret_file in os.listdir(retrieval_folder):
        if not ret_file.endswith(".json"):
            continue
        ret_path = os.path.join(retrieval_folder, ret_file)
        with open(ret_path, "r", encoding="utf-8") as f:
            ret_json = json.load(f)
        for result in ret_json.get("results", []):
            prefix = result.get("prefix")
            if prefix:
                retrieval_map[prefix] = (
                    result.get("ground_truth_instructions", []),
                    result.get("predicted_instructions", []),
                )

    # Now update objects in data_folder
    for data_file in os.listdir(data_folder):
        if not data_file.endswith(".json"):
            continue
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, "r", encoding="utf-8") as f:
            objects = json.load(f)

        for obj in objects:
            pair_id = obj.get("pair id for retrieval")
            if pair_id and pair_id in retrieval_map:
                gt_instr, pred_instr = retrieval_map[pair_id]
                obj["gt_instructions"] = gt_instr
                obj["retrieved_instructions"] = pred_instr

        # Save updated objects to output folder (overwrite)
        with open(data_path, "w", encoding="utf-8") as out_f:
            json.dump(objects, out_f, indent=2, ensure_ascii=False)


def add_generative_instructions(data_folder, generative_folder):
    # Build a mapping from image_id (prefix) to predicted_instructions
    generative_map = {}

    for gen_file in os.listdir(generative_folder):
        if not gen_file.endswith(".json"):
            continue
        gen_path = os.path.join(generative_folder, gen_file)
        with open(gen_path, "r", encoding="utf-8") as f:
            gen_json = json.load(f)
        for result in gen_json.get("results", []):
            prefix = result.get("prefix")
            if prefix:
                generative_map[prefix] = result.get("predicted_instructions", [])

    # Now update objects in data_folder
    for data_file in os.listdir(data_folder):
        if not data_file.endswith(".json"):
            continue
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, "r", encoding="utf-8") as f:
            objects = json.load(f)

        for obj in objects:
            image_id = obj.get("image id")
            if image_id and image_id in generative_map:
                obj["generated_instructions"] = generative_map[image_id]

        # Save updated objects
        with open(data_path, "w", encoding="utf-8") as out_f:
            json.dump(objects, out_f, indent=2, ensure_ascii=False)


data_dir = "../Visualize/InstructionsCompare/data"
if not os.path.exists(data_dir):
    print(f"Data directory '{data_dir}' does not exist. Please check the path.")
retrieval_folder = "D:/Revamping/output2"
generative_folder = "D:/InverseCookingTemp/JSON_according_to_retrieval"
add_retrieved_instructions(data_dir, retrieval_folder)
add_generative_instructions(data_dir, generative_folder)
