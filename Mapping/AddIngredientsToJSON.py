import os
import json


def add_retrieved_ingredients(data_folder, retrieval_folder):
    # Build a mapping from prefix to ingredient info
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
                retrieval_map[prefix] = {
                    "gt_ingredients": result.get("gt_ingredients", []),
                    "gt_indices": result.get("gt_indices", []),
                    "retrieved_ingredients": result.get("pred_ingredients", []),
                    "retrieved_indices": result.get("pred_indices", []),
                }

    for data_file in os.listdir(data_folder):
        if not data_file.endswith(".json"):
            continue
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, "r", encoding="utf-8") as f:
            objects = json.load(f)

        for obj in objects:
            pair_id = obj.get("pair id for retrieval")
            if pair_id and pair_id in retrieval_map:
                info = retrieval_map[pair_id]
                obj["gt_ingredients"] = info["gt_ingredients"]
                obj["gt_indices"] = info["gt_indices"]
                obj["retrieved_ingredients"] = info["retrieved_ingredients"]
                obj["retrieved_indices"] = info["retrieved_indices"]

        with open(data_path, "w", encoding="utf-8") as out_f:
            json.dump(objects, out_f, indent=2, ensure_ascii=False)


def add_generated_ingredients(data_folder, generative_folder):
    # Build a mapping from prefix (image id) to generated ingredient info
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
                generative_map[prefix] = {
                    "generated_ingredients": result.get("pred_ingredients", []),
                    "generated_indices": result.get("pred_indices", []),
                }

    for data_file in os.listdir(data_folder):
        if not data_file.endswith(".json"):
            continue
        data_path = os.path.join(data_folder, data_file)
        with open(data_path, "r", encoding="utf-8") as f:
            objects = json.load(f)

        for obj in objects:
            image_id = obj.get("image id")
            if image_id and image_id in generative_map:
                info = generative_map[image_id]
                obj["generated_ingredients"] = info["generated_ingredients"]
                obj["generated_indices"] = info["generated_indices"]

        with open(data_path, "w", encoding="utf-8") as out_f:
            json.dump(objects, out_f, indent=2, ensure_ascii=False)


data_dir = "IngredientsCases"
retrieval_folder = "D:/Revamping/output2"
generative_folder = "D:/InverseCookingTemp/JSON_according_to_retrieval"
add_retrieved_ingredients(data_dir, retrieval_folder)
add_generated_ingredients(data_dir, generative_folder)
