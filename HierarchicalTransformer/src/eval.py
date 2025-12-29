# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from config import get_eval_args
import random

random.seed(1234)
import os
import pickle
from utils.metrics import compute_metrics
import argparse
import json


def computeAverageMetrics(imfeats, recipefeats, ids, k, t, forceorder=False):
    """Computes retrieval metrics for two sets of features

    Parameters
    ----------
    imfeats : np.ndarray [n x d]
        The image features..
    recipefeats : np.ndarray [n x d]
        The recipe features.
    k : int
        Ranking size.
    t : int
        Number of evaluations to run (function returns the average).
    forceorder : bool
        Whether to force a particular order instead of picking random samples

    Returns
    -------
    dict
        Dictionary with metric values for all t runs.

    """

    glob_metrics = {}
    i = 0
    with open("subset_metrics.txt", "w") as out_file:
        for run_idx in range(t):
            if forceorder:
                sub_ids_idx = np.array(range(i, i + k))
                i += k
            else:
                sub_ids_idx = random.sample(range(0, len(imfeats)), k)
            imfeats_sub = imfeats[sub_ids_idx, :]
            recipefeats_sub = recipefeats[sub_ids_idx, :]
            ids_sub = ids[sub_ids_idx]

            out_file.write(f"\nRun {run_idx+1}:\n")
            out_file.write("Subset IDs: " + ", ".join([str(x) for x in ids_sub]) + "\n")
            out_file.write("Image features:\n" + str(imfeats_sub) + "\n")
            out_file.write("Recipe features:\n" + str(recipefeats_sub) + "\n")

            metrics, raw = compute_metrics(
                imfeats_sub, recipefeats_sub, recall_klist=(1, 5, 10), return_raw=True
            )

            for metric_name, metric_value in metrics.items():
                if metric_name not in glob_metrics:
                    glob_metrics[metric_name] = []
                glob_metrics[metric_name].append(metric_value)
                out_file.write(
                    f"Metric '{metric_name}' for this subset: {metric_value}\n"
                )
    return glob_metrics


def eval(args):

    # Load embeddings
    with open(args.embeddings_file, "rb") as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)

    # sort by name so that we always pick the same samples
    idxs = np.argsort(ids)
    ids = ids[idxs]
    recipefeats = recipefeats[idxs]
    imfeats = imfeats[idxs]

    # Save sorted IDs to a text file
    with open("sorted_ids.txt", "w") as id_file:
        for id_val in ids:
            id_file.write(str(id_val) + "\n")

    if args.retrieval_mode == "image2recipe":
        glob_metrics = computeAverageMetrics(
            imfeats, recipefeats, ids, args.medr_N, args.ntimes
        )
    else:
        glob_metrics = computeAverageMetrics(
            recipefeats, imfeats, ids, args.medr_N, args.ntimes
        )

    for k, v in glob_metrics.items():
        print(k + ":", np.mean(v))


def evalVer2(args):
    # Load embeddings
    with open(args.embeddings_file, "rb") as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)

    # sort by name so that we always pick the same samples
    idxs = np.argsort(ids)
    ids = ids[idxs]
    recipefeats = recipefeats[idxs]
    imfeats = imfeats[idxs]

    # For each image, find the top-1 retrieved recipe and its ground truth
    retrieval_pairs = []
    for i, img_feat in enumerate(imfeats):
        # Compute cosine similarity to all recipe features
        sims = np.dot(recipefeats, img_feat) / (
            np.linalg.norm(recipefeats, axis=1) * np.linalg.norm(img_feat) + 1e-8
        )
        top1_idx = np.argmax(sims)
        retrieved_id = ids[top1_idx]
        gt_id = ids[i]
        retrieval_pairs.append(
            (str(retrieved_id), str(gt_id))
        )  # convert to str for JSON compatibility

    # Write retrieval_pairs to a JSON file
    output_path = "retrieval_pairs.json"
    with open(output_path, "w") as json_file:
        json.dump(retrieval_pairs, json_file, indent=2)


if __name__ == "__main__":

    args = get_eval_args()
    evalVer2(args)
