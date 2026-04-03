## Purpose
This folder is use to map for later usage of comparing output of Retrieval Model and Inverse Cooking. 

## Explanation
For each recipe (image + instruction), there is a generated recipe and a retrieved recipe, both are stored in JSON file for easy usage. We take the JSON file of retrieved model as the base and try to map and extend generated recipe to create a JSON file contain of 3 recipes, 1 for ground truth, 1 for retrieval and 1 for generative

## The order to run file in this folder
### For mapping retrieval pair with image ID used in Inverse Cooking
Run ```MappingRetrievalPairsForReport.py``` script

### For filtering out and mapping recipes for report writing
1. Choose retrieval pair
2. Run ```FilterRetrievalPairsForReport.py``` script to get the pair of retrieval, which contains both recipe ID for ground truth and retrieal
3. Then continue adding image if needed by running ```Visualize\IngredientsCompare\AddImageIdToCombinedJSON.py``` and ```Visualize\IngredientsCompare\AddImagePathToCombinedJSON.py``` scripts for ingredients visualizing or 
