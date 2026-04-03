## Steps to run
Step 1: Download output data for two folders Metric and Recipe of both Inverse and Retrieval folders from this link: https://drive.google.com/drive/folders/1JDBZ5-Ksygy2YetS-I6_-ukRkF5411Bc?usp=sharing

Step 2: Run python script files to bring metrics result from JSON files to excel file for both Inverse and Retrieval. Take a look at ```Inverse/json_to_excel.py``` and ```Retrieval/json_to_excel.py ```

Step 3: Run python script file to map metrics result of Inverse Cooking and Retrieval for each input test and split cases for comparison. Take a look at ```mapping_IC_RET.py ```

Step 4: Run python script files to create histograms of metrics result for general and three F1 cases. Take a look at Take a look at ```create_general_chart.py``` and ```create_case_chart.py ```