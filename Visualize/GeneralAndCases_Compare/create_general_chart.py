
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load two sheets from Excel
df_model1 = pd.read_excel('Inverse/Metric_IC.xlsx', sheet_name='per_image_avg')
df_model2 = pd.read_excel('Retrieval/Metric_RET.xlsx', sheet_name='Sheet1')

# Convert IoU, F1, ROUGE-L to percentage
for df in [df_model1, df_model2]:
    df['IoU'] = df['IoU'] * 100
    df['F1'] = df['F1'] * 100
    df['ROUGE-L'] = df['ROUGE-L'] * 100

# Metrics to compare
metrics = ['IoU', 'F1', 'ROUGE-L', 'SacreBLEU']

# Colors for models
color_model1 = '#637bef'  # Blue
color_model2 = '#ef6363'  # Red

# Create one figure with 4 subplots
plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Drop NaN values
    data1 = df_model1[metric].dropna()
    data2 = df_model2[metric].dropna()
    
    # Define bins
    bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 50)
    
    # Compute histogram counts
    counts1, _ = np.histogram(data1, bins=bins)
    counts2, _ = np.histogram(data2, bins=bins)
    
    # Compute bin centers
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4
    
    # Plot bars side by side (centered)
    plt.bar(centers - width/2, counts1, width=width, color=color_model1, alpha=0.8, label='Inverse Cooking')
    plt.bar(centers + width/2, counts2, width=width, color=color_model2, alpha=0.8, label='Retrieval')

    # Titles and axis
    plt.title(f'{metric} Comparison')
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)

    if metric in ['IoU', 'F1', 'ROUGE-L']:
        plt.xlabel(f'{metric} (%)')
    else:
        plt.xlabel(metric)
    
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.savefig('Charts/all_metrics_comparison.png')
plt.show()

