
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd
import numpy as np

all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10, 30, 50]  
all_extra_amounts = [50, 100]


# Define the categories and their corresponding values
categories = ['Mean/mode imputation', 'MICE', 'kNN', 'MissForest', 'GAIN']
values_list = [20, 35, 30, 15, 50], [20, 35, 30, 15, 50]

# Create a new figure and a grid of axes
fig, axs = plt.subplots(1, 2)

# Define the colors for each category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
num_colors = 5
colors = mcolors.ListedColormap(mcolors.TABLEAU_COLORS).colors[:num_colors]

# Plot the bars for each subplot
legend_handles = []
legend_labels = []
for i, ax in enumerate(axs.flat):
    bars = ax.bar(categories, values_list[i], color=colors)
    for j, bar in enumerate(bars):
        bar.set_label(categories[j])
        if i == 0:
            legend_handles.append(bar)
            legend_labels.append(categories[j])
    ax.set_title(f'Chart {i+1}')
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([])

# Add the legend to the figure
fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', ncol=len(categories), bbox_to_anchor=(0.5, 0.0))

# Show the plot
plt.show()




data = pd.read_csv("results/Results - Results without CTGAN.csv")

print(data)