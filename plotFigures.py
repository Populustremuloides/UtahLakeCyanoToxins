import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os

cwd = os.getcwd()
figuresPath = os.path.join(cwd, "figures")

# Read the CSV file
df1 = pd.read_csv("feature_importances.csv")

# Make a copy of the DataFrame and drop specified columns
groupedDf = copy.copy(df1)
groupedDf = groupedDf.drop(["toxin", "replicate"], axis=1)

# Group by 'feature' and calculate the mean
groupedDf = groupedDf.groupby("feature").mean()

# Sort features by 'importance'
sortedVars = groupedDf.sort_values(by="importance", ascending=True).index

# Create a figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True, sharey=True)

# Define a coolwarm color palette
coolwarm_colors = sns.color_palette("coolwarm", len(sortedVars))

# Increase font sizes
plt.rc('axes', titlesize=18)   # fontsize of the axes title
plt.rc('axes', labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=16)  # legend fontsize

# Function to plot boxplot with the specified colors
def plot_colored_boxplot(ax, data, title, show_xticks=False):
    sns.boxplot(data=data, x="feature", y="importance", order=sortedVars, ax=ax, palette=coolwarm_colors)
    if show_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)

# Plot overall
plot_colored_boxplot(axes[0], df1, "Overall")

# Plot for 'ana' toxin
df_ana = df1[df1["toxin"] == "ana"]
plot_colored_boxplot(axes[1], df_ana, "Anatoxin-a")

# Plot for 'cylindro' toxin
df_cylindro = df1[df1["toxin"] == "cylindro"]
plot_colored_boxplot(axes[2], df_cylindro, "Cylindrospermopsin")

# Plot for 'micro' toxin
df_micro = df1[df1["toxin"] == "micro"]
plot_colored_boxplot(axes[3], df_micro, "Microcystin", show_xticks=True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(figuresPath, "featureImportances.png"))
plt.show()

