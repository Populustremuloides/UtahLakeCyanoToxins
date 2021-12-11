import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("model_accuracies.csv")
toxins = list(set(list(df["toxin"])))
toxins.sort()
for toxin in toxins:
    ldf = df[df["toxin"] == toxin]
    print(toxin, np.mean(ldf["score"]), np.std(ldf["score"]))
# print(df)


plt.grid(alpha=0.3)
sns.violinplot(data=df, x="toxin",y="score")
plt.title("GBR Model Accuracies, n=100")
plt.show()



# df = pd.read_csv("numFeatures.csv")
# sns.lineplot(data=df, x="numFeatures",y="score",hue="toxin")
# plt.show()

df1 = pd.read_csv("feature_importances.csv")
df = df1[df1["toxin"] == "ana"]
sns.boxplot(data=df, x="feature",y="importance")
plt.xticks(rotation=90)
plt.title("feature importances - ana")
plt.show()

df = df1[df1["toxin"] == "micro"]
sns.boxplot(data=df, x="feature",y="importance")
plt.xticks(rotation=90)
plt.title("feature importances - micro")
plt.show()

df = df1[df1["toxin"] == "cylindro"]
sns.boxplot(data=df, x="feature",y="importance")
plt.xticks(rotation=90)
plt.title("feature importances - cylindro")
plt.show()