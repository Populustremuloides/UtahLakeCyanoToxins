import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import random
import numpy as np
import copy

random.seed(0)
df = pd.read_csv("preparedData_-1.csv")
print(df)
features = list(df.columns)
print(len(features))
targets = ["micro","ana","cylindro"]
features = list(set(features).difference(set(targets)))
features.sort()
print(len(features))
indices = list(range(len(df["micro"])))

dataDict1 = {"toxin":[],"numEstimators":[],"score":[],"replicate":[]}
dataDict2 = {"toxin":[],"importance":[],"feature":[],"replicate":[]}


featuresOriginal = copy.deepcopy(features)
dataDict = {}
# dataDict["toxin"] = []
# dataDict["replicate"] = []
# dataDict["score"] = []
# dataDict["numFeatures"] = []

for target in targets:
    for replicate in range(100):
        random.shuffle(indices)
        trainIndices = indices[:180]
        testIndices = indices[180:]
        tdf = df[targets]
        xdf = df[features]

        yTrain = np.asarray(tdf[target])[trainIndices]
        yTest = np.asarray(tdf[target])[testIndices]

        # print(replicate)
        numEstimators = 50
        xTrain = xdf.iloc[trainIndices].to_numpy()
        xTest = xdf.iloc[testIndices].to_numpy()

        model = GradientBoostingRegressor(n_estimators=numEstimators)
        model.fit(xTrain, yTrain)
        score = model.score(xTest, yTest)

        dataDict1["toxin"].append(target)
        dataDict1["numEstimators"].append(numEstimators)
        dataDict1["score"].append(score)
        dataDict1["replicate"].append(replicate)

        for i in range(len(features)):
            dataDict2["toxin"].append(target)
            dataDict2["importance"].append(model.feature_importances_[i])
            dataDict2["feature"].append(features[i])
            dataDict2["replicate"].append(replicate)


df1 = pd.DataFrame.from_dict(dataDict1)
df1.to_csv("model_accuracies.csv", index=False)

df2 = pd.DataFrame.from_dict(dataDict2)
df2.to_csv("feature_importances.csv", index=False)
