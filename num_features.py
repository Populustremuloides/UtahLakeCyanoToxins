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
random.shuffle(indices)
print(len(indices))
trainIndices = indices[:180]
testIndices = indices[180:]

tdf = df[targets]


# dataDict1 = {"toxin":[],"numEstimators":[],"score":[],"replicate":[]}
# dataDict2 = {"toxin":[],"importance":[],"feature":[],"replicate":[]}


featuresOriginal = copy.deepcopy(features)
dataDict = {}
dataDict["toxin"] = []
dataDict["replicate"] = []
dataDict["score"] = []
dataDict["numFeatures"] = []

for target in targets:
    featuresCopy = copy.deepcopy(features)
    while len(featuresCopy) > 10:
        xdf = df[featuresCopy]
        importances = []
        yTrain = np.asarray(tdf[target])[trainIndices]
        yTest = np.asarray(tdf[target])[testIndices]
        replicateScores = []
        for replicate in range(10):
            # print(replicate)
            numEstimators = 50
            xTrain = xdf.iloc[trainIndices].to_numpy()
            xTest = xdf.iloc[testIndices].to_numpy()

            model = GradientBoostingRegressor(n_estimators=numEstimators)
            model.fit(xTrain, yTrain)
            score = model.score(xTest, yTest)
            replicateScores.append(score)
            importances.append(model.feature_importances_)

            dataDict["toxin"].append(target)
            dataDict["replicate"].append(replicate)
            dataDict["score"].append(score)
            dataDict["numFeatures"].append(len(featuresCopy))

        importances = np.asarray(importances)
        meanImportances = np.mean(importances, axis=0)
        worst = np.argmin(meanImportances)
        del featuresCopy[worst]
        print(len(features))
        print(len(featuresCopy))

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("numFeatures.csv", index=False)
