import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import random
import numpy as np

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
xdf = df[features]

dataDict = {"toxin":[],"numEstimators":[],"score":[]}

for numEstimators in [1,2,4,6,8,10,50,100,200,300,500]:
    print(numEstimators)
    xTrain = xdf.iloc[trainIndices].to_numpy()
    xTest = xdf.iloc[testIndices].to_numpy()

    for target in targets:
        model = GradientBoostingRegressor(n_estimators=numEstimators)
        yTrain = np.asarray(tdf[target])[trainIndices]
        yTest = np.asarray(tdf[target])[testIndices]
        model.fit(xTrain, yTrain)
        score = model.score(xTest, yTest)

        dataDict["toxin"].append(target)
        dataDict["numEstimators"].append(numEstimators)
        dataDict["score"].append(score)

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("numEstimators.csv", index=False)

