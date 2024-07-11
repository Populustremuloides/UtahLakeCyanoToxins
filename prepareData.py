import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt



df = pd.read_csv("UL_Data_Brian.csv")
df = df.drop(df.columns[-1], axis=1)

df = df[~df["micro"].isna()]
df = df[~df["ana"].isna()]
df = df[~df["cylindro"].isna()]
print(df)

# change nominal to numeric data
for col in df.columns:
    le = preprocessing.LabelEncoder()
    x = np.asarray(df[col])
    try:
        x = [float(y) for y in x]
        if col == "Dilution":
            le.fit(x)
            xHat = le.transform(x)
            df[col] = xHat
    except:
        le.fit(x)
        xHat = le.transform(x)
        df[col] = xHat

def sqrtArray(array):
    array = np.asarray([float(x) for x in array])
    array = array + 0.0000000001
    return np.sqrt(array)

def logArray(array):
    array = np.asarray([float(x) for x in array])
    array = array + 2
    return np.log(array)

def replaceNan(array):
    newArray = []
    for item in array:
        if item == "nan":
            newArray.append(np.nan)
            print("got one!")
        else:
            newArray.append(item)
    return newArray

# for element in replaceNan(df["Phyco"]):
#     print(element)
# # replace na with -1
# df = df.drop("rdom rfu", axis=0)
# print(df.columns[-1])
# df = df.drop(df.columns[-1], axis=1)
# quit()
numBad = 0
numTotal = 0
for col in df.columns:
    numBad += np.sum(np.asarray(df[col].isna()))
    numTotal += len(list(df[col]))
print(numBad)
print(numTotal)
print(len(df.columns))
print(df)
quit()

i = 0
for col in df.columns:
    # df[col]
    # if col == df.columns[-1]:
    #     print(col)
    maskGood = np.asarray(~df[col].isna())
    maskBad = np.asarray(df[col].isna())
    x = np.asarray(df[col])
    if i > 8:
        if col == "cylindro":
            xStar = sqrtArray(sqrtArray(x[maskGood]))
        else:
            xStar = logArray(x[maskGood])
        # if col == df.columns[-1]:
        #     littleMask = np.asarray(pd.isna(xStar))
            # print(xStarStar[littleMask])
            # print(np.mean(xStar))
            # print(np.std(xStar))
    else:
        xStar = x[maskGood]
    xStar = (xStar - np.mean(xStar)) / np.std(xStar)
    x[maskGood] = xStar
    x[maskBad] = np.min(x[maskGood]) -1
    df[col] = x
    i = i + 1

# quit()
# for col in df.columns:
#     # print(col)
#     # print(df[col])
#     plt.hist(df[col], bins=50)
#     plt.title(col)
#     plt.show()

    # plt.hist(logArray(df[col]), bins=30)
    # plt.title(col)
    # plt.show()

df.to_csv("preparedData_-1.csv", index=False)
