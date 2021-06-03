import pickle
import numpy as np
import pandas as pd


# ######################################################################################################################
# Unpickles and creates three .csv files to store mean steady state (ss), stall and resurrection (resur) data as
# provided by Nord et al. (2017).

fileNames = ["NilsD300_V2.p", "NilsD500_V2.p", "NilsD1300_V2.p"]

ssFinal = []
stallFinal = []
resurFinal = []

for i in range(3):
    unpickle = open(fileNames[i], "rb")
    data = pickle.load(unpickle)
    motorKeys = np.fromiter(data.keys(), dtype=int)

    ssList = []
    stallList = []
    resurList = []
    ssArray = []
    stallArray = []
    resurArray = []

    for j in range(len(motorKeys)):
        ss = data[motorKeys[j]]["statnum_before_stall"]
        ss = ss[::1000]
        ss = ss[:405]
        ssList.append(ss)
        ssArray = np.array(ssList)

        stall = data[motorKeys[j]]["statnum_after_release"]
        stall = stall[::1000]
        stall = stall[:331]
        stallList.append(stall)
        stallArray = np.array(stallList)

        try:
            resur = data[motorKeys[j]]["statnum_resurrection"]
            resur = resur[::1000]
            resur = resur[:270]
            resurList.append(resur)
            resurArray = np.array(resurList)

        except KeyError:
            pass

    ssMean = np.mean(ssArray, axis=0)
    ssFinal.append(ssMean)

    stallMean = np.mean(stallArray, axis=0)
    stallFinal.append(stallMean)

    resurMean = np.mean(resurArray, axis=0)
    resurFinal.append(resurMean)

ssData_mean = pd.DataFrame(ssFinal)
stallData_mean = pd.DataFrame(stallFinal)
resurData_mean = pd.DataFrame(resurFinal)

ssData_mean.to_csv("mean_ssData.csv")
stallData_mean.to_csv("mean_stallData.csv")
resurData_mean.to_csv("mean_resurData.csv")
