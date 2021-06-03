import pickle
import numpy as np
import pandas as pd


########################################################################################################################
# Unpickles and creates three .csv files to store steady state (ss), stall and resurrection (resur) data as provided by
# Nord et al. (2017).

fileNames = ["NilsD300_V2.p", "NilsD500_V2.p", "NilsD1300_V2.p"]

ssFinal = pd.DataFrame()
stallFinal = pd.DataFrame()
resurFinal = pd.DataFrame()

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
        ss = ss[::100]
        ss = ss[:-1]
        ssList.append(ss)
        ssArray = pd.DataFrame(ssList)

        stall = data[motorKeys[j]]["statnum_after_release"]
        stall = stall[::100]
        stall = stall[:-1]
        stallList.append(stall)
        stallArray = pd.DataFrame(stallList)

        try:
            resur = data[motorKeys[j]]["statnum_resurrection"]
            resur = resur[::100]
            resur = resur[:-1]
            resurList.append(resur)
            resurArray = pd.DataFrame(resurList)

        except KeyError:
            pass

    ssFinal = pd.concat([ssFinal, ssArray])
    stallFinal = pd.concat([stallFinal, stallArray])
    resurFinal = pd.concat([resurFinal, resurArray])

ssFinal.to_csv("ssData.csv")
stallFinal.to_csv("stallData.csv")
resurFinal.to_csv("resurData.csv")
