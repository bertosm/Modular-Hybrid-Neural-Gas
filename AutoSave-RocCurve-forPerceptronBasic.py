# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 20:16:28 2021

@author: Bertosm
"""

import pandas as pd

from os import path, makedirs
from EA_GNG.core.dataset import ls, ls_dirs

pathDirs = "C:/Users/Bertosm/Desktop/3PCA-MyGNG-withPerceptronBasic-weights0to1/"

# A veces falla con 0 tipo INT:
rocCurveForLimit = "0"

directories = ls_dirs(pathDirs)

dfs = {"AN0": pd.DataFrame(columns=("learningRate", "falsosPositivos", "verdaderosPositivos")),
"AN1": pd.DataFrame(columns=("learningRate", "falsosPositivos", "verdaderosPositivos")),
"AN3": pd.DataFrame(columns=("learningRate", "falsosPositivos", "verdaderosPositivos")),
"AN5": pd.DataFrame(columns=("learningRate", "falsosPositivos", "verdaderosPositivos"))}

for directory in directories:
    # print(directory)
    
    learningRate = directory.split("-")[0].replace("lr", "")
    epochs = directory.split("-")[1].replace("epch", "")
    
    files=ls(pathDirs + directory + "/")
    
    for filePath in files: 
        indexpoint = filePath.rfind('.')
        
        #si el fichero no tiene extension, evitarlo
        if indexpoint == -1:
            continue
        
        elif not filePath[indexpoint:] in ('.csv'):
            continue
        
        elif "final-RocCurve" not in filePath:
            continue
    
        AN = filePath.split("-")[0]
        print("AN: ", AN, directory, "filePath: ", filePath)
        
        dfResult = pd.read_csv(pathDirs + directory + "/" + filePath, sep=";", names=("limit","falsosPositivos", "verdaderosPositivos"))
        # print(dfResult.shape, dfResult.columns.tolist(), dfResult.iloc[0])
        # print("lr: ", learningRate, "ep: ", epochs, "result: ", dfResult)
        
        # print("df", type(dfResult['limit'].iloc[1]))
        print("limit0: ", dfResult[dfResult['limit'] == rocCurveForLimit])
        
        dfs[AN] = dfs[AN].append(pd.DataFrame({"learningRate": learningRate, "falsosPositivos":dfResult[dfResult['limit'] == rocCurveForLimit]['falsosPositivos'], "verdaderosPositivos":dfResult[dfResult['limit'] == rocCurveForLimit]['verdaderosPositivos']}), ignore_index=True)
        print(dfs[AN].shape)
    for df in dfs:
        if not dfs[df].empty:
            print()
            dfs[df].to_csv("{}{}-final-RocCurve.csv".format(pathDirs, df), ";")
    
    
        
    
    
    
    
    