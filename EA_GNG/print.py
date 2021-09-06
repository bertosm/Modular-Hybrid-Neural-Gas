#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:29:24 2021

@author: Berto Sosa
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
# from EA_GNG.core.dataset import ls
from core.dataset import ls

PlotOptions = {"AN0": {"marker":".", "linestyle":(0, (3, 1, 1, 1, 1, 1))},
               "AN1":  {"marker":".", "linestyle":"dotted"},
               "AN3":  {"marker":".", "linestyle":"solid"},
               "AN5":  {"marker":".", "linestyle":"--"},
               "ANNOT":  {"marker":"+", "linestyle":(0, (3, 1, 1, 1, 1, 1))}
               }

def plotRocCurve_fromPath(path):

    #coger todos los ficheros que se encuentra en la dirección pasada
    files = ls(path)
        
    fileAuc = open("{}AUCValues.txt".format(path), "w") #fichero metrica resultante
    fileAuc.write('file;AUC\n')
        
        
    for filePath in files: 
        indexpoint = filePath.rfind('.')
        
        #si el fichero no tiene extension, evitarlo
        if indexpoint == -1:
            continue
        elif not filePath[indexpoint:] in ('.csv'):
            continue
        elif "final-RocCurve" not in filePath:
            continue
    
        splitNameFile = filePath.split("-")
        if len(splitNameFile) > 3:
            moreInfoLabel = splitNameFile[3].split(".")[0]
        else:
            moreInfoLabel = ""
            
        AN = splitNameFile[0]
        
        print(AN)
        fileAuc.write(filePath+";")
        plotRocCurve(path + filePath, AN, fileAuc, moreInfoLabel)
        
    plotRocCurveConfiguration(path)
    fileAuc.close()
    
    
def plotRocCurve(filePath, AN = "ANNOT", fileAuc = None, moreInfoLabel=""):
    
    if "Basic" in filePath:
        dfRoc = pd.read_csv(filePath, sep=",")
    elif "BackPropagation" in filePath:
        dfRoc = pd.read_csv(filePath, sep=";")
        
    print(dfRoc)
    
    sensibilidad =dfRoc["falsosPositivos"]    
    
    sensibilidad.loc[-1] = 0
    sensibilidad.loc[50] = 1
    sensibilidad.index = sensibilidad.index + 1
    sensibilidad.sort_index(inplace=True)
    
    unoMenosEspecificidad = dfRoc["verdaderosPositivos"] 
    
    unoMenosEspecificidad.loc[-1] = 0
    unoMenosEspecificidad.loc[50] = 1
    unoMenosEspecificidad.index = unoMenosEspecificidad.index + 1
    unoMenosEspecificidad.sort_index(inplace=True)
    
    print("falsosPositivos: ", sensibilidad)
    print("verdaderosPositivos: ", unoMenosEspecificidad)
    
    try:
        aucValue = auc(sensibilidad, unoMenosEspecificidad)
    except:
        print("error Not lineal Roc")
        aucValue = 0.5
        
    print("AUC of " + AN, aucValue)
    
    if fileAuc != None:
        fileAuc.write("{}\n".format(aucValue))
        
    plt.plot(sensibilidad, unoMenosEspecificidad, marker=PlotOptions[AN]['marker'],
          label ="{}-Auc{}".format(AN, round(aucValue,2)), linestyle=PlotOptions[AN]['linestyle'])
    
    
def plotRocCurveConfiguration(saving_path):

    
    plt. plot((0,1), (0,1), linestyle="--")
    plt.xlabel('False Positive Rate - FPR')
    plt.ylabel('True Positive Rate - TPR')
    plt.legend()
    plt.title('ROC curve')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.rcParams['axes.facecolor'] = 'none'
    plt.style.use('grayscale')
    
    savename = saving_path+'RocCurve-final.png'
    plt.savefig(savename,bbox_inches = 'tight')
    plt.close()



print("Plotting Roc Curve from Path!")
plotRocCurve_fromPath("C:/Users/Bertosm/Desktop/3PCA-MyGNG-withPerceptronBasic-weights0to1/lr0.2-epch20/")

""" versión anterior:

# filepath1 = "/Users/pedrososa/Desktop/ArticuloEscrito/VersionesAnteriores/Articulo4_0/curveRocAct1.xlsx"
# dfRoc1 = pd.read_excel(filepath1)
# sensibilidad1 = dfRoc1["a"]
# unoMenosEspecificidad1 = dfRoc1["b"]
# auc1 = auc(sensibilidad1, unoMenosEspecificidad1)
# print("AUC1: ", auc1)

# filepath3 = "/Users/pedrososa/Desktop/ArticuloEscrito/VersionesAnteriores/Articulo4_0/curveRocAct3.xlsx"
# dfRoc3 = pd.read_excel(filepath3)
# sensibilidad3 = dfRoc3["a"].5
# unoMenosEspecificidad3 = dfRoc3["b"]
# auc3 = auc(sensibilidad3, unoMenosEspecificidad3)
# print("AUC3: ", auc3)

filepath5 = "C:/Users/Bertosm/Desktop/PruebasArticulo/CurvaRoc/CurvaRoc-Vecindad5.xlsx"
dfRoc5 = pd.read_excel(filepath5)
sensibilidad5 = dfRoc5["a"]
unoMenosEspecificidad5 = dfRoc5["b"]
auc5 = auc(sensibilidad5, unoMenosEspecificidad5)
print("AUC5: ", auc5)



# fileOnlyPerceptron = "/Users/pedrososa/Desktop/ArticuloEscrito/VersionesAnteriores/Articulo4_0/curveRocPerceptron.xlsx"
# dfRocPerc = pd.read_excel(fileOnlyPerceptron)
# sensibilidadPerc = dfRocPerc["a"]
# unoMenosEspecificidadPerc = dfRocPerc["b"]
# aucPerc = auc(sensibilidadPerc, unoMenosEspecificidadPerc)
# print("AUC-Perceptron: ", aucPerc)


# bbox1 = dict(boxstyle="round", fc="black")
# bbox3 = dict(boxstyle="round", fc="black", alpha=0.6)
bbox5 = dict(boxstyle="round", fc="black", alpha=0.3)
# bboxPerc = dict(boxstyle="round", fc="black", alpha=0.1)

arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

offset=50

# plt.plot(sensibilidad1, unoMenosEspecificidad1, marker=".",
#          label ="MyGNG-AN1", linestyle="dotted")
# plt.plot(sensibilidad3, unoMenosEspecificidad3, marker=".",
#          label ="MyGNG-AN2", linestyle="solid")
plt.plot(sensibilidad5, unoMenosEspecificidad5, marker=".",
         label ="MyGNG-3PC-Neighborhood5", linestyle="--")

# plt.plot(sensibilidadPerc, unoMenosEspecificidadPerc, marker="+",
#           label ="perceptron", linestyle= (0, (3, 1, 1, 1, 1, 1)))

# plt.annotate("BestAN1", (0.18181818, 0.90909091), 
#              xytext=(offset, 7),
#              textcoords='offset points',
#              color='white',
#              bbox=bbox1, arrowprops=arrowprops)
# plt.annotate("BestAN2", (0.06060606, 0.72727273),
#              xytext=(0, -offset),
#              textcoords='offset points',
             # color='white',
             # bbox=bbox3, arrowprops=arrowprops)
plt.annotate("Best3PC-AN5", (0.06060606, 0.75757576),
             xytext=(offset, 7),
             textcoords='offset points', 
             color='white',
             bbox=bbox5, arrowprops=arrowprops)
# plt.annotate("BestPerceptron", (0.01515152, 0.63636364),
#               xytext=(-0.5*offset, offset*1.4),
#               textcoords='offset points', 
#               color='white',
#               bbox=bbox5, arrowprops=arrowprops)

plt. plot((0,1), (0,1), linestyle="--")
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.legend()
plt.title('ROC curve--AUC={}'.format(auc5),pad=15)
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.rcParams['axes.facecolor'] = 'none'
plt.style.use('grayscale')

savename = 'C:/Users/Bertosm/Desktop/PruebasArticulo/CurvaRoc/RocCurve-3PC-Neighborhood5.png'
plt.savefig(savename,bbox_inches = 'tight')
plt.close()

"""