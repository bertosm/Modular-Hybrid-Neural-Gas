#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:41:29 2021

@author: BertoSosa
"""


import EA_GNG.EA_GNG
import warnings

# from EA_GNG.print import plotRocCurve_fromPath

from EA_GNG.core.dataset import loadDataset_CN_MCI_AD_fromPKL

warnings.filterwarnings("ignore")



filesPath = "C:/Users/alber/Desktop/GNG-Alzheimer-Comciencia/Datasets/MCI-AD/"
loadedDataset = EA_GNG.EA_GNG.loadDatasets(filesPath, concretFile='baseline_MCI-AD_ADNI2.xlsx')



# loadedDataset =  dict()
# loadedDataset["pkl"] = {"filePath":"C:/Users/alber/Desktop/GNG/Datasets/CN-MCI-AD/",
#                         "fileName":"CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl",
#                         "num_components":4,
#                         "scaled":"RobustScaler",
#                         "projection":"FactorAnalysis"
#                         }

# loadedDataset["pkl"] = {"filePath":"C:/Users/alber/Desktop/GNG/Datasets/CN-MCI-AD/",
#                         "fileName":"CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl",
#                         "num_components":3,
#                         "scaled":"RobustScaler",
#                         "projection":"FactorAnalysis"
#                         }


# list_epoch = (500, )
# list_max_age= (14, 16, 18, 20, 22, 24, 26, 28, 30)
# list_lambda = (158, 315, 625, 800, 948, 1250)
# list_max_nodes = (39, 51, 59, 67, 79)
# list_step = (0.2, 0.1)
# list_neighbour_step = (0.05, 0.006, 0.0006)

# datasetpath = "C:/Users/alber/Desktop/GNG/Datasets/CN-MCI-AD/"
# datasetFile = 'CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl'
# X_train, X_test, Y_train, Y_test = loadDataset_partitionared_CN_MCI_AD_fromPKL(datasetpath, datasetFile, num_components=4, scaled = "RobustScaler", projection="FactorAnalysis")

list_epoch = (500, )
list_max_age= (6, 8, 10, 12, 16, 20, 22, 26, 30, 40,)
list_lambda = (375, 625, 750, 1125)
list_max_nodes = (10, 20, 30, 40, 50, 60)
list_step = (0.3, 0.2, 0.1)
list_neighbour_step = (0.05, 0.006, 0.0006)

# """Parámetros perceptrón

list_neighborsActivation = (0,) #set 0 to use all the distances, 1 for only 1 neuron and other numbers for X nearest neurons

# list_learningRate = (0.0,0.02,0.04,0.06,0.08,
#                       0.1,0.12,0.14, 0.16,0.18,
#                       0.2,0.22,0.24, 0.26,0.28,
#                       0.3,0.32,0.34, 0.36,0.38,
#                       0.4,0.42,0.44, 0.46,0.48,
#                       0.5,0.52,0.54, 0.56,0.58,
#                       0.6,0.62,0.64, 0.66,0.68,
#                       0.7,0.72,0.74, 0.76,0.78,
#                       0.8,0.82,0.84, 0.86,0.88,
#                       0.9,0.92,0.94, 0.96,0.98, 1)

list_learningRate = (0.2, )

list_epochPerceptron = (20, )

dictConfig = EA_GNG.EA_GNG.makeConfigDict(list_epoch, list_max_age, list_lambda, list_max_nodes, 
                                          list_step = list_step, list_neighbour_step=list_neighbour_step,
                                          list_learningRate=list_learningRate, list_epochPerceptron=list_epochPerceptron,
                                          list_neighborsActivation=list_neighborsActivation)


# savingPathGNG= "C:/Users/alber/Desktop/GNG-AN_MCI_AD-balanced-RobustScaled-FactorAnalysis-partitionaired-4components-SecondTest/"

savingPathGNG= "C:/Users/alber/Desktop/MCI-AD_NOTPCA-StandardScaler-GNGv2/"
# savingPathGNG= "C:/Users/alber/Desktop/Articulo/Articulo-Portatil/Pruebas/pruebaPorcentajeClusters/"
# savingPathGNG= "C:/Users/alber/Desktop/3ComponentesPCA-Articulo/Pruebas/ProbandoCodigo/"

EA_GNG.EA_GNG.loopGrowingNeuralGas_perceptron(dictConfig, savingPathGNG, loadedDatasets = loadedDataset,  PCA=False, PCA_n_components = 3, savedGNG=False, saveProcess=True, hibrid =False)


