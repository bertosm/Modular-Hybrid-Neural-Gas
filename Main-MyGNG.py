#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:41:29 2021

@author: BertoSosa
"""


import EA_GNG.EA_GNG
import warnings

warnings.filterwarnings("ignore")

filesPath = "C:/Users/Bertosm/Desktop/GNG-Alzheimer-Comciencia/Datasets/MCI-AD/"
# filesPath = "C:/Users/Bertosm/Desktop/Datasets/"
loadedDataset = EA_GNG.EA_GNG.loadDatasets(filesPath, concretFile='baseline_MCI-AD_ADNI2.xlsx')

# list_epoch = (500, )
# list_max_age= (14, 16, 18, 20, 22, 24, 26, 28, 30)
# list_lambda = (158, 315, 625, 800, 948, 1250)
# list_max_nodes = (39, 51, 59, 67, 79)
# list_step = (0.2, 0.1)
# list_neighbour_step = (0.05, 0.006, 0.0006)


list_epoch = (500, )
list_max_age= (6, )
list_lambda = (625,)
list_max_nodes = (27, )
list_step = (0.2, )
list_neighbour_step = (0.05, )

# """Parámetros perceptrón

list_neighborsActivation = (0,) #set 0 to use all the distances, 1 for only 1 neuron and other numbers for X nearest neurons

list_learningRate = (0.2,)
list_epochPerceptron = (20,)

dictConfig = EA_GNG.EA_GNG.makeConfigDict(list_epoch, list_max_age, list_lambda, list_max_nodes, 
                                          list_step = list_step, list_neighbour_step=list_neighbour_step,
                                          list_learningRate=list_learningRate, list_epochPerceptron=list_epochPerceptron,
                                          list_neighborsActivation=list_neighborsActivation)


savingPathGNG= "C:/Users/Bertosm/Desktop/testGNG/"
# savingPathGNG= "C:/Users/alber/Desktop/Articulo/Articulo-Portatil/Pruebas/pruebaPorcentajeClusters/"
# savingPathGNG= "C:/Users/alber/Desktop/3ComponentesPCA-Articulo/Pruebas/ProbandoCodigo/"

EA_GNG.EA_GNG.loopGrowingNeuralGas_perceptron(dictConfig, savingPathGNG, loadedDatasets = loadedDataset,  PCA=True, PCA_n_components = 2, savedGNG=False, saveProcess=True, hibrid =False)


