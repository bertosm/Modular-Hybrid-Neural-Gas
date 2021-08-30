#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:18:22 2021

@author: Berto Sosa
"""

from EA_GNG.core.dataset import splitDataset
from EA_GNG.print import plotRocCurve_fromPath
import pickle

import EA_GNG.perceptronBiClase as pctron
# import EA_GNG.perceptronBackpropagationBiclase as pctron

def GNG_perceptron(param_dict_gng, param_dict_perceptron, df, saving_path, nameDataset = "", verbose= False, savedGNG=False, saveProcess=False, target = "DX_bl"):
    trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY = splitDataset(df, 0.20, target, param_dict_gng['seed'])
    
    trainLabelsTrueY.replace(2,0, True)
    trainLabelsTrueY.replace(3,1, True)
    testLabelsTrueY.replace(2,0, True)
    testLabelsTrueY.replace(3,1, True)

    saving_path = "{}lr{}-epch{}/".format(saving_path, param_dict_perceptron["learningRate"], param_dict_perceptron["epochs"])
    
    if not savedGNG:
        print("revisar esta parte de código (GNG+perceptron entranamiento continuado. codigo antiguo gng+percetron)")
        raise
        
    #  Este código sirve para entrenar y guardar la red requerida:
    #     gng_neupy = growingNeuralGas_perceptron(param_dict_gng,
    #                                             trainDataX=trainDataX, 
    #                                             trainLabelsTrueY=trainLabelsTrueY, 
    #                                             columns=df.columns.tolist(), 
    #                                             saving_path=saving_path, 
    #                                             target=target, 
    #                                             verbose=verbose, 
    #                                             nameDataset=nameDataset, 
    #                                             saveProcess=saveProcess)
    #     gng_neupy.saveGNG("C:/Users/Bertosm/Desktop/Config420epochs/savedGNG/gngSave114Calinski.pkl")
        
    #     raise
    #     limit=0.5
    #     percep = perceptron(param_dict_perceptron, limit)
    #     percep.train(trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY, saving_path, gng_neupy)
        
        
        
    else:
        # gng_neupy = openGNG("C:/Users/Bertosm/Desktop/PruebasEuclideanDistance/savedGNG/gngSave147Calinski.pkl")
        # gng_neupy = openGNG("C:/Users/Bertosm/Desktop/GNG/GNG-Tests/3PCA-6FeaturesFCBF/Calinski113-30_05_21/savedGNG/gngSave113Calinski.pkl")
        gng_neupy = openGNG("C:/Users/Bertosm/Desktop/GNG-Alzheimer-Comciencia/MyGNG-TEST/MCI-AD/2PCA/GNG+Perceptron/2PCA-6FCBF/calinski147-30_05_21/savedGNG/gngSaved147Calinski-2PCA.pkl")
        print("neuroas vecinas activacion: ", param_dict_perceptron["activation_neigbor"])
        print("Nº neuronas en el grafo preguardado: ", len(gng_neupy.graph.nodes))
        
        # Entrenamiento único ( es decir con límite/umbral establecido )
        limit=0
        # percep = pctron.perceptron(param_dict_perceptron, limit, gng_neupy) #MyGNG perceptron after a GNG graph trained
        percep = pctron.perceptron(param_dict_perceptron, limit) #Only perc
        percep.train(trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY,saving_path) #MyGNG perceptron after a GNG graph trained
        
        # Entrenamiento Variando límite/umbral y exposición de curva Roc final. 
        
        
        # limit = 1
        # while( limit >= 0):
        #     print("\n   limite: ", limit)
        #     """Asegurar el parametro gng_neupy se incluye!"""
        #     # percep = pctron.perceptron(param_dict_perceptron, limit, gng_neupy) #MyGNG perceptron after a GNG graph trained
        #     percep = pctron.perceptron(param_dict_perceptron, limit) #Only perceptron!
        #     percep.train(trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY,saving_path) #MyGNG perceptron after a GNG graph trained
        #     limit -= 0.02
        
        # plotRocCurve_fromPath(saving_path)



def openGNG(path):
    with open(path, "rb") as input:
        return pickle.load(input)
    
  