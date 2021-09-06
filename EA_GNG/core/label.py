#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:45:48 2020

@author: Berto Sosa
"""

import pandas as pd
import numpy as np
    


def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"
   


      
def get_diffents_key(val, colorPredicted, dictClusterLabel):
    for key, value in colorPredicted.items():
        if val == value:
            if key.split('-')[0] in dictClusterLabel:
                continue
            else:
                return key
    return "key doesn't exist"

def getBetterLabelPred(cor, colorPredicted):
    valueMax = 0
    keyMax = "key doesn't exist"
    for key, value in colorPredicted.items():
        if cor == key.split('-')[1]:
            if value > valueMax:
                valueMax = value
                keyMax = key.split('-')[0]
    return keyMax



    
def labelClustersResult(trainData, trainLabels, gng_graph, color_dict, testData = None, num_neighbor = 5):

   """Identificación de clústeres formados y su asociación a las etiquetas de las etiquetas reales"""
   dictClusterLabel = labelingClusters(trainData, trainLabels, gng_graph, color_dict, num_neighbor)
   
   if not testData is None:
       listLabelPredicted = prediction(testData, gng_graph, color_dict, num_neighbor)
   else:
       listLabelPredicted = prediction(trainData, gng_graph, color_dict, num_neighbor)
       
   def changeColor(listLabelPredicted):
       # si el pred tiene mas clases que el real se traduce por un valor cualquiera ya que ignoraremos posteriormente este caso
       key = get_key(listLabelPredicted, dictClusterLabel)
       if key == "key doesn't exist":
           
           key =  "NotLabelAsociated"
       return key
    
   if len(dictClusterLabel) != 0 and listLabelPredicted != None:
        listLabelPredicted = list(map(changeColor, listLabelPredicted))

   return dictClusterLabel, listLabelPredicted         
   
def labelingClusters(data, trainLabels, gng_graph, color_dict, num_neighbor = 5):
    
    colorPredicted = dict()

    # Se recorre cada dato de entrada para observar a que clúster pertenece según los 5 más cercanos.
    for i, vector in data.iterrows():
        
        colorVector = NearestClusterToVector(vector, gng_graph, color_dict)
        
        # Si es un conjunto de datos supervisado
        # y el número de clústeres es el mismo que el número de etiquetas
        # se crea la votaciones de las posibles combinaciones.
        if not trainLabels is None:
            a = sorted(colorVector.values())
            name = '{}-{}'.format(str(trainLabels[i]), get_key(a[-1], colorVector))
            if name in colorPredicted:
                colorPredicted[name] += 1
            else:
                colorPredicted[name] = 1
     
    colorPredicted = balanceClustersIdentify(colorPredicted, trainLabels.value_counts())
    
    # print(colorPredicted)
    dictClusterLabel = dict()
    number_Of_labels = len(pd.unique(trainLabels))
    number_Of_clusters = len(pd.unique(list(color_dict.values())))
    done = False
    # Proceso de asociación clústeres-etiqueta si es un conjunto de datos supervisado 
    # y el número de clústeres es el mismo que el número de etiquetas.
    if not trainLabels is None and number_Of_labels == number_Of_clusters:
        b = sorted(colorPredicted.values())
        clustersNotIdentified = number_Of_labels
        # identificar que clusters corresponden a las etiquetas
        for combination in range(1, len(colorPredicted)+1):
            # print("clusters1: ",dictClusterLabel)
            # Se busca la etiqueta que no este ya incluida en dictClusterLabel
            key = get_diffents_key(b[-combination], colorPredicted, dictClusterLabel)
            if key == "key doesn't exist":
                continue
            key = key.split('-')
            if not key[0] in dictClusterLabel:
                if not key[1] in list(dictClusterLabel.values()):
                    dictClusterLabel[str(key[0])] = key[1]
                    clustersNotIdentified -= 1    
                    if clustersNotIdentified == 0:
                        done = True
                        break

        # print("clusters2: ",dictClusterLabel) 
        # Si no se han encontrado combinaciones para las etiquetas-clusteres menos identificados.
        # se relacionan entre ellos.
        if not done:
            for label in pd.unique(trainLabels):
                if str(label) in dictClusterLabel:
                    continue
                else:
                    for clust in pd.unique(list(color_dict.values())):
                        if not clust in list(dictClusterLabel.values()):
                            dictClusterLabel[label] = clust
    # print("clusters3: ",dictClusterLabel)             
    return dictClusterLabel       


def prediction(data, gng_graph, color_dict, num_neighbor = 5):
    
    listLabelsPredicted = list()
    for i, vector in data.iterrows():
        
        colorVector = NearestClusterToVector(vector, gng_graph, color_dict, num_neighbor)
        # Seleccionando el clúster más cercano
        a = sorted(colorVector.values())
        listLabelsPredicted.append(get_key(a[-1], colorVector))
       
    return listLabelsPredicted


def balanceClustersIdentify(combinationColorClusterPredicted, labelsFrequence):
    # for key, frequence in labelsFrequence.items():
    #     print("key: ", key, "ocurrencias: ", frequence)
    # # print(labelsFrequence[2])
    for key, frequence in combinationColorClusterPredicted.items():
        label = int(key.split("-")[0])
        labelTrueFrequence = labelsFrequence[label]
        percentage = int(frequence/labelTrueFrequence * 100)
        combinationColorClusterPredicted[key] = percentage

    # print("combinaciones" , combinationColorClusterPredicted)
    return combinationColorClusterPredicted
    
def NearestClusterToVector(vector, gng_graph, color_dict, num_neighbor = 5):
    colorVector = dict()
    # Calculo distancia euclidiana del dato de entrada a todas las neuronas.
    nodes = gng_graph.nodes
    weights = np.concatenate([node.weight for node in nodes])
    vector = vector.to_numpy()
    distance = np.linalg.norm(weights - vector, axis=1)
        
    # Ordenación de las distancias y seleccionando las 5 más cercanas.
    neuron_ids = np.argsort(distance)
    listaneighborID = neuron_ids[:num_neighbor]
    for IDneighbor in listaneighborID:
        neuron = nodes[IDneighbor]
        # print("color_dict", color_dict)
        # print("neuron name", neuron.name, "color: ", color_dict[neuron.name])
        color = color_dict[neuron.name]
        # votación para saber a que clúster pertenece.
        if color in colorVector:
            colorVector[color] += 1
        else:
            colorVector[color] = 1
    # print("colorVector: ", colorVector)
    
    return colorVector

def toListofStrings(labels):
    if not isinstance(labels, (list, tuple)):
        labels = labels.tolist()  
     # se pasa a sting para que la comparativa se haga con el mismo formato
    labels = [str(e) for e in labels]
    return labels