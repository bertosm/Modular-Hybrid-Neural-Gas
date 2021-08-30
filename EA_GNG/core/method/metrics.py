#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:48:39 2020

@author: Berto Sosa
"""

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, homogeneity_completeness_v_measure, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, fowlkes_mallows_score, normalized_mutual_info_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
# import os.path

# import matplotlib.pyplot as plt

from EA_GNG.core.label import toListofStrings



def prettyConfusionMatrix(labelsTrue, labelsPred, labelsOrdering = None, verbose = False):
   
        
    labelsTrue = toListofStrings(labelsTrue)
    labelsPred = toListofStrings(labelsPred)
    
    unique_labelsTrue = pd.unique(labelsTrue)
    unique_labelsPred = pd.unique(labelsPred)
    
    labelsOrdering = list(set(unique_labelsPred).union(set(unique_labelsTrue)))
    
    #labelsOrdering es de tamaño 1D donde se encuentra el orden para mostrar cada labels en la confusion matrix
    #example: ['CN', 'MCI', 'AD']

    confusionMatrix = confusion_matrix(labelsTrue, labelsPred, labels=labelsOrdering)
    dfCM = pd.DataFrame(data = confusionMatrix, index = labelsOrdering, columns = labelsOrdering)
 
    if verbose:
        print("matriz confusion:\n",dfCM)
        
    return dfCM
    
def calculateClassificationMetrics(labelsTrue, labelsPred, labelsOrdering = None, verbose = False):
    
    labelsTrue = toListofStrings(labelsTrue)
    labelsPred = toListofStrings(labelsPred)
    
    unique_labelsTrue = pd.unique(labelsTrue)
    unique_labelsPred = pd.unique(labelsPred)
    
    labelsOrdering = list(set(unique_labelsPred).union(set(unique_labelsTrue)))
    
    try:
        classificationRep = classification_report(labelsTrue, labelsPred, target_names=labelsOrdering, output_dict=True)
        rf = pd.DataFrame(classificationRep)

        precision = rf.loc['precision', 'weighted avg']
        accuracy = rf.loc['f1-score', 'accuracy']
        recall = rf.loc['recall', 'weighted avg']
        f1Score = rf.loc['f1-score', 'weighted avg']
    except ValueError:
        print("error calculate precision, recall and F1 score")
        precision = precision_score(labelsTrue, labelsPred, average = 'macro')
        accuracy = accuracy_score(labelsTrue, labelsPred)
        recall = recall_score(labelsTrue, labelsPred, average = 'macro')
        if (precision+recall) == 0:
            f1Score = -1
        else:
            f1Score = 2*(precision * recall) / (precision + recall)


    falsos_positivos,verdaderos_positivos,_ = roc_curve(labelsTrue, labelsPred)
        
    if verbose:
        classificationRep = classification_report(labelsTrue, labelsPred, target_names= labelsOrdering, output_dict=False)
        print(classificationRep) 
        
    return accuracy, precision, recall, f1Score, falsos_positivos, verdaderos_positivos
  
   
def evaluateUnsupervisedClusteringQuality(data, labelsPred, seed = 1, verbose = False):
    
    labelsPred = toListofStrings(labelsPred)
    
    try:
        dbs = davies_bouldin_score(data, labelsPred)
    except ValueError:
        dbs = -1
       
    try:
        sil = silhouette_score(data, labelsPred, random_state = seed)
    except ValueError:
        sil = -1
        # print('sil failed')
    
    try:
        calinski = calinski_harabasz_score(data, labelsPred)
    except ValueError:
        calinski = -1
        # print('calinski failed')    
    
    if verbose:
        print('daviesBouldin= ', dbs,'calinski= ', calinski,'silhouette= ', sil)
        
    return dbs, calinski, sil
  
   
def evaluateSupervisedClusteringQuality(data, labelsTrue, labelsPred, seed = 1, verbose = False):
    
    labelsTrue = toListofStrings(labelsTrue)
    labelsPred = toListofStrings(labelsPred)
    
    unique_labelsTrue = pd.unique(labelsTrue)
    unique_labelsPred = pd.unique(labelsPred)
    
    labelsOrdering = list(set(unique_labelsPred).union(set(unique_labelsTrue)))
    
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labelsTrue, labelsPred)
    ari = adjusted_rand_score(labelsTrue, labelsPred)
   
    fowlkes = fowlkes_mallows_score(labelsTrue, labelsPred)
    
    normalizedmutualInfo = normalized_mutual_info_score(labelsTrue, labelsPred)

    def purity_score(labelsTrue, labelsPred):
        confusionMatrix = confusion_matrix(labelsTrue, labelsPred, labels = labelsOrdering)
        return np.sum(np.amax(confusionMatrix, axis = 0))/np.sum(confusionMatrix)
    purity = purity_score(labelsTrue, labelsPred)
    
    
    if verbose:
        print( 'homogeneity= ', homogeneity, 'completeness= ', completeness, 'v_measure= ', v_measure, 'adjustedRand= ', ari,
              'normalizedmutualInfo= ', normalizedmutualInfo,'fowlkes= ', fowlkes,'purity= ',purity)
        
    return homogeneity, completeness, v_measure, ari, normalizedmutualInfo, fowlkes, purity
  

def saveClassificationMetrics(count, saving_path, metrics):
    
     if len(count) == 1 and count[0] == '1':
         pass

def saveSupervisedClusteringMetrics(count, saving_path, metrics):
    
    if len(count) == 1 and count[0] == '1':
        pass

def saveUnsupervisedClusteringMetrics(count, saving_path, metrics):
    
    if len(count) == 1 and count[0] == '1':
        pass

def saveMetrics(count, saving_path, metrics, sTitle=None, timeNeupyGNG=None, n_clusters=None, limit = None, bestAcc=-1):
    
    if timeNeupyGNG == None:
       saveMetricsPerceptron(count, saving_path, metrics,sTitle, limit, bestAcc)
       return
    
     #iniciar el guardado de las figuras, metricas, clusters encontrados, etc.
    if len(count) == 1 and count[0] == '1':
        # Creando ficheros resultantes, si existen previamente son reescritos.
        t = open("{}config_param_time.txt".format(saving_path), "w") #fichero tiempo de ejecución
        c = open("{}config_param_cluster.txt".format(saving_path), "w") #fichero cluste, número de cluster encontrados
        mrn = open("{}config_param_metricsReportNeupy.txt".format(saving_path), "w") #fichero metrica resultante
        cqrn = open("{}config_param_clusteringQualityReportNeupy.txt".format(saving_path), "w") #fichero quality resultante
         # Encabezados de los ficheros resultantes.
        t.write('config;params;timeGNG\n')
        c.write('config;params;clusters_neupy\n')
        mrn.write('config;params;accuracy;precision;recall;f1Score\n')
        cqrn.write('config;params;homogeneity;completeness;v_measure;adjustedrand;daviesbouldin;normalizedmutualInfo;fowlkes;calinski;purity;silhouette;bestSilhouette;bestCalinski\n')
        
        
       
    else:
        #Abriendo ficheros resultantes, previamente existentes, si no existen los crea.
        t = open("{}config_param_time.txt".format(saving_path), "a")
        c = open("{}config_param_cluster.txt".format(saving_path), "a")
        mrn = open("{}config_param_metricsReportNeupy.txt".format(saving_path), "a")
        cqrn = open("{}config_param_clusteringQualityReportNeupy.txt".format(saving_path), "a")
    
        
    #guardar los parametros
    t.write("config {};{}".format(count,sTitle.replace('\n',' ')))
    c.write("config {};{}".format(count,sTitle.replace('\n',' ')))
    mrn.write("config {};{}".format(count,sTitle.replace('\n',' ')))
    cqrn.write("config {};{}".format(count,sTitle.replace('\n',' ')))
    
    #guardar los tiempo de ejecucion.
    t.write(";tiempo GNG neupy:{}\n".format(timeNeupyGNG))

    #guardar el número de cluster encontrados.
    c.write(";clustersNeupy:{}\n".format(n_clusters))
    
    # guardar las métricas obtenidas.
    mrn.write(";accuracy:{};precision:{};recall:{};f1Score:{}\n".format(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1Score']))
    
    #guardar las metricas de calidad obtenidas, quality.
    cqrn.write(";homogeneity:{};completeness:{};v_measure:{};adjustedrand:{};daviesbouldin:{};normalizedmutualInfo:{};fowlkes:{};calinski:{};purity:{};silhouette:{};bestSilhouette:{};bestCalinski:{}\n".format(metrics['homogeneity'], metrics['completeness'], metrics['v_measure'], metrics['ari'], metrics['dbs'], metrics['normalizedmutualInfo'], metrics['fowlkes'], metrics['calinski'], metrics['purity'], metrics['sil'], metrics['bestSilhouette'], metrics['bestCalinski'] ))
    
    #cerrar la escritura en fichero
    t.close()
    c.close()
    mrn.close()
    cqrn.close()
    
def saveMetricsPerceptron(count, saving_path, metrics, sTitle = None, limit = None, activationNeigbors=None, bestAcc= -1):
   

    falsoPositivo =  metrics["falsos_positivos"][1] #Array [0, FalsoPositivo, 1]
    verdaderoPositivo = metrics["verdaderos_positivos"][1] #Array [0, VerdaderoPositivo, 1]
    
    if limit==1:

        mrn = open("{}AN{}-config_param_metricsReportNeupy.txt".format(saving_path, activationNeigbors), "w") #fichero metrica resultante
        mrn.write('config;param;limit;accuracy;precision;recall;f1Score;falsosPositivos;verdaderosPositivos\n')
        
        rocFile= open("{}AN{}-final-RocCurve.csv".format(saving_path, activationNeigbors), "w") #fichero curve Roc
        rocFile.write('limit;falsosPositivos;verdaderosPositivos\n')        
        
    else:
        
        mrn = open("{}AN{}-config_param_metricsReportNeupy.txt".format(saving_path, activationNeigbors), "a")
        rocFile= open("{}AN{}-final-RocCurve.csv".format(saving_path, activationNeigbors), "a") #fichero curve Roc
    
    mrn.write("config {};{};limit:{}".format(count, sTitle.replace('\n',' '), limit))
    
    # guardar las métricas obtenidas.
    mrn.write(";accuracy:{};bestAcc:{};precision:{};recall:{};f1Score:{};auc:{};falsosPositivos:{};falsosV:{}\n".format(metrics['accuracy'], bestAcc, metrics['precision'], metrics['recall'], metrics['f1Score'], metrics["auc"], metrics["falsos_positivos"],metrics["verdaderos_positivos"] ))
    mrn.close()
    
    rocFile.write('{};{};{}\n'.format(limit,falsoPositivo, verdaderoPositivo))
    rocFile.close()
    
    
   
       
    