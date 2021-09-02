# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:46:34 2020

@author: Berto Sosa
"""


from os import makedirs, path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import sys
from joblib import Parallel, delayed


from EA_GNG.core.dataset import removeNan, imputerData, loadOneDataset, ls, saveDatasets, modifyLabels, escalarValues, makeDatasetDict, takeTypeDataScaling, splitDataset, loadDataset_partitionared_CN_MCI_AD_fromPKL
from EA_GNG.core.figure import createFigures, saveFigures, maketitle, checkColorListAndColorRange, showdata
from EA_GNG.core import gng

from EA_GNG.core.method.featureSelection import featureSelectionMethods
from EA_GNG.core.method.statistics import isPaired, isParametric, selectStatistic
from EA_GNG.core.method.metrics import evaluateSupervisedClusteringQuality, evaluateUnsupervisedClusteringQuality, saveUnsupervisedClusteringMetrics, saveSupervisedClusteringMetrics


# -------------------------- PREPROCESING ADNI ----------------------------------------------
def preprocesingAdni(filePath,
                     savingPath,
                     label='DX_bl', listFeatures=list(), 
                     dropAllCDR=False, dropSMC=False, classifProblem = '', listImputer = ('notImputer', )):
    
    limitPath = filePath.rfind("/")
    df = loadOneDataset(filePath[:limitPath+1], filePath[limitPath+1:])

    #obligamos a ser lista para utilizar las operaciones, append and remove
    listFeatures = list(listFeatures)
    
    #obligamos a que la etiqueta aparezca al final 
    if label in listFeatures:
        listFeatures.remove(label)
    
    #comprobamos que la lista no solo posea la etiqueta
    if len(listFeatures)!=0:
        listFeatures.append(label)
    
    df = df.drop(['VISCODE'],axis=1)
    #eliminamos los campos de etiqueta exceptuando el que queremos
    dx_list = ['DX_bl','DX','DXCURREN','DIAGNOSIS','DXCHANGE']
    dx_list.remove(label)
    df = df.drop(dx_list, axis=1)
    # df = df.drop(['RID'], axis=1)
 

    #eliminamos los campos vacios del campo etiqueta
    df = df.dropna(subset=[label])
    
    # pasar la columna 0 (target) a la ultima columna para su proximo uso en featureSelection
    mask = df.columns.values != label
    dfall = df.loc[:, mask]
    dflabel = df[label]
    df = pd.concat([dfall,dflabel], axis = 1)
    
    #modificar la etiqueta a numeros
    df[label] = df[label].replace({'CN': 0, 'EMCI': 1, 'LMCI': 2, 'AD':3 ,'SMC':4})

    #eliminamos los casos extremos de alzheimer
    if dropSMC:
        df = df[df[label] != 4]
    
    #dividimos el dataset en varios datasets segun el estudio al que pertenecen
    dfADNI1 = df[df['Phase'] == 'ADNI1']
    dfADNI2 = df[df['Phase'] == 'ADNI2']
    dfADNIGO = df[df['Phase'] == 'ADNGO']
    # dfADNIGO = None
    dfADNI3 = df[df['Phase'] == 'ADNI3']
    df = df[df['Phase'] != None]
    
    #eliminamos el campo Phase, tras haber sido divididos
    dfADNI1 = dfADNI1.drop(['Phase'],axis=1)
    dfADNI2 = dfADNI2.drop(['Phase'],axis=1)
    dfADNIGO = dfADNIGO.drop(['Phase'],axis=1)
    dfADNI3 = dfADNI3.drop(['Phase'],axis=1)
    
    #creamos el dataset especifico, si la lista ha sido establecida
    if len(listFeatures) != 0:
        df = df[[element for element in listFeatures]]
        dfADNI1 =dfADNI1[[element for element in listFeatures]]
        dfADNI2 =dfADNI2[[element for element in listFeatures]]
        dfADNIGO =dfADNIGO[[element for element in listFeatures]]
        dfADNI3 = dfADNI3[[element for element in listFeatures]]
        
        #eliminamos los campos CDR si asi ha sido solicitado
    if dropAllCDR:
        try:
            df = df.drop(['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL'],axis=1)
            dfADNI1 = dfADNI1.drop(['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL'],axis=1)
            dfADNI2 = dfADNI2.drop(['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL'],axis=1)
            dfADNIGO = dfADNIGO.drop(['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL'],axis=1)
            dfADNI3 = dfADNI3.drop(['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL'],axis=1)
            print('Se han eliminados todos los features CDR (dropAllCDR = True)')
                    
        except KeyError:
            print('ERROR- Si quiere usar features CDR en la lista específica, deshabilite dropAllCDR')
        

    #eliminamos las columnas nulas    
    df = removeNan(df)
    dfADNI1 = removeNan(dfADNI1)
    dfADNI2 = removeNan(dfADNI2)
    dfADNIGO = removeNan(dfADNIGO)
    dfADNI3 = removeNan(dfADNI3)
      
    # eliminamos las filas completamente nulas
    dfADNI1 = dfADNI1.dropna(how='all')
    dfADNI2 = dfADNI2.dropna(how='all')
    dfADNIGO = dfADNIGO.dropna(how='all')
    dfADNI3 = dfADNI3.dropna(how='all')

    
    # imputamos los valores de los datasets
    if len(listImputer) != 0:
        for imputer in listImputer:
            savingPathAux = savingPath + imputer + '/'
            dfphase = df['Phase']
            df = df.drop(['Phase'], axis = 1)
            dfaux = imputerData(df, imputer, label)
            dfaux['Phase'] = dfphase
            dfADNI1aux = imputerData(dfADNI1, imputer, label)
            dfADNI2aux = imputerData(dfADNI2, imputer, label)
            dfADNI3aux = imputerData(dfADNI3, imputer, label)
            
            # modificamos los labels según el caso de estudio que estamos realizando
            dfaux = modifyLabels(dfaux, classifProblem, label)
            dfADNI1aux = modifyLabels(dfADNI1aux, classifProblem, label)
            dfADNI2aux = modifyLabels(dfADNI2aux, classifProblem, label)
            dfADNI3aux = modifyLabels(dfADNI3aux, classifProblem, label)
            
            saveDatasets(savingPathAux, listFeatures = (len(listFeatures) != 0), dropAllCDR = dropAllCDR,
                        df=dfaux, dfADNI1=dfADNI1aux, dfADNI2=dfADNI2aux, dfADNI3=dfADNI3aux, classifProblem = classifProblem)
    else:
        print("ERROR, es necesario especificar método de imputación. Si no se requiere utilizar especificar NotImputer en la lista.")
    
    
   
# -------------------------- LOAD DATASETS ----------------------------------------------
def loadDatasets(files_path = 'C:/Users/comciencia/Desktop/AlbertoSosaPE/Tareas/Python/datasets/', listMethod = ("unscaled", ), concretFile = None):
    """cargamos los datos de cada fichero y los agregamos a un diccionario con todos los datasets"""
    #iniciar el diccionario
    loadedDatasets = dict()
    #coger todos los ficheros que se encuentra en la dirección pasada
    files = ls(files_path)
        
    for file in files: 
        indexpoint = file.rfind('.')
        
        #si el fichero no tiene extension, evitarlo
        if indexpoint == -1:
            continue
        elif not file[indexpoint:] in ('.xlsx','.csv'):
            continue
        
        #Abrir un único fichero
        if concretFile != None and concretFile == file:
            for method in listMethod:
                loadedDatasets["{}:>{}".format(method,file)]= escalarValues(loadOneDataset(files_path,file), method)
        elif concretFile == None:
            for method in listMethod:
                loadedDatasets["{}:>{}".format(method,file)]= escalarValues(loadOneDataset(files_path,file), method)
    return loadedDatasets


        
   
# -------------------------- FEATURE SELECTION ---------------------------------------------
def featureSelection(loadedDatasets, savingPath, method = "both",
                         colorList=["#C1182A", "#FBAD3C", "#E8E64C", "#ABD715", "#0CA299", "#75117E", "#F02AA2", "#F97513", "#0665EE", "#7FDF65"], 
                         colorRange=["0.8-1.0", "0.6-0.8", "0.4-0.6","0.2-0.4","0.0-0.2"],
                         bestFeatures=5, threshold=0.2, seed=1, saveFile=None):
    
    
    colorList, colorRange = checkColorListAndColorRange(colorList, colorRange)
    
    savingPathdir = '/'.join(savingPath.split('/')[:-1])
    if not path.isdir(savingPathdir):
        makedirs(savingPathdir)
    
        
    fileSavingPath = savingPath + "featureSelection.txt"
    
    if saveFile:
        t = open(fileSavingPath, "w")
        t.write("nombre_dataSet;imputed_method;data_shape;label_shape;fcbf;XGBOOST_Xtrain;XGBOOST_Ytrain;XGBOOST_Xtest;XGBOOST_Ytest\n")
    else:
        t = None
            
    #pasar el df a diccionario para no tocar el resto del código.
    if isinstance(loadedDatasets, pd.DataFrame):
        aux = dict()
        aux['dataset'] = loadedDatasets
        loadedDatasets = aux
        
    for i, dataset in zip(range(len(loadedDatasets)), loadedDatasets):
        
        
        print('\n>>>>>>',dataset, '<<<<<<')

        if t != None:
            t.write("{};{}".format(dataset, fileSavingPath.split('/')[-2]))
            
        # barajando los datos
        loadedDatasets[dataset] = loadedDatasets[dataset].sample(frac=1, random_state = seed)
        
       
        savingPathSave = savingPath + "featureSelection-" + dataset
            
        #call featureSelection
        featureSelectionMethods(loadedDatasets[dataset].iloc[:, :loadedDatasets[dataset].shape[1]-2], 
                         loadedDatasets[dataset].iloc[:, loadedDatasets[dataset].shape[1]-1], savingPathSave, method,
                         colorList, colorRange, seed, threshold, bestFeatures, t)
        if t!=None:
            t.write("\n")
    
    if t != None:
        t.close()

    
        
   
# -------------------------- STATISTICS ----------------------------------------------
def statistics(loadedDatasets, savingPath = None, target = 'DX_bl', allPaired = None, alpha = 0.05, verbose = False):
    
    if savingPath != None:
        if not path.isdir(savingPath):
            makedirs(savingPath, exist_ok = True)
        
        t = open("{}statistics.txt".format(savingPath), "w") #fichero tiempo de ejecución
        t.write("Dataset;feature;parametric;paired;test;stat;pvalue;resolution;alpha\n")
    else:
        t = None
        
    #pasar df de entrada a diccionario.
    if isinstance(loadedDatasets, pd.DataFrame):
        aux = dict()
        aux['dataset'] = loadedDatasets
        loadedDatasets = aux
    
    for dataset in loadedDatasets:
        if allPaired == None:
            paired = None
            while(paired == None):
                paired = isPaired()
        elif not isinstance(allPaired, bool):
            print("error allPaired cant be: {}\n".format(type(allPaired)))
            return
        else:
            paired = allPaired
        
        """prueba de todas las características"""
        parametric = isParametric(loadedDatasets[dataset])
        name, stat, pvalue, resolution = selectStatistic(loadedDatasets[dataset], target, parametric, paired, alpha)
        if verbose:
            print("{};{};{};{};{};{};{};{};{}\n".format(dataset,"ALLFEATURES",parametric,paired,name,stat,pvalue,resolution, alpha))
        if savingPath != None: 
            t.write("{};{};{};{};{};{};{};{};{}\n".format(dataset,"ALLFEATURES",parametric,paired,name,stat,pvalue,resolution, alpha))
    
        """prueba de cada característica en particular"""
        df = loadedDatasets[dataset]
        for feature in df.columns:
            if feature == target:
                continue
            dffeature = df[[feature, target]]
            parametric = isParametric(dffeature)
            name, stat, pvalue, resolution = selectStatistic(dffeature, target, parametric, paired, alpha)
            if verbose:
                print("{};{};{};{};{};{};{};{};{}\n".format(dataset,feature,parametric,paired,name,stat,pvalue,resolution, alpha))
            if savingPath != None: 
                t.write("{};{};{};{};{};{};{};{};{}\n".format(dataset,feature,parametric,paired,name,stat,pvalue,resolution, alpha))
    
    if savingPath != None:
        t.close()
                
        
 

        
# -------------------------- GROWING NEURAL GAS ----------------------------------------------    
def growingNeuralGas(param_dict, df, saving_path, target = "DX_bl", labelsOrdering = None, verbose = False, nameDataset = 'NoNamedDataset', saveProcess = False):
   
    np.random.seed(param_dict['seed'])
    random.seed(param_dict['seed'])
    tf.random.set_random_seed(param_dict['seed'])
    tf.set_random_seed(param_dict['seed'])
    
    if not 'count' in param_dict:
        param_dict['count'] = 1;
        
    if not path.isdir(saving_path):
        makedirs(saving_path, exist_ok = True)
    
    if isinstance(df, dict):
        trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY = loadDataset_partitionared_CN_MCI_AD_fromPKL(df["filePath"], df["fileName"], df["num_components"], df["scaled"], df["projection"])
    else:
        trainDataX, trainLabelsTrueY, testDataX, testLabelsTrueY = splitDataset(df, 0.20, target, param_dict['seed'])
    
    print("size training data: ", trainDataX.shape[0])
    print("size testing data: ", testDataX.shape[0])
    print("size training data: ", trainLabelsTrueY.shape[0])
    print("size testing data: ", testLabelsTrueY.shape[0])
    param_dict['n_features'] = trainDataX.shape[1]

    if not saveProcess:
        
        sTitle = maketitle(param_dict, nameDataset)
        fig3d, fig, ax1, ax2 = createFigures(nameDataset, param_dict["typeDataScaling"], sTitle, param_dict['n_features'])
        columns = df.columns.tolist()
        showdata(trainDataX,trainLabelsTrueY, ax1, fig3d, columns, nameDataset)
    
    else:
        sTitle = ""
        fig3d, fig, ax1, ax2 = None, None, None, None
        
    timeNeupyGNG, n_clusters, labelsPred, bestCalinski, bestSilhouette = gng.neupy_growingneuralgas(trainDataX, param_dict, ax2, fig3d, trainLabelsY=trainLabelsTrueY, testDataX = testDataX, testLabel=testLabelsTrueY, saveProcess=saveProcess, saving_path = saving_path)
  
    print("best Calinski founded: ", bestCalinski)
    print("best Silhouette founded: ", bestSilhouette)
    
    
    
    if not saveProcess:
        # saveFigureLabelsPred(testDataX, testLabelsTrueY, labelsPred, saving_path, param_dict["count"], sTitle)
        metrics = dict()
        
        # Not supervised metrics at end training model.
        metrics['dbs'], metrics['calinski'], metrics['sil'] = evaluateUnsupervisedClusteringQuality(testDataX, labelsPred, param_dict['seed'], verbose)
        saveUnsupervisedClusteringMetrics(saving_path, count = param_dict["count"], calinski = metrics['calinski'], silhouette = metrics['sil'], dbs = metrics["dbs"])

        # Supervised metrics at end training model.
        metrics['homogeneity'], metrics['completeness'], metrics['v_measure'], metrics['ari'], metrics['normalizedmutualInfo'], metrics['fowlkes'], metrics['purity'] = evaluateSupervisedClusteringQuality(testDataX, testLabelsTrueY, labelsPred, 
                                                                                                                                                                                                             param_dict['seed'], verbose)
        saveSupervisedClusteringMetrics(saving_path, metrics = metrics, count = param_dict["count"])
    
        ax2.set_title('GNG_NEUPY_PACKAGE - Nº clusters= {}'.format(n_clusters))
        saveFigures(fig3d, fig, saving_path, param_dict)
        
    else:
        saveUnsupervisedClusteringMetrics(saving_path, count = param_dict["count"], calinski = bestCalinski, silhouette = bestSilhouette, dbs = "notCalculated")
 
    
        
    # # sacamos las metricas
    # if labelsPred is None:
    #     metrics = dict()
    #     metrics["bestCalinski"] = bestCalinski
    #     metrics["bestSilhouette"] = bestSilhouette
    #     # metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1Score'] = -10,-10,-10,-10
    #     # metrics['homogeneity'], metrics['completeness'], metrics['v_measure'], metrics['ari'], metrics['dbs'], metrics['normalizedmutualInfo'], metrics['fowlkes'], metrics['calinski'], metrics['purity'], metrics['sil'] = -10,-10,-10,-10,-10,-10,-10,-10,-10,-10
    #     # saveMetrics(param_dict['count'], saving_path, sTitle, timeNeupyGNG, n_clusters, metrics)
        
    # else:
    #     # saveFigureLabelsPred(testDataX, testLabelsTrueY, labelsPred, saving_path, param_dict["count"], sTitle)
    #     # dfCM = prettyConfusionMatrix(testLabelsTrueY, labelsPred, labelsOrdering, verbose = verbose)
    #     metrics = dict()
    #     metrics["bestCalinski"] = bestCalinski
    #     metrics["bestSilhouette"] = bestSilhouette
    #     # metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1Score'] = calculateClassificationMetrics(testLabelsTrueY, labelsPred, labelsOrdering, verbose)
    #     # metrics['homogeneity'], metrics['completeness'], metrics['v_measure'], metrics['ari'], metrics['dbs'], metrics['normalizedmutualInfo'], metrics['fowlkes'], metrics['calinski'], metrics['purity'], metrics['sil'] = evaluateClusteringQuality(testDataX, testLabelsTrueY, labelsPred, param_dict['seed'], verbose)
    #     # saveMetrics(param_dict['count'], saving_path, metrics, sTitle, timeNeupyGNG, n_clusters)
    #     # confusion_matrixSavePath = '{}config{}-confusion_matrix-{}.csv'.format(saving_path, param_dict['count'], param_dict['typeDataScaling'])
    #     # dfCM.to_csv(confusion_matrixSavePath)
    
    # ax2.set_title('GNG_NEUPY_PACKAGE - Nº clusters= {}'.format(n_clusters))
    
    # saveFigures(fig3d, fig, saving_path, param_dict)
 
    
    
        
   
# -------------------------- LOOP GROWING NEURAL GAS ----------------------------------------------        
def loopGrowingNeuralGas_perceptron(loopDict, saving_path, loadedDatasets = None, PCA=False, PCA_n_components = 3, list_distance = (3.0, ), list_number_samples = (333, ), labelsOrdering = None, verbose = False, shuffle_data = True, seed = 1, saveProcess = False, savedGNG=False, hibrid=True):
    
    print("datasets: ", loadedDatasets.keys())
    if "pkl" in loadedDatasets.keys():
        print(loadedDatasets["pkl"])
    print("ejecutando con las siguientes condiciones:\n", "savingPath:> {}\nPCA:> {} -//- NºPCA:> {}\nhibrid:> {} -//- savedGNG:> {} -//- saveProcess:> {}".format(saving_path, PCA, PCA_n_components, hibrid, savedGNG, saveProcess))
    print("are you sure?")
    print("The path already exist? ", path.isdir(saving_path))
    print("introduce ok or yes to continue:")
    itsSure = input()
    if itsSure not in ['yes', 'y', 'ok']:
        print("stop running..")
        sys.exit()
    #Si no hay dataset pasado por parámetro crearlo.
    if loadedDatasets == None:
        if len(list_distance) == 0 or len(list_number_samples) == 0:
            print("error en parametros; introducir conjunto de datos o parámetros para crearlo\n")
            return None
        else:
            loadedDatasets = makeDatasetDict(list_number_samples, list_distance, shuffle_data, seed)
    
    #pasar el df a diccionario para no tocar el resto del código.
    if isinstance(loadedDatasets, pd.DataFrame):
        aux = dict()
        aux['dataset'] = loadedDatasets
        loadedDatasets = aux
        
        
    #Si el ´número de configuraciones es 8 o más se ejecutarán paralelamenta las configuraciones (8 por el número de procesadores).
    #Si no se ejecutará paralelamente el diccionario con más objetos.
    if len(loopDict) >=8 or len(loopDict) >= len(loadedDatasets):
        print("lanzando paralelismo priorizando configuraciones\n")
        print("configuraciones = {}, conjunto de datos = {} ->> lanzamientos: {}\n".format(len(loopDict), len(loadedDatasets), len(loopDict) * len(loadedDatasets)))
        for dataset in loadedDatasets:
            #paralel llama a loop_gng
            Parallel(n_jobs=1)(delayed(gng.loop_gng)(loopDict[config], saving_path, loadedDatasets[dataset], PCA, PCA_n_components, hibrid, takeTypeDataScaling(dataset), labelsOrdering, dataset, seed, shuffle_data, verbose, savedGNG=savedGNG, saveProcess=saveProcess) for config in loopDict)
    else:
        print("lanzando paralelismo priorizando conjunto de datos\n")
        for config in loopDict:
            Parallel(n_jobs=1)(delayed(gng.loop_gng)(loopDict[config], saving_path, loadedDatasets[dataset], PCA, PCA_n_components, hibrid, takeTypeDataScaling(dataset), labelsOrdering, dataset, seed, shuffle_data, verbose, savedGNG=savedGNG, saveProcess=saveProcess ) for dataset in loadedDatasets)       
        
        

   
# -------------------------- MAKE CONFIG DICT ----------------------------------------------
def makeConfigDict(list_epoch = (100, ),list_max_age = (25, ),list_lambda = (45, ),list_max_nodes = (200, ), 
                   list_step = (0.2, ), list_neighbour_step = (0.05, ), list_start_nodes = (2, ), 
                   list_error_decay = (0.99, ), list_error_decay_afterSplit = (0.5, ),
                   list_min_distance = (0.0, ), list_learningRate=(0.2, ), list_epochPerceptron=(50, ), 
                   list_neighborsActivation=(1, ), nameConfig = ''):
 
    #creamos el conjunto de configuraciones en un diccionario para realizar el paralelismo de pruebas
    
    dictAll = dict()
    
    count = 1
    for config in itertools.product(list_step, list_neighbour_step, list_error_decay, list_error_decay_afterSplit,
                               list_min_distance, list_start_nodes, list_max_age, list_lambda,
                               list_max_nodes, list_epoch, list_neighborsActivation, list_learningRate, list_epochPerceptron):

        #El parametro name permite añadir al identificador config+count una string para especificar la ejecución.
        namefig = str(count) + nameConfig
        # previous version (all fors concatenated):
        # dictAll["config{}".format(count)] = (epochs,max_nodes,lambda_,max_age,
        #                                      start_nodes, min_distance,
        #                                      error_decay_afterSplit, error_decay,
        #                                      neighbour_step, step, neighbour_activation, learningRate, epochPerceptron, namefig)
        dictAll["config{}".format(count)] = (config[9],config[8],config[7],config[6],
                                             config[5], config[4],
                                             config[3], config[2],
                                             config[1], config[0], config[10], config[11], config[12], namefig)
        count +=1

    
    return dictAll