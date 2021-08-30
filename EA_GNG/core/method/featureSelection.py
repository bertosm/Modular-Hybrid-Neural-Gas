#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:48:38 2020

@author: Berto Sosa
"""
from collections import Counter

import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from EA_GNG.core.method.fcbf import fcbf

def featureSelectionMethods(dataset_data, dataset_target, savingPath, method = "both", 
                     colorList=["#C1182A", "#FBAD3C", "#E8E64C", "#ABD715", "#0CA299", "#75117E", "#F02AA2", "#F97513", "#0665EE", "#7FDF65"],
                     colorRange=["0.8-1.0", "0.6-0.8", "0.4-0.6","0.2-0.4","0.0-0.2"], seed = 1, threshold = 0, bests_features = 5, t = None): 
    print('\n--------------------------------')
    print('--------------------------------\n')
    if(t != None):
        t.write(";{};{}".format(dataset_data.shape, Counter(dataset_target)))

        
    #por si no se ha usado el preprocesado ADNI.py, elimina las columnas etiquetas sobre el dataset 
    dx_list = ['DX_bl','DX','DXCURREN','DIAGNOSIS','DXCHANGE']
    for feature in dx_list:
        if feature in dataset_data.columns:
            dataset_data.drop([feature],axis=1)

    if len(dataset_target.unique()) < 2:
        print('el dataset introducido no tiene etiqueta o la etiqueta es un valor único. Los métodos de feature selection desarrollados necesitan 2 o más valores de etiqueta diferentes')
        return
    
    if method in ['both', 'fcbf']:
        print('\n---->fcbf')
        fcbffunc(dataset_data, dataset_target, colorList, colorRange, savingPath, bests_features, threshold, t)
 
#     """    no permite con valores categoricos ValueError: DataFrame.dtypes for data must be int, float or bool """
    if method in ['both', 'xgboost']:
        print('\n---->xgboost')
        xgboostclassifier(dataset_data, dataset_target, colorList, colorRange, savingPath, bests_features,threshold,seed, t)



        
def fcbffunc(dataset_data, dataset_target, colorList, colorRange, savingPath, bests_features = 5, thresh = 0.0, t=None):
    
    #fcbf feature selection
    fig = plt.figure(1)
    if(t != None):
        t.write(";{}".format(Counter(dataset_target)))
        
    fcbfresult = fcbf(dataset_data,dataset_target,thresh)
    if(isinstance(fcbfresult, int)):
            return
        
    print("fcbfResult: ", fcbfresult)
    listfeature = list()
    colorfeature = list()
    
    #coger las caracteristicas principales
    
    fcbfresultPrincipales = fcbfresult[:bests_features,:]
    
    for i in range(fcbfresultPrincipales.shape[0]):
        feature = dataset_data.columns[fcbfresultPrincipales[i,1]]
        listfeature.append(feature)
        
        for j in range(len(colorRange)):
            rango = colorRange[j]
            # print('rango', rango)
            if '-' in rango:
                rango = rango.split("-")
                # print('inicio del rango', rango[0])
                if(fcbfresultPrincipales[i,0] > float(rango[0])):
                    colorfeature.append(colorList[j])
                    break
                
            else:
                # print("no rango")
                if(fcbfresultPrincipales[i,0] == float(rango)):
                 colorfeature.append(colorList[j])
                 break
                          
            
    print("principales features FCBF:", listfeature)
    
 
    
    #plot de las caracteristicas seleccionadas
    y_values = range(fcbfresultPrincipales.shape[0])
    

    
    for i in range(len(colorList)):
        color2 = "vacio"
        try:
            color = colorfeature.index(colorList[i])
        except:
            continue
            
        for j in range(i+1,len(colorList)):
            # print(j)
            try:
                color2 = colorfeature.index(colorList[j])
                break
            except:
                continue
                   
        
        total = fcbfresultPrincipales.shape[0]
        ceros = np.zeros(total).tolist()
        
        if(color2 == 'vacio'):
            featuresClass =  fcbfresultPrincipales[color:,0].tolist()
            fcbfPart = featuresClass[::-1]
            fcbfPart.extend(ceros[:color])
            plt.barh(range(fcbfresultPrincipales.shape[0]),fcbfPart, color = colorfeature[color], label = colorRange[i])
        else:
            fcbfPart = ceros[color2:]
            featuresClass = fcbfresultPrincipales[color:color2,0]
            fcbfPart.extend(featuresClass[::-1])
            fcbfPart.extend(ceros[:color])
            plt.barh(range(fcbfresultPrincipales.shape[0]),fcbfPart, color = colorfeature[color], label = colorRange[i])
        
        
    if(len(y_values) > 20):
         plt.yticks(y_values, listfeature[::-1], fontsize= 6)
    elif(len(y_values) > 10):
         plt.yticks(y_values, listfeature[::-1], fontsize = 8)
    else:
         plt.yticks(y_values, listfeature[::-1], fontsize = 11)
        
    plt.xlabel("Feature Ranking by FCBF")
    plt.legend()
    
    #guardar la figura resultante
    savingPathSave = savingPath + '-FCBF.png'
    plt.savefig(savingPathSave, bbox_inches = 'tight')
    plt.close(fig)
  
    

    
def xgboostclassifier(dataset_data,dataset_target, colorList, colorRange, savingPath, numbers_bests_features, thresh = 0.0, seed = 1, t = None):
    # split data into train and test sets
    fig = plt.figure(2)
    X_train, X_test, y_train, y_test = train_test_split(dataset_data, dataset_target, test_size=0.33, random_state=seed)
    
    if(t != None):
        t.write(";{};{};{};{}".format(X_train.shape,Counter(y_train),X_test.shape,Counter(y_test)))
        
    print("xtrain: ", len(X_train), "ytrain: ", len(y_train), "xtest: ", len(X_test), "ytest: ", len(y_test))
    print('y_train: ', Counter(y_train))
    print('y_test: ', Counter(y_test))

    # fit model on all training data
    model = XGBClassifier(random_state=seed, booster = 'gbtree')
    """ scale_pos_weight = scalePosWeight, eval_metric = 'auc' """
    model.fit(X_train, y_train)

    feature_importances_ = np.sort(model.feature_importances_)
    feature_importances_ = feature_importances_[::-1]
    print("principales features XGBoost: ",feature_importances_)

    positions = np.argsort(model.feature_importances_)
    positions = positions[::-1]
    colorfeature = list()
    
    if(feature_importances_[0] == 0.0):
        
        return
    
    for i in range(len(feature_importances_)):
        
        if(feature_importances_[i] < thresh or feature_importances_[i] == 0.0):
            if (i < numbers_bests_features):
                numbers_bests_features = i
                break
            
        
        for j in range(len(colorRange)):
            rango = colorRange[j]
            if '-' in rango:
                rango = rango.split("-")
                if(feature_importances_[i] > float(rango[0])):
                    colorfeature.append(colorList[j])
                    break
                
            else:
                if(feature_importances_[i] == float(rango)):
                    colorfeature.append(colorList[j])
                    break
             
    
    for i in range(len(colorList)):
        color2 = "vacio"
        try:
            color = colorfeature.index(colorList[i])
        except:
            continue

        for j in range(i+1,len(colorList)):
            try:
                color2 = colorfeature.index(colorList[j])
                break
            except:
                continue
        
        total = len(feature_importances_[:numbers_bests_features])
        ceros = np.zeros(total).tolist()
        
        if(color2 == 'vacio'):
            featuresClass = feature_importances_[color:numbers_bests_features].tolist()
            xgbPart = featuresClass[::-1]
            xgbPart.extend(ceros[:color])     
            plt.barh(range(len(feature_importances_[:numbers_bests_features])), xgbPart, color = colorfeature[color], label = colorRange[i])

        else:

            xgbPart = ceros[color2:numbers_bests_features]
            featuresClass = feature_importances_[color:color2]
            xgbPart.extend(featuresClass[::-1])
            xgbPart.extend(ceros[:color])
            plt.barh(range(len(feature_importances_[:numbers_bests_features])), xgbPart, color = colorfeature[color], label = colorRange[i])

    
    #plot the results
    y_values = range(len(feature_importances_[:numbers_bests_features]))
    etiquetas = list(dataset_data.columns[positions[:numbers_bests_features]])

    if(len(colorfeature) > 20):
        plt.yticks(y_values, etiquetas[::-1], fontsize=11)
    elif(len(colorfeature) > 10):
        plt.yticks(y_values, etiquetas[::-1],  fontsize=11)
    elif (len(colorfeature) > 0):
        plt.yticks(y_values, etiquetas[::-1], fontsize=11)
    else:
        plt.yticks(y_values, etiquetas[::-1],  fontsize=11, visible = False)

   
    plt.xlabel("Feature Ranking by XGBoost")
    plt.legend()
    #guardar la figura resultante
    savingPathSave = savingPath + '-XGBoost.png'
    plt.savefig(savingPathSave, bbox_inches = 'tight')
    plt.close(fig)

