#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:12:38 2020

@author: Berto Sosa
"""
from os import getcwd, scandir, makedirs, path

import pandas as pd
import numpy as np


from joblib import load
from os.path import join

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as scl
from sklearn.impute import SimpleImputer


def loadcsvdata(path,file):
    filepath = path + file
    data = pd.read_csv(filepath)
    return data



def loadxlsxdata(path,file):
    filepath = path + file
    data = pd.read_excel(filepath)
    return data


def loadOneDataset(files_path, file):
    fileType = file[file.rfind('.'):]
    if fileType == '.csv':
        df = loadcsvdata(files_path,file)
    elif fileType == '.xlsx':
        df = loadxlsxdata(files_path,file)
    return df


def loadAllCsvfiles(files_path = 'C:/Users/comciencia/Desktop/AlbertoSosaPE/Tareas/Python/datasets/', listMethod = ("unscaled", ), concretFile = None):
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
        elif not file[indexpoint:] in ('.csv'):
            continue
        
        #Abrir un único fichero
        if concretFile != None and concretFile == file:
            for method in listMethod:
                loadedDatasets["{}:>{}".format(method,file)]= escalarValues(loadOneDataset(files_path,file), method)
        elif concretFile == None:
            for method in listMethod:
                loadedDatasets["{}:>{}".format(method,file)]= escalarValues(loadOneDataset(files_path,file), method)
    return loadedDatasets



def ls(path = getcwd()):
    #metodo para obtener los nombres de todos los ficheros que se encuentran en la carpeta
    return [arch.name for arch in scandir(path) if arch.is_file()]



def make_dataset(param_dict):

    """Inputs data examples creater from sklearn package"""
    
    print("--------------Creando ejemplos---------------")
    # print(param_dict)
    X, y = make_blobs (n_samples= param_dict['n_samples'], n_features= param_dict['n_features'],centers=param_dict['n_centers_samples'],
                       shuffle = param_dict['shuffle_data'], random_state=param_dict['seed'], cluster_std=.5)
   
    df = pd.DataFrame(X, columns = ('x', 'y'))
    df['target'] = y
 
    return df



def makeDatasetDict(list_number_samples, list_distance, shuffle_data = True, seed = 1):
    count = 1
    loadedDatasets = dict()
    for samples in list_number_samples:
        for distance in list_distance:
            loadedDatasets["blob_d{}_samples{}".format(distance, samples)] = make_dataset({'n_centers_samples': [[0, 6], [0, 0], [distance, 0]], 'n_samples': (samples, samples, samples), 'n_features':2, 'shuffle_data': shuffle_data, 'seed':seed})
            count+=1
    return loadedDatasets


def splitDataset(dataset, testSize, target , seed = 1):
    trainData, testData = train_test_split(dataset, test_size = testSize, random_state = seed)
    trainDataX = trainData.loc[:, trainData.columns != target]
    trainDataY = trainData[target]
    testDataX = testData.loc[:, testData.columns != target]
    testDataY = testData[target]
    
    return trainDataX, trainDataY, testDataX, testDataY


def escalarValues(df, metodo, target = 'DX_bl'):
    # escalar los datasets, segun el metodo pasado
    #borrar la etiqueta del dataset
    DXbl = df[target]
    df2 = df.drop(target, axis=1)
    if metodo == 'unscaled':
        df2[target] = DXbl
        return df2 
   
    #escalar el dataset
    method_to_call = getattr(scl, metodo)
    escalar = method_to_call()
    xEscalada = escalar.fit_transform(df2.values)
    x = pd.DataFrame(data = xEscalada, columns = df2.columns)
    
    #agregar nuevamente la etiqueta
    x[target] = DXbl
    return x




def createPCA(data, target = 'DX_bl', n_components = 3, verbose = False):
    #realizar PCA para cada dataset pasado anteriormente por el escalar
    label = data[target]
    dataWithoutLabel = data.drop(target, axis=1)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(dataWithoutLabel)  
    if verbose:
        print(data.columns)
        print("components: ", pca.components_)
        print("variance ratio: ", pca.explained_variance_ratio_)
    
    if n_components >= 3:
        # columnas = ['Principal component 1', 'Principal component 2','Principal component 3', 'Principal component 4', 'Principal component 5']
        columnas = ['Principal component 1', 'Principal component 2','Principal component 3']
    elif n_components == 2:
        columnas = ['Principal component 1', 'Principal component 2']
    dfPrincipal = pd.DataFrame(data = principalComponents, columns = columnas)
    dfPrincipal[target] = label
    return dfPrincipal




def imputerData(df, imputerType, target = 'DX_bl'):
    # introducir datos, sklearn.impute.SimpleImputer.
    print("imputando....")
    
    
    if(imputerType == 'notImputer'):
        df = df.dropna()
        return df

    else:
        dfResult = 0
        for label in pd.unique(df[target]):
            dfaux = df[df[target] == label]
            dfaux = removeNan(dfaux)
    
            imp = SimpleImputer(missing_values = np.nan, strategy=imputerType)
            imp.fit(dfaux)
            dfaux2 = imp.transform(dfaux)
        
            dfaux = pd.DataFrame(dfaux2, columns = dfaux.columns)
      
            if(isinstance(dfResult, int)):
                dfResult = dfaux
            else:
                dfResult = pd.concat([dfResult, dfaux], axis=0, ignore_index=True, join ='inner')
         
        featureEliminated = list()
        for element in df.columns:
            if element not in dfResult.columns:
                featureEliminated.append(element)
        print("campos eliminados por no tener ningún valor para alguna de las clases a comparar", featureEliminated)
        return dfResult
            
        


def modifyLabels(df, classifProblem, target = 'DX_bl'):
    dfCN = df[df[target] == 0]
    dfEMCI = df[df[target]== 1]
    dfLMCI = df[df[target]== 2]
    dfAD = df[df[target] == 3]
    
    if classifProblem == 'CN-AD':
        df = pd.concat([dfCN, dfAD])
    elif classifProblem == 'CN-LMCI-EMCI':
        df = pd.concat([dfCN, dfLMCI, dfEMCI])
    elif classifProblem == 'LMCI-EMCI': 
        df = pd.concat([dfLMCI, dfEMCI])
    elif classifProblem != 'ALL':
        
        df[target] = df[target].replace({1:2})
        dfCN = df[df[target] == 0]
        dfMCI = df[df[target]== 2]
        dfAD = df[df[target] == 3]
            
        if classifProblem == 'CN-MCI':
            df = pd.concat([dfCN, dfMCI])
        elif classifProblem == 'MCI-AD':
            df = pd.concat([dfMCI, dfAD])
        elif classifProblem == 'CN-MCI-AD': 
            df = pd.concat([dfCN, dfMCI, dfAD])
        else:
    
            df[target] = df[target].replace({1:3, 2:3})
            dfCN = df[df[target] == 0]
            dfAD = df[df[target] == 3]
        
            if classifProblem == 'CN-MCI+AD':
                df = pd.concat([dfCN, dfAD])
    else:
        df = df[df[target] != 4]
    return df




def saveDatasets(savingPath, listFeatures, dropAllCDR, df=None, dfADNI1=None, dfADNI2=None, dfADNIGO=None, dfADNI3=None, classifProblem=''):
    listADNI = ('ADNIALL','ADNI1','ADNI2','ADNIGO','ADNI3')

    if not path.isdir(savingPath):
        makedirs(savingPath)
        
    for name in listADNI:
        # guardamos los datasets
        if listFeatures:
            writer = pd.ExcelWriter(savingPath + '{}_{}_Especific.xlsx'.format(classifProblem, name))
        elif dropAllCDR: 
            writer = pd.ExcelWriter(savingPath + '{}_{}.xlsx'.format(classifProblem,name))
        else:
            writer = pd.ExcelWriter(savingPath + 'AllCDR_{}_{}.xlsx'.format(classifProblem,name))
        
        if name == 'ADNIALL':
            if df is not None:
                print('guardando: ', name)
                df.to_excel(writer,'Hoja1',index=False)
                writer.save()
        elif name == 'ADNI1':
            if dfADNI1 is not None:
                print('guardando: ', name)
                dfADNI1.to_excel(writer,'Hoja1',index=False)
                writer.save()
        elif name == 'ADNI2':
            if dfADNI2 is not None:
                print('guardando: ', name)
                dfADNI2.to_excel(writer,'Hoja1',index=False)
                writer.save()
        elif name == 'ADNIGO':
            if dfADNIGO is not None:
                print('guardando: ', name)
                dfADNIGO.to_excel(writer,'Hoja1',index=False)
                writer.save()
        else:
            if dfADNI3 is not None:
                print('guardando: ', name)
                dfADNI3.to_excel(writer,'Hoja1',index=False)
                writer.save()
 
    
 
    
def removeNan(df):
    #elimina las columnas vacias
    countList = df.count()
    for index in countList.index:
        if countList[index] == 0:
            df = df.drop([index],axis = 1)
    return df



def takeTypeDataScaling(dataset):
    typeDataScaling = dataset.split(":>")[0]
    if len(dataset) == len(typeDataScaling):
        typeDataScaling = None
    elif not typeDataScaling in ('unscaled', 'StandardScaler', 'RobustScaler', 'PowerTransformer', 'Normalizer', 'MinMaxScaler', 'MaxAbsScaler'):
        typeDataScaling = None
    return typeDataScaling


def loadDataset_CN_MCI_AD_fromPKL(path, pklFileName, num_components, scaled, projection):
    
    s= join(path, pklFileName)
    a = load(s)
    
    data_train =  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split {}___num_components={} {}'.format(scaled, num_components, projection)]['data_train']
    data_test = a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split {}___num_components={} {}'.format(scaled, num_components, projection)]['data_test']
    labels_train = a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split {}___num_components={} {}'.format(scaled, num_components, projection)]['labels_train']
    labels_test = a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split {}___num_components={} {}'.format(scaled, num_components, projection)]['labels_test']
    
    return data_train, data_test, labels_train, labels_test
    