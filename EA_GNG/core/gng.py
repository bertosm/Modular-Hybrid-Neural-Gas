#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:22:48 2020

@author: Berto Sosa
"""


from operator import attrgetter

import pickle
import math
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import random

from sklearn.preprocessing import normalize

from os import getcwd, scandir, makedirs, path

from .label import labelClustersResult
from .dataset import createPCA
from .figure import createFigures, saveFigures, showdata, showResult, determineClusters
import EA_GNG.EA_GNG as gng

from EA_GNG.GNGperceptron import GNG_perceptron

from neupy import utils
from neupy.algorithms.competitive.growing_neural_gas import GrowingNeuralGas, NeuronNode, NeuralGasGraph, sample_data_point,StopTraining

from EA_GNG.core.method.metrics import evaluateClusteringQuality
from EA_GNG.core.method.statistics import round_decimals_down




class contador():
    
    def __init__(self, inicial=0):
        self.contador = inicial
    
    def aumentarCont(self, aum = 1):
        self.contador += aum
        return self.contador

#variable global             
neuronCount = contador(0)

       

    
def neupy_growingneuralgas(trainDataX, param_dict, ax, fig, trainLabelsY = None, testDataX = None, testLabel=None, saveProcess=False, saving_path = None): 
    """growing neural network algorithm from neupy packeage"""
    
    utils.reproducible(param_dict['seed'])
    np.random.seed(param_dict['seed'])
    random.seed(param_dict['seed'])
    tf.random.set_random_seed(param_dict['seed'])
    tf.set_random_seed(param_dict['seed'])
                
    print("[inicializando] growing neural gas ->NEUPY<- ")
    
    #Marca de tiempo inicial
    tini = time.time()
    
    
    #n_inputs =  Number of features in each sample.  
    #Creando el modelo GNG de NeuPy
    gng_neupy = neupy_gng(n_inputs = param_dict['n_features'],                                                                          
                          step=param_dict['winner_step'],
                          neighbour_step = param_dict['neighbour_step'],
                          max_edge_age = param_dict['max_edge_age'],
                          n_iter_before_neuron_added = param_dict['n_iter_before_neuron_added'],
                          error_decay_rate= param_dict['error_decay_rate'],
                          after_split_error_decay_rate = param_dict['after_split_error_decay_rate'],
                          n_start_nodes=param_dict['n_start_nodes'],
                          max_nodes = param_dict['max_nodes'],
                          min_distance_for_update = param_dict['min_distance_for_update'],
                          shuffle_data = param_dict['shuffle_data'])
           
 
    
    print("[Entrenando] growing neural gas ->NEUPY<- ")
       
    #Entrenando el modelo
    # print("train Data: ", trainDataX)
    # print("train Label: ", trainLabelsY)
    print("saving_path preGNG: ", saving_path)
    bestCalinski, bestSilhouette = gng_neupy.train(trainDataX, trainLabelsY, X_test=testDataX, y_test=testLabel, epochs=param_dict['epochs'], saveProcess = saveProcess, saving_path=saving_path+"config{}".format(param_dict["count"]))
    
    
    #Marca de tiempo final, Tiempo que tarda en entrenar.
    tend= time.time()
    
    if not saveProcess:
        # Si no se quiere guardar el proceso, se construye la red final
        color_dict, n_clusters = determineClusters(gng_neupy.graph)
                      
        print("[Resultado] growing neural gas ->NEUPY<- ")    
         
        if trainLabelsY is None:
            colorPred = None
            dictcolor_label = None
        else:
            dictcolor_label, colorPred = labelClustersResult(trainDataX, trainLabelsY, gng_neupy.graph, color_dict, testDataX)
            
        showResult(gng_neupy.graph, color_dict, ax, fig, trainDataX.columns, dictcolor_label)
    else:
        n_clusters = None
        colorPred = None
        
    return tend - tini, n_clusters, colorPred, bestCalinski, bestSilhouette




class neupy_NamedNeuronNode(NeuronNode):
       
       def __init__(self, weight):

              NeuronNode.__init__(self, weight)
              self.name = neuronCount.aumentarCont(1)
            
       def __repr__(self):
        return "<{} error={}>".format(
            self.__class__.__name__,
            round(float(self.error), 6))
        
   

def loop_gng(config, saving_path, df, PCA=False, PCA_n_components = 3, hibrid=False, typeDataScaling=None, labelsOrdering = None, nameDataset = 'NoNamedDataset', seed = 1, shuffle_data = True, verbose = False, savedGNG=False, saveProcess=False):
  
    
    # Pre processing del conjunto de datos (PCA y seleccion de características si es necesario)
    target = df.columns.tolist()[-1]
    
    # df = df[["MMSCORE", "FAQSHOP", "ADAS_Q7", "DX_bl"]]
    print(df)
    
    if PCA and len(df.columns.tolist()) > 3:
        df = createPCA(df, target, n_components = PCA_n_components, verbose = True)
        print("PCA: ", df)


    if not path.isdir("C:/Users/Bertosm/Desktop/dataset/"):
        makedirs("C:/Users/Bertosm/Desktop/dataset/")
        
        
    # Se imprime excel con los overlapping que existen en el cuerpo de entrada. 
    overlapping = df.groupby(df.columns.tolist(), as_index = False).size()
    writer = pd.ExcelWriter( 'C:/Users/Bertosm/Desktop/dataset/overlappingPCA-3Comp.xlsx')
    overlapping.to_excel(writer,'Hoja1',index=False)
    writer.save()


    listFeatures = df.columns.tolist()
    
    
    # creamos un diccionario con los parámetros pasados de la configuracion
    param_dict_gng = {'n_start_nodes': config[4],
                  'winner_step': config[9], #0.2
                  'neighbour_step': config[8], #0.06
                  'max_edge_age': config[3], # 30 333
                  'n_iter_before_neuron_added':config[2], # 50 separated 2.9 333
                  'error_decay_rate': config[7],
                  'after_split_error_decay_rate': config[6],
                  'max_nodes': config[1], #200 333
                  'min_distance_for_update': config[5],
                  'shuffle_data': shuffle_data,
                  'epochs': config[0],
                  'seed': seed,
                  'typeDataScaling': typeDataScaling,
                  'listFeatures': listFeatures,
                  'count': config[-1]}
    
    param_dict_perceptron = {
                'activation_neigbor': config[10],
                'learningRate': config[11],
                'epochs': config[12],
                'seed': seed,
                'count': config[-1]}
    

    if hibrid:

        GNG_perceptron(param_dict_gng=param_dict_gng, param_dict_perceptron=param_dict_perceptron, df=df, target = target, saving_path = saving_path,nameDataset = nameDataset, verbose = verbose, savedGNG = savedGNG, saveProcess=saveProcess)
    
    else:
        gng.growingNeuralGas(param_dict_gng, df,target = target, saving_path = saving_path, labelsOrdering = labelsOrdering, nameDataset = nameDataset, verbose = verbose, saveProcess=saveProcess)
    
    
    
class neupy_NeuralGasGraph(NeuralGasGraph):
    
    def add_node(self, node):
        self.edges_per_node[node] = dict()
        
    def add_edge(self, node_1, node_2):
        if node_2 in self.edges_per_node[node_1]:
            return self.reset_edge(node_1, node_2)

        self.edges_per_node[node_1][node_2] = 0
        self.edges_per_node[node_2][node_1] = 0
        self.edges[(node_1, node_2)] = 0
        
        
    def remove_edge(self, node_1, node_2):
        edge_id = self.find_edge_id(node_1, node_2)

        self.edges_per_node[node_1].pop(node_2)
        self.edges_per_node[node_2].pop(node_1)

        del self.edges[edge_id]

        
class neupy_gng(GrowingNeuralGas):
    
    def initialize_nodes(self, data):
        self.graph = neupy_NeuralGasGraph()
        for sample in sample_data_point(data, n=self.n_start_nodes):
            self.graph.add_node(neupy_NamedNeuronNode(sample.reshape(1, -1)))
            
           
    def train(self, trainData, y_train = None, X_test = None, y_test = None, epochs=100, saveProcess=False, saving_path = None):
        
        print("saving_path pre training: ", saving_path)
        # print("train data preFloat: ", trainData)
        X_train = self.format_input_data(trainData)
        # print("train data postFloat: ", X_train)
        if not self.graph.nodes:
           self.initialize_nodes(X_train)
           
  
      
        return super(GrowingNeuralGas, self).train(
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    epochs=epochs, data=trainData, saveProcess=saveProcess, saving_path=saving_path)
                
    def one_training_update(self, X_train, epoch, y_train=None, X_test=None, y_test=None, data=None, saveProcess=False, saving_path=None, bestCalinski=0, bestSilhouette=-2):

        graph = self.graph
        step = self.step
        neighbour_step = self.neighbour_step
        
        max_nodes = self.max_nodes
        max_edge_age = self.max_edge_age
        
        error_decay_rate = self.error_decay_rate
        after_split_error_decay_rate = self.after_split_error_decay_rate
        n_iter_before_neuron_added = self.n_iter_before_neuron_added
        
        # We square this value, because we deal with
          # squared distances during the training.
        min_distance_for_update = np.square(self.min_distance_for_update)
        n_samples = len(X_train)
        total_error = 0
        did_update = False

             
        for sample in X_train:
            nodes = graph.nodes
            weights = np.concatenate([node.weight for node in nodes])

            distance = np.linalg.norm(weights - sample, axis=1)
            neuron_ids = np.argsort(distance)

            closest_neuron_id, second_closest_id = neuron_ids[:2]
            
            closest_neuron = nodes[closest_neuron_id]
            second_closest = nodes[second_closest_id]
            
            total_error += distance[closest_neuron_id]

            if distance[closest_neuron_id] < min_distance_for_update:
                continue

            self.n_updates += 1
            did_update = True

            closest_neuron.error += distance[closest_neuron_id]
            closest_neuron.weight += step * (sample - closest_neuron.weight)

            graph.add_edge(closest_neuron, second_closest)
  
            for to_neuron in list(graph.edges_per_node[closest_neuron]):
                edge_id = graph.find_edge_id(to_neuron, closest_neuron)
                age = graph.edges[edge_id]

                if age >= max_edge_age:
                    graph.remove_edge(to_neuron, closest_neuron)

                    if not graph.edges_per_node[to_neuron]:
                        graph.remove_node(to_neuron)

                else:
                    graph.edges[edge_id] += 1
                    to_neuron.weight += neighbour_step * (
                        sample - to_neuron.weight)

            time_to_add_new_neuron = (
                self.n_updates % n_iter_before_neuron_added == 0 and
                graph.n_nodes < max_nodes)

            if time_to_add_new_neuron:
                nodes = graph.nodes
                largest_error_neuron = max(nodes, key=attrgetter('error'))
              
                neighbour_neuron = max(
                    graph.edges_per_node[largest_error_neuron],
                    key=attrgetter('error'))

                largest_error_neuron.error *= after_split_error_decay_rate
                neighbour_neuron.error *= after_split_error_decay_rate

                new_weight = 0.5 * (
                    largest_error_neuron.weight + neighbour_neuron.weight
                )
                new_neuron = neupy_NamedNeuronNode(weight=new_weight.reshape(1, -1))

                graph.remove_edge(neighbour_neuron, largest_error_neuron)
                graph.add_node(new_neuron)
                graph.add_edge(largest_error_neuron, new_neuron)
                graph.add_edge(neighbour_neuron, new_neuron)

            for node in graph.nodes:
                node.error *= error_decay_rate

        if saveProcess and saving_path != None:
            calinski, silhouette = saveGraphPerEpoch(graph, X_train, epoch, saving_path, y_train, X_test, y_test, data, bestCalinski, bestSilhouette, max_nodes)
        else:
            calinski=0
            silhouette=-2
    
        if not did_update and min_distance_for_update != 0 and n_samples > 1:
              raise StopTraining(
                  "Distance between every data sample and neurons, closest "
                  "to them, is less then {}".format(min_distance_for_update))
              
        return total_error / n_samples, calinski, silhouette
    
    def saveGNG(self, saving_path):
        # saving_path += "gngSave.pkl"
        with open(saving_path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    def outputActivation(self, dataX, num_neigbor=1):
        
        #Si el numero de neigbor es 0, se le mandará la distancia euclidiana como salida.
        # Distancia euclidiana de cada neurona al valor de entrada.
        
        nodes = self.graph.nodes
        weights = np.concatenate([node.weight for node in nodes])
        distance = np.linalg.norm(weights - dataX, axis=1)
        
        if num_neigbor == 0:
            # print("distance: ", distance)
            normalized_distance = distance/ np.linalg.norm(distance)
            # print("normalized distance: ", normalized_distance)
            
            return normalized_distance
        else:
            activationVector = np.zeros(len(nodes))
            
            neuron_ids = np.argsort(distance)
    
            percentageActivation = round_decimals_down(1/num_neigbor, 2)
            activationValue = 1
            for nearestNeurons in neuron_ids[:num_neigbor]:
                activationVector[nearestNeurons] = activationValue
                activationValue -= percentageActivation
    
            # print("activationVector: ", activationVector)
            return activationVector
        
        


def saveGraphPerEpoch(graph, trainData, epoch, saving_path,  trainLabel, testData, testLabel, data=None, bestCalinski=0, bestSilhouette= -2, max_nodes=2):

    # print("Calinski: ", bestCalinski)
    # print("numero de neuronas: ", len(graph.nodes))
    color_dict, n_clusters = determineClusters(graph)

    if not data is None:
        if n_clusters == len(pd.unique(trainLabel)):
            # print(epoch)
            # print("dataX: ", data)
            # print("columns: ", data.columns.tolist())
            # print("labelY: ", trainLabel)
            dictcolor_label, labelsPred = labelClustersResult(data, trainLabel, graph, color_dict, testData=testData)
     
            # print("test X: ", testData.shape)
            # print("test Y: ", testLabel.shape)
            # print(" labelsPred: ", len(labelsPred))
        
            homogeneity, completeness, v_measure, ari, dbs, normalizedmutualInfo, fowlkes, calinski, purity, sil= evaluateClusteringQuality(testData, testLabel, labelsPred)
            # print("silhouette: ", sil)
            if (calinski >= bestCalinski or sil >= bestSilhouette) and len(graph.nodes) > max_nodes*0.1:
                # Proceso de guardado de figura en cada epoch
                if calinski < bestCalinski:
                    calinski = bestCalinski
                if sil < bestSilhouette:
                    sil = bestSilhouette
                    
                fig3d, fig, ax1, ax2 = createFigures("EpochsTest", "Unescaled", "numberOfEpoch:{}-Calinski:{}-silhouette:{}".format(epoch, calinski, sil), trainData.shape[1], numOfAx=2)
                
                showdata(data, trainLabel, ax1, fig3d, data.columns.tolist(), "epochsTest")
                showResult(graph, color_dict, ax2, fig3d, data.columns.tolist(), dictcolor_label)
        
                saving_path = saving_path
                # print("calinski: ", calinski, "silhouette: ", sil)
                saveFigures(fig3d, fig, saving_path, epoch = epoch, calinski=calinski, silhouette=sil)
                return calinski, sil
            
            
 

    return bestCalinski, bestSilhouette


def batchLabel(trainData, trainLabel):
    
    indices = np.arange(trainData.shape[0])
    np.random.shuffle(indices)

    arrayTrainLabel = trainLabel.to_numpy()
    
    # print("arrayLabel: ", arrayTrainLabel)

    yield apply_slices(arrayTrainLabel, indices)
    
    
    
def apply_slices(inputs, indices):
    if inputs is None:
        return inputs

    if isinstance(inputs, (list, tuple)):
        return [apply_slices(input_, indices) for input_ in inputs]

    return inputs[indices]
