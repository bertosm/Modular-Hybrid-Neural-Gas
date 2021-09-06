# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 19:05:52 2021

@author: Bertosm
"""
import numpy as np
from os import path, makedirs
import sys
from EA_GNG.core.method.metrics import prettyConfusionMatrix, calculateClassificationMetrics, saveMetricsPerceptron
from EA_GNG.core.figure import maketitle

print("!!!!!!------Imported Perceptron backpropagation - biclass--------!!!!!!\n")

class perceptron():
    
    """Class perceptron:

       Attributes:
         eta.- Learning rate
         epoch.-
         seed.- Random process seed
         weights.- 
         
       Methods:
         __init__()
         train()
         _net_input()
         predict(p_x) .- Method to predict the output, y

    """

    def __init__(self, param_dict, limit, gng_neupy= None):
        self.learningRate = param_dict["learningRate"]
        self.seed = param_dict["seed"]
        self.epochs = param_dict["epochs"]
        self.count  = param_dict["count"]
        self.activationNeigbors = param_dict["activation_neigbor"]
        self.limit = limit
        self.gng_neupy = gng_neupy
        self.param_dict = param_dict
        print("lr:{}-epochs:{}-limit:{}".format(self.learningRate,self.epochs, self.limit))
        
        np.random.seed(self.seed)
    
    def train(self, trainDataX, 
                    trainLabelY, 
                    validationDataX, 
                    validationLabelY,
                    saving_path):
        
        bestAcc = 0
        
        if not path.isdir(saving_path):
            makedirs(saving_path, exist_ok = True)
            
           
        if(self.gng_neupy != None):
            print("neuronas de entrada perceptron: {} procedentes de GNG".format(self.gng_neupy.graph.n_nodes))
            
            # inicialize between -0.01 to 0.01 (Code from SI1)
            # self.weights =np.random.RandomState(seed=1).normal(loc=0, scale=0.01, size=self.gng_neupy.graph.n_nodes+1)
            # print("weight -0.01 to 0.01")
            
            # ini0cialize between 0 to 1 (np.random)
            self.weights =np.random.rand(self.gng_neupy.graph.n_nodes+1)
            print("weight 0 to 1")
            
        else:
            print("neuronas de entrada perceptron: {} procedentes del conjunto de datos (No GNG)".format(trainDataX.shape[1]))
            
            # inicialize between -0.01 to 0.01 (Code from SI1)
            # self.weights =np.random.RandomState(seed=1).normal(loc=0, scale=0.01, size=trainDataX.shape[1]+1)
            # print("weight -0.01 to 0.01")
            
            
            # inicialize between 0 to 1 (np.random)
            self.weights =np.random.rand(trainDataX.shape[1]+1)
            print("weight 0 to 1")
            
            
        # print("weightsShape: ", len(self.weights))
        # print("weightsMatrix: ",self.weights)
        
        iteraciones = 0
        
        for i in range(self.epochs):
            
            for index, row in trainDataX.iterrows():
                iteraciones += 1



# =============================================================================
#                 Forward
# =============================================================================
                outputActivation = self.activation(row)
# =============================================================================
#                 Backward
# =============================================================================

   
                def det(output_value):
                    derivate = np.multiply(output_value, (1 - output_value))
                    return derivate

                derivatedOutput = det(outputActivation)
                errorOutput = 2*np.multiply((trainLabelY.loc[index] - outputActivation), derivatedOutput)
                
                d_output_layer_ = (self.learningRate * errorOutput * self.activate_gngVector(row))
                self.weights[1:] = self.weights[1:] + d_output_layer_
                self.weights[0] = self.weights[0] + (self.learningRate * errorOutput)


# =============================================================================
#                Validation per epoch
# =============================================================================
                labelsPredicted = self.predict(validationDataX.to_numpy())
                
                metrics = dict()
                metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1Score'], metrics["falsos_positivos"], metrics["verdaderos_positivos"] = calculateClassificationMetrics(validationLabelY, labelsPredicted, verbose=False)
                
                acc = metrics["accuracy"]
          
                if iteraciones == 1:
                    meanAccuracy = acc
                
                meanAccuracy = (meanAccuracy + acc)/2
                
            print("epoch:{}--iteraciones{}---Acc{}".format(i, iteraciones, meanAccuracy))
            
            if meanAccuracy > bestAcc:
                bestAcc = meanAccuracy
                
            
# =============================================================================
#        Final Validation (testeo pero con los mismos datos de validación)
# =============================================================================


        labelsPredicted = self.predict(validationDataX.to_numpy())
        # saveFigureLabelsPred(validationDataX, validationLabelY, labelsPredicted, saving_path, self.count, sTitle=maketitle(self.param_dict, "perceptron"))
        
        metrics = dict()
        metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1Score'], metrics["falsos_positivos"], metrics["verdaderos_positivos"] = calculateClassificationMetrics(validationLabelY, labelsPredicted, verbose=False)                
        
        metrics['auc']=0.5
        dfCM = prettyConfusionMatrix(validationLabelY, labelsPredicted, verbose = False)
        confusion_matrixSavePath = '{}config{}-limit{}-Final-confusion_matrix.csv'.format(saving_path, self.count, self.limit)
        dfCM.to_csv(confusion_matrixSavePath)
        
        
        print("saving_path previous call saveMetrics: ",  saving_path)
        
        saveMetricsPerceptron(self.count, saving_path, metrics, sTitle=maketitle(self.param_dict, "perceptron"), limit=self.limit, activationNeigbors=self.activationNeigbors, bestAcc=bestAcc)
        

        print("Validación Final-Acc{}".format(metrics["accuracy"]))
            
    
    
    def activate_gngVector(self, data):
        
        try:
            data.shape[1]
        except:
            data = data.values.reshape(1,-1)

        np.set_printoptions(threshold=sys.maxsize)     
        if self.gng_neupy != None:
            vectorPredicted=np.zeros(shape=(data.shape[0], self.gng_neupy.graph.n_nodes))
            for i, row in zip(range(data.shape[0]), data):
                vectorPredicted[i] =  self.gng_neupy.outputActivation(row, self.activationNeigbors)
      
            return vectorPredicted
        else:
            return data
        
        
    def _net_input(self, data):
        return np.dot(self.activate_gngVector(data), self.weights[1:]) + self.weights[0]
    
    def activation(self, data):
        return 1.0 / (1.0 + np.exp(-self._net_input(data)))             
        
    def predict(self, data):
        return np.where(self.activation(data) >= self.limit, 1, 0)
    