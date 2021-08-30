#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:36:54 2020

@author: Berto Sosa
"""
import textwrap
from random import randint
from os import makedirs, path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from plotly.offline import plot
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from EA_GNG.core.label import get_key

def showdata(dataX, labelsY, ax, fig, columns=(), nameDataset = 'blob'):
    
    """ploting data"""
    
    print("--------------dibujando ejemplos---------------")
    
    color_vector = ['green', 'lightgrey', 'gold', 'purple', 'black', 'orange', 'olive', 'lime', 'blue', 'magenta']
    
    aux = np.unique(labelsY)
    dataX['target'] = labelsY
    
    if  'blob' in nameDataset:
        for label in aux:
            color = color_vector.pop()
            dfAux = dataX[dataX['target'] == label] 
            fig.add_scatter(x = dfAux.iloc[:,0], y = dfAux.iloc[:,1], mode = 'markers', row = 1, col = 1, name='{}'.format(label), marker= {'color': color}, legendgroup = 'inputs/data')
            
            ax.scatter(dfAux.iloc[:,0], dfAux.iloc[:,1], c = color, label = label)       
    
    else:
        for label in aux:
            color = color_vector.pop()
            datalabels = dataX[dataX['target'] == label]
            
            if(dataX.shape[1] > 3):
                fig.add_scatter3d(x = datalabels[columns[0]],y = datalabels[columns[1]], z=datalabels[columns[2]], mode = 'markers', row = 1, col = 1, name='{}'.format(label), marker= {'color': color}, legendgroup = 'inputs/data', opacity=0.25)
                fig.update_layout(scene={'zaxis':{'title':{'text':columns[2],'font':{'size':12}}}})
                ax.scatter(datalabels.iloc[:,0], datalabels.iloc[:,1], datalabels.iloc[:,2], c = color, label = label,  alpha = 0.25)
                ax.set_zlabel(columns[2])
            else:
                fig.add_scatter(x = datalabels[columns[0]],y = datalabels[columns[1]], mode = 'markers', row = 1, col = 1, name='{}'.format(label), marker= {'color': color}, legendgroup = 'inputs/data', opacity=0.25)
                fig['layout']['xaxis']['title']= columns[0]
                fig['layout']['yaxis']['title']= columns[1]
                ax.scatter(datalabels.iloc[:,0], datalabels.iloc[:,1], c = color, label = label, alpha = 0.25)
                
        
    dataX.drop('target', axis=1, inplace=True)
    
    
    ax.legend()
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    fig.update_layout(scene={'xaxis':{'title':{'text':columns[0],'font':{'size':12}}},'yaxis':{'title':{'text':columns[1], 'font':{'size':12}}}})
  
def showdataPredicted(dataX, labelsY, labelsPred, fig, sTitle="data predicted", columns=()):
    
    """ploting data"""
    
    print("--------------dibujando ejemplos---------------")
    
    color_vector = [ 'green', 'lightgrey', 'gold', 'purple', 'black', 'orange', 'olive', 'lime', 'blue', 'magenta', 'black', 'silver']
    
    aux = np.unique(labelsY)
    dataX['target'] = labelsY
 
    for label in aux:
        
        color = color_vector.pop()
        datalabels = dataX[dataX['target'] == label]
        if label == 0:
            label = "MCI"
        else:
            label = "AD"
            
        if(dataX.shape[1] > 3):
            plt.scatter3d(datalabels.iloc[:,0], datalabels.iloc[:,1], datalabels.iloc[:,2], label = "{}-True".format(label), c=color)
            plt.zlabel(columns[2])
        else:
             plt.scatter(datalabels.iloc[:,0], datalabels.iloc[:,1], label = "{}-True".format(label),  c=color)
                
    
    aux = np.unique(labelsPred)
    dataX['target'] = labelsPred
    color_vector = ['black', 'gray', 'green', 'lightgrey', 'gold', 'purple', 'black', 'orange', 'olive', 'lime', 'blue', 'magenta', 'black', 'silver']
    for label in aux:
        
        color = color_vector.pop()
        datalabels = dataX[dataX['target'] == label]
        
        if label == 0:
            label = "MCI"
        else:
            label = "AD"

        if(dataX.shape[1] > 3):
            plt.scatter(datalabels.iloc[:,0], datalabels.iloc[:,1], datalabels.iloc[:,2], label = "{}-Pred".format(label),  marker="s", facecolors='none',  edgecolors=color, s = 60)
        else:
            plt.scatter(datalabels.iloc[:,0], datalabels.iloc[:,1], label = "{}-pred".format(label), marker="s",  facecolors='none', edgecolors=color, s = 60)
    dataX.drop('target', axis=1, inplace=True)
    
    plt.rcParams['axes.facecolor'] = 'none'
    plt.style.use('grayscale')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title("Data predicted")
    

def showResult(gng_graph, color_dict, ax, fig, columns, dictcolor_label = None):

    print("pintando resultados")
    #introducir df para plotly
    df = pd.DataFrame()
    legend = list()
    legend2 = list()
    
    for node_1, node_2 in gng_graph.edges:
        
        cor = color_dict[node_1.name]
        if not dictcolor_label == None:
            labelPred = "->" + str(get_key(cor, dictcolor_label))
        else:
            labelPred=""
        #figura 2D
        weights = np.concatenate([node_1.weight, node_2.weight])
        if(weights.shape[1] >3):
            weights = weights[:,:3]
        if not cor in legend:
            ax.scatter(*weights.T, c=cor, label = 'cluster{}{}'.format(len(legend), labelPred))
            legend.append(cor)
        else:
            ax.scatter(*weights.T, c=cor)
        ax.plot(*weights.T,color ='black')
        pesos = pd.DataFrame(weights)
        pesos['label'] = color_dict[node_1.name]

        # figura 3D color base
        if pesos.shape[1] == 3:
            fig.add_scatter(x = pesos[0], y = pesos[1], mode = 'lines', row = 1, col = 2, showlegend = False, name='edges', ids = (0,0), line = {'color': 'black', 'width': 4}, legendgroup='edges')
        else:
            fig.add_scatter3d(x = pesos[0], y = pesos[1], z=pesos[2], mode = 'lines', row = 1, col = 2, showlegend = False, name='edges', ids = (0,0), line = {'color': 'black', 'width': 4}, legendgroup='edges', uirevision='')
            ax.set_zlabel(columns[2])
            
        df = pd.concat([df,pesos], ignore_index = True)
  
        
        # figura 3D coloreando puntos
        for unique, i in zip(pd.unique(df['label']), range(len(pd.unique(df['label'])))):
            if not dictcolor_label == None:
                labelPred2 = "->" + str(get_key(unique, dictcolor_label))
            else:
                labelPred2=""
            if not i in legend2:
                legend2.append(i)
                dfaux = df[df['label'] == unique]
                if pesos.shape[1] == 3:
                    fig.add_scatter(x = dfaux[0], y = dfaux[1], mode = 'markers', row = 1, col = 2, showlegend=True, marker={'color': unique}, name = 'cluster{}{}'.format(i,labelPred2), legendgroup = 'clusters')
                else:
                    fig.add_scatter3d(x = dfaux[0], y = dfaux[1], z=dfaux[2], mode = 'markers', row = 1, col = 2, showlegend=True, marker={'color': unique}, name = 'cluster{}{}'.format(i,labelPred2), legendgroup = 'clusters')
                    fig.update_layout(scene2={'zaxis':{'title':{'text':columns[2],'font':{'size':12}}}})
            else:
                dfaux = df[df['label'] == unique]
                if pesos.shape[1] == 3:
                    fig.add_scatter(x = dfaux[0], y = dfaux[1], mode = 'markers', row = 1, col = 2, showlegend = False, marker={'color': unique}, name = 'cluster{}{}'.format(i,labelPred2), legendgroup = 'clusters')
                    fig['layout']['xaxis2']['title']= columns[0]
                    fig['layout']['yaxis2']['title']= columns[1]
                    fig['layout']['yaxis2']['showticklabels']= True
                else:
                    fig.add_scatter3d(x = dfaux[0], y = dfaux[1], z=dfaux[2], mode = 'markers', row = 1, col = 2, showlegend = False, marker={'color': unique}, name = 'cluster{}{}'.format(i, labelPred2), legendgroup = 'clusters')
                    fig.update_layout(scene2={'zaxis':{'title':{'text':columns[2],'font':{'size':12}}}})
                    fig.update_layout(scene2={'xaxis':{'title':{'text':columns[0],'font':{'size':12}}},'yaxis':{'title':{'text':columns[1], 'font':{'size':12}}}})
  
    ax.legend(prop={'size':6})
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    

def determineClusters(gng_graph):
    #Pintando las neuronas en las gráficas resultantes.
    color_dict=dict()
    #Pintando por clusters encontrados
    #colorcluster modifica el diccionario de colores(color_dict) según los clusters resultantes.
    n_clusters = colorcluster(color_dict, gng_graph, basecolor = 'black')

    return color_dict, len(n_clusters)
    

def colorcluster(color_dict, graph, basecolor = 'black'):
    
    #Estableciendo un color base negro.
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])  
        if(weights.shape[1]>3):
            weights = weights[:,:3]
        color_dict[node_1.name] = basecolor
        color_dict[node_2.name] = basecolor
        
    
    """inicializing color cluster"""
    color_vector = ['green', 'lightgrey', 'gold', 'purple', 'black', 'orange', 'olive', 'lime', 'blue', 'magenta']
    n_clusters = list()
    for node in graph.nodes:
        recursive_neupy(node, color_dict, color_vector, graph, n_clusters, basecolor)
    return n_clusters
 

   

def recursive_neupy(node, color_dict, color_vector, graph, n_clusters, basecolor = 'black'):
    
    """recursive color neurons of a cluster for gng - neupy package"""
    
## comprobar que ni el node ni sus neighbor son basecolor
    if not color_dict[node.name] is basecolor:

        colorcontrol = False
        for neighbor in graph.edges_per_node[node]:
            if color_dict[neighbor.name] is basecolor:
                colorcontrol = True
                break
        if not colorcontrol:
            return
       
    if color_dict[node.name] is basecolor:
        n_clusters.append(0)
        
        if len(color_vector) == 0:
            color_vector = ['green', 'lightgrey', 'gold', 'purple', 'black', 'orange', 'olive', 'lime', 'blue', 'magenta']
            
        color_dict[node.name] = color_vector.pop()
        if color_dict[node.name] is basecolor:
            color_dict[node.name] = color_vector.pop()
              
    for neighbor in graph.edges_per_node[node]:        
        
        if color_dict[neighbor.name] is basecolor:
            color_dict[neighbor.name] = color_dict[node.name]

    for neighbor in graph.edges_per_node[node]:
        recursive_neupy(neighbor, color_dict, color_vector, graph,n_clusters, basecolor)    

    

    
def maketitle(param_dict, nameDataset):
    
    """creating a string from a dictionary"""
    
    if not "max_edge_age" in  param_dict:
         dictTitle = {'learningRate': param_dict['learningRate'],
                     'Epochs': param_dict['epochs'],
                     'Seed': param_dict['seed']}
        
    else:
    
        dictTitle = {'MaxAgeEdge': param_dict['max_edge_age'],
                     'lambda': param_dict['n_iter_before_neuron_added'],
                     'Epochs': param_dict['epochs'],
                     'maxNodes': param_dict['max_nodes'],
                     'WinnerStep': param_dict['winner_step'],
                     'NeighborStep': param_dict['neighbour_step'],
                     'Seed': param_dict['seed'],
                     'Nfeatures': param_dict['n_features'],
                     'ErrorDecayRate': param_dict['error_decay_rate'],
                     'features': param_dict['listFeatures']}
         
    stringaux = textwrap.wrap(' '.join('%s=%s' % (str(k), str(v)) for k, v in dictTitle.items()),width = 110)
    return '\n'.join(fila for fila in stringaux)




def createFigures(nameDataset, typeDataScaling, sTitle, sizeDataset, numOfAx=2):
    input_Title = 'Inputs - typeDataScaling:{}'.format(typeDataScaling)

    fig = plt.figure()
    
    if sizeDataset == 2:
        
        if numOfAx == 2:
            # """-----------------Projection 2d -------------------
            # Make figure with subplots
            fig3d = make_subplots(rows=1,
                                  cols=2,
                                  shared_xaxes=True, 
                                  shared_yaxes=True, 
                                  specs=[[{"type": "scatter"},
                                          {"type": "scatter"}]],
                                  subplot_titles=(input_Title,
                                                  'CLUSTERS RESULTS (ANN)'))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, sharex = ax1, sharey = ax1)
            # --------------------------------------------------"""
            
            ax1.set_title(input_Title)
    
            fig3d.update_layout(title={'text': sTitle.replace('\n', '<br>'), 'x': 0.5, 'y': 0.025, 'font':{'size': 12}})
            fig.suptitle(sTitle, y = 1.05)
    
            l, b, w, h = ax1.get_position().bounds
            ax1.set_position([l-0.1,b,w,h])
            l, b, w, h = ax2.get_position().bounds
            ax2.set_position([l+0.1,b,w,h])
        else:
            ax1 = fig
            ax2 = None
            fig3d = None
        
    else:
        if numOfAx == 2:
        
            # -----------------Projection 3d -------------------
            # Make figure with subplots
            fig3d = make_subplots(rows=1,
                                  cols=2,
                                  shared_xaxes=True, 
                                  shared_yaxes=True, 
                                  specs=[[{"type": "scatter3d"},
                                          {"type": "scatter3d"}]],
                                  subplot_titles=(input_Title,
                                                  'CLUSTERS RESULTS (ANN)'))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            # --------------------------------------------------
    
            ax1.set_title(input_Title)
    
            fig3d.update_layout(title={'text': sTitle.replace('\n', '<br>'), 'x': 0.5, 'y': 0.025, 'font':{'size': 12}})
            fig.suptitle(sTitle, y = 1.05)
    
            l, b, w, h = ax1.get_position().bounds
            ax1.set_position([l-0.1,b,w,h])
            l, b, w, h = ax2.get_position().bounds
            ax2.set_position([l+0.1,b,w,h])
            
        else:
            ax1 = None
            ax2 = None
            fig3d = None
    
    
    return fig3d, fig, ax1, ax2


def saveFigures(fig3d, fig, saving_path, param_dict = None, epoch = None, calinski=None, silhouette=None):
    
    if not path.isdir(saving_path):
        makedirs(saving_path)
    
    #guardar imagen 3D y 2D            
    if param_dict is None:
        # print("silhouette pre print", silhouette)
        
        
        # fileName = '{}configEpoch{}-Calinski{}-silhouette{}.html'.format(saving_path, epoch, int(calinski), str(silhouette))
        # savename = '{}configEpoch{}-Calinski{}-silhouette{}.png'.format(saving_path, epoch, int(calinski), str(silhouette))
        
        fileName = '{}bestConfigSilhouette.html'.format(saving_path)
        savename = '{}bestConfigSilhouette.png'.format(saving_path)
    else:
        fileName = '{}config{}-3Dplot-{}.html'.format(saving_path, param_dict['count'], param_dict['typeDataScaling'])
        savename = saving_path + 'config{}-{}.png'.format(param_dict['count'], param_dict['typeDataScaling'])

    if fig3d != None:
        plot(fig3d, filename = fileName, auto_open=False)
        
    plt.savefig(savename,bbox_inches = 'tight')
    plt.close(fig)
    
def saveFigureLabelsPred(testDataX, labelsTrue, labelsPred, saving_path, count = 1, sTitle="data predicted"):
    fig3d, fig, ax1, ax2 = createFigures("predicted", "", sTitle, 2, 1)
    showdataPredicted(testDataX, labelsTrue, labelsPred, fig, sTitle= sTitle, columns=testDataX.columns.tolist())
    saving_path=saving_path + "config{}-labelsPred".format(count)
    saveFigures(fig3d, fig, saving_path)
    
    
def checkColorListAndColorRange(colorList, colorRange):
    if len(colorList) == 0:
        colorList = ["#C1182A", "#FBAD3C", "#ABD715","#0CA299","#75117E"]
    
    if len(colorRange) == 0:
        colorRange = ["all"]
    
    if (len(colorRange) != len(colorList)):
        if(len(colorRange) > len(colorList)):
            for i in range(len(colorRange) - len(colorList)):
                colorList.append('#%06X' % randint(0, 0xFFFFFF))
        else:
            colorList = colorList[:len(colorRange)]
            
    return colorList, colorRange
    
        