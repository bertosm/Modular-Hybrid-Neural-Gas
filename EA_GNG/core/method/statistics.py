#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:48:37 2020

@author: Berto Sosa
"""
import pandas as pd
import math

from scipy import stats
# from statsmodels.stats.anova import AnovaRM


def nonParametric2classes(x, y, alpha = 0.05):
    stat, pvalue = stats.mannwhitneyu(x, y)
    # print("stat:",stat, "-- pvalue:", pvalue)
    if pvalue < alpha:
        resolution = "hay significación estadística de los dos grupos";
    else:
        resolution = "NO hay significación estadística de los dos grupos";
    return "Mann-Whitney U", stat, pvalue, resolution




def nonParametric3orMoreClasses(arg, alpha=0.05):
    stat, pvalue = stats.kruskal(*arg)
    if pvalue < alpha:
        resolution = "hay significación estadística de los grupos";
    else:
        resolution = "NO hay significación estadística de los grupos";
    return "Kruskal", stat, pvalue, resolution




def nonParametric2classesPaired(x, y, alpha=0.05):
    stat, pvalue = stats.wilcoxon(x, y)
    if pvalue < alpha:
        resolution = "hay significación estadística de los dos grupos";
    else:
        resolution = "NO hay significación estadística de los dos grupos";
    return "Wilcoxon", stat, pvalue, resolution




def nonParametric3orMoreClassesPaired(arg, alpha=0.05):
    stat, pvalue = stats.friedmanchisquare(*arg)
    if pvalue < alpha:
        resolution = "hay significación estadística de los grupos";
    else:
        resolution = "NO hay significación estadística de los grupos";
    return "Friedman Chi-square", stat, pvalue, resolution
   

    

def parametric2classes(x, y, alpha=0.05):
    stat, pvalue = stats.ttest_ind(x, y)
    if pvalue < alpha:
        resolution = "hay significación estadística de los dos grupos";
    else:
        resolution = "NO hay significación estadística de los dos grupos";
    return "Student’s t-Test", stat, pvalue, resolution




def parametric3orMoreClasses(arg, alpha=0.05):
    stat, pvalue = stats.f_oneway(*arg)
    if pvalue < alpha:
        resolution = "hay significación estadística de los grupos";
    else:
        resolution = "NO hay significación estadística de los grupos";
    return "ANOVA one way", stat, pvalue, resolution




def parametric2classesPaired(x, y, alpha=0.05):
    stat, pvalue = stats.ttest_rel(x, y)
    if pvalue < alpha:
        resolution = "hay significación estadística de los dos grupos";
    else:
        resolution = "NO hay significación estadística de los dos grupos";
    return "Paired Student’s t-Test", stat, pvalue, resolution




def parametric3orMoreClassesPaired(df, target, alpha=0.05):
    """test recomendado: Repeated anova measures ANOVA test // No implementado"""

    return "Repeated Measures ANOVA Test-NotImplemented", 0, -1, "test ANOVARM no implementado"




   
def selectStatistic(df, target = 'DX_bl', parametric = False, paired=False, alpha = 0.05):      
        
    if target not in df.columns:
        print("please specify the label column")
        return
        
    unique = pd.unique(df[target])
    listaDF = list()
    for label in unique:
        dfaux = df[df[target] == label].drop(target, axis=1)
        listaDF.append(dfaux)
        
    if parametric:
        if unique.shape[0] == 2 and paired:
            return parametric2classesPaired(listaDF[0], listaDF[1], alpha)
        elif unique.shape[0] == 2 and not paired:
            return parametric2classes(listaDF[0], listaDF[1], alpha)
        elif unique.shape[0] > 2 and paired:
            return parametric3orMoreClassesPaired(listaDF, alpha)
        else:
            return parametric3orMoreClasses(df, target, alpha)
    else:
        if unique.shape[0] == 2 and paired:
            return nonParametric2classesPaired(listaDF[0], listaDF[1], alpha)
        elif unique.shape[0] == 2 and not paired:
            return nonParametric2classes(listaDF[0], listaDF[1], alpha)
        elif unique.shape[0] > 2 and paired:
            return nonParametric3orMoreClassesPaired(listaDF, alpha)
        else:
            return nonParametric3orMoreClasses(listaDF, alpha)
        
    

def isPaired():
    print("Does the dataset have paired data?\nEnter 'y', 'yes', 'n' or 'no'.")
    respose = input()
    paired = False
    if respose in ('y', 'yes', 'Y', 'YES'):
        paired = True
    elif respose not in ('n', 'no', 'N', 'NO'):
        print("Not valid, please Enter 'y', 'yes', 'n' or 'no'.")
        paired = None
    return paired



def isParametric(df, alfa = 0.05):
    parametric = False
    stat, pvalue = stats.shapiro(df)
    if not pvalue <= alfa:
        parametric = True
    return parametric


def round_decimals_down(number, decimals=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor