# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:39:26 2021

@author: Bertosm
"""

import pickle


# with open("C:/Users/Bertosm/Desktop/GNG-Alzheimer-Comciencia/Datasets/CN-MCI-AD/CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl", "rb") as f:
#     data = pickle.load(f)
    
from joblib import load
from os.path import join

s= join("C:/Users/Bertosm/Desktop/GNG-Alzheimer-Comciencia/Datasets/CN-MCI-AD/", "CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl")
a = load(s)

print(a[0].keys())

print("\n", a[0]["projected_scaled_data_dict"].keys())
print("\n", a[0]["class_balanced_projected_scaled_data_dict"].keys())
print("\n", a[0]["partitioned_class_balanced_projected_scaled_data_dict"].keys())
