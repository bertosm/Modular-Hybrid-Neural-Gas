# -*- coding: utf-8 -*-
'''
Created on Wed Sep  1 17:39:26 2021

@author: Bertosm
'''

from joblib import load
from os.path import join

import pandas as pd

datasetpath = "C:/Users/alber/Desktop/GNG/Datasets/CN-MCI-AD/"
datasetFile = 'CN-MCI-AD-ADNI1-prepared_data-20210901_17h20m.pkl'

s= join(datasetpath, datasetFile)
a = load(s)

# print(a[0].keys())

# print('\n', a[0]['projected_scaled_data_dict'].keys())
# print('\n', a[0]['class_balanced_projected_scaled_data_dict'].keys())
# print('\n', a[0]['partitioned_class_balanced_projected_scaled_data_dict'].keys())


# print("\n a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected']: ",
#       a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected'].shape)

# print("\n a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected']: ",
#       a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected'])

# print("\n a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['FactorAnalysis']: ",
#       a[0]['projected_scaled_data_dict']['Unscaled___num_components=2']['FactorAnalysis'].shape)

# print("\n a[0]['class_balanced_projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected']: ",
#       a[0]['class_balanced_projected_scaled_data_dict']['Unscaled___num_components=2']['Unprojected'].shape)
# print("\n a[0]['class_balanced_projected_scaled_data_dict']['Unscaled___num_components=2']['FactorAnalysis']: ",
#       a[0]['class_balanced_projected_scaled_data_dict']['Unscaled___num_components=2']['FactorAnalysis'].shape)

# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_train']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split RobustScaler___num_components=4 FactorAnalysis']['data_train'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_test']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_test'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_train']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_train'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_test']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_test'].shape)

# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_train']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_train'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_test']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['data_test'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_train']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_train'].shape)
# print("\n  a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_test']: ",
#       a[0]['partitioned_class_balanced_projected_scaled_data_dict']['train_test_split Unscaled___num_components=2 Unprojected']['labels_test'].shape)


datasetToSave = a[0]['class_balanced_projected_scaled_data_dict']['Unscaled___num_components=4']['FactorAnalysis']
## convert your array into a dataframe
df = pd.DataFrame(datasetToSave)

## save to xlsx file

filepath = 'C:/Users/Bertosm/Desktop/GNG-Alzheimer-Comciencia/Datasets/CN-MCI-AD/CN-MCI-AD_balanced_scaled-robustScaler_projected-FactorAnalysis_nComponents-4.xlsx'

df.to_excel(filepath, index=False)

