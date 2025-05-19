import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
path = '/nfs/turbo/coe-rbg/mmakar/tableshift/melanoma-classification'
#isic_train.csv  jpeg  sample_submission.csv  test  test.csv  tfrecords  train  train.csv
train_df = pd.read_csv(os.path.join(path, 'train.csv'))
isic_df = pd.read_csv(os.path.join(path, 'isic_training_cleaned.csv'))

#print(train_df.columns)
#print(train_df.shape[0])
print(isic_df.columns)
print(isic_df.shape[0])
first_row = train_df.iloc[0]
#print(first_row)
unique_diagnoses = train_df["diagnosis"].unique()
#print(unique_diagnoses)
scale_ct = pd.crosstab(isic_df['scale'], isic_df['target'], margins=True)
print("Scale vs Target:\n", scale_ct)
"""
Diagnosis: ['unknown' 'nevus' 'melanoma' 'seborrheic keratosis' 'lentigo NOS'
 'lichenoid keratosis' 'solar lentigo' 'cafe-au-lait macule'
 'atypical melanocytic proliferation']
 
"""