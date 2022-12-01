# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:49:48 2022

@author: Vikram Vasudevan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve

def calc_prevalence(y):
    return sum(y)/len(y)

def calc_specificity(y_actual, y_pred, thresh):
 # calculates specificity
 return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
 auc = roc_auc_score(y_actual, y_pred)
 accuracy = accuracy_score(y_actual, (y_pred > thresh))
 recall = recall_score(y_actual, (y_pred > thresh))
 precision = precision_score(y_actual, (y_pred > thresh))
 specificity = calc_specificity(y_actual, y_pred, thresh)
 print('AUC:%.3f'%auc)
 print('accuracy:%.3f'%accuracy)
 print('recall:%.3f'%recall)
 print('precision:%.3f'%precision)
 print('specificity:%.3f'%specificity)
 print('prevalence:%.3f'%calc_prevalence(y_actual))
 print(' ')
 return auc, accuracy, recall, precision, specificity

df = pd.read_csv('medical_appointments_dataset.csv')

noShowValueCounts = df['No-show'].value_counts()

df['OUTPUT_LABEL'] = (df['No-show'] == 'Yes').astype('int')

prevalence = calc_prevalence(df['OUTPUT_LABEL'])

df['ScheduledDay'] = pd.to_datetime(
    df['ScheduledDay'], format='%Y-%m-%dT%H:%M:%SZ')
df['AppointmentDay'] = pd.to_datetime(
    df['AppointmentDay'], format='%Y-%m-%dT%H:%M:%SZ')

badRecords = df.loc[(df['ScheduledDay'] > df['AppointmentDay'])]

assert df['ScheduledDay'].isnull().sum() == 0, 'Missing Scheduled Day'
assert df['AppointmentDay'].isnull().sum() == 0, 'Missing Appointment Day'

df['AppointmentDay'] = df['AppointmentDay'] + \
    pd.Timedelta('1d') - pd.Timedelta('1s')

badRecords = df.loc[(df['ScheduledDay'] > df['AppointmentDay'])]

# Remove all records where appointment day is lesser than scheduled day
df = df.loc[(df['ScheduledDay'] <= df['AppointmentDay'])].copy()

badRecords = df.loc[(df['ScheduledDay'] > df['AppointmentDay'])]

# Break timestamp into components so we can analyze the data
df['ScheduledDay_year'] = df['ScheduledDay'].dt.year
df['ScheduledDay_month'] = df['ScheduledDay'].dt.month
df['ScheduledDay_day'] = df['ScheduledDay'].dt.day
df['ScheduledDay_week'] = df['ScheduledDay'].dt.isocalendar().week
df['ScheduledDay_hour'] = df['ScheduledDay'].dt.hour
df['ScheduledDay_minute'] = df['ScheduledDay'].dt.minute
df['ScheduledDay_dayofweek'] = df['ScheduledDay'].dt.dayofweek

df['AppointmentDay_year'] = df['AppointmentDay'].dt.year
df['AppointmentDay_month'] = df['AppointmentDay'].dt.month
df['AppointmentDay_day'] = df['AppointmentDay'].dt.day
df['AppointmentDay_week'] = df['ScheduledDay'].dt.isocalendar().week
df['AppointmentDay_hour'] = df['AppointmentDay'].dt.hour
df['AppointmentDay_minute'] = df['AppointmentDay'].dt.minute
df['AppointmentDay_dayofweek'] = df['AppointmentDay'].dt.dayofweek

df.head()

df.info()

byYear = df.groupby('AppointmentDay_year').size()
byMonth = df.groupby('AppointmentDay_month').size()
byDayOfWeek = df.groupby('AppointmentDay_dayofweek').size()

isDayOfWeekPredictive = df.groupby('AppointmentDay_dayofweek').apply(
    lambda g: calc_prevalence(g.OUTPUT_LABEL.values))

df['delta_days'] = (df['AppointmentDay'] - df['ScheduledDay']
                    ).dt.total_seconds()/(60*60*24)


plt.hist(df.loc[df.OUTPUT_LABEL == 1, 'delta_days'],
         label='Missed', bins=range(0, 60, 1))
plt.hist(df.loc[df.OUTPUT_LABEL == 0, 'delta_days'],
         label='Not Missed', bins=range(0, 60, 1), alpha=0.5)
plt.legend()
plt.xlabel('Days until appointment')
plt.ylabel('distribution')
plt.xlim(0, 40)
plt.show()

# shuffle the sample
df = df.sample(n=len(df), random_state=42)
df = df.reset_index(drop=True)

# 30% validation set
df_validation = df.sample(frac=0.3, random_state=42)
# remaining 70% training set
df_training = df.drop(df_validation.index)

print('Valid prevalence(n= % d): % .3f' %
      (len(df_validation), calc_prevalence(df_validation.OUTPUT_LABEL.values)))
print('Train prevalence(n=% d): % .3f' %
      (len(df_training), calc_prevalence(df_training.OUTPUT_LABEL.values)))

col2use = ['ScheduledDay_day', 'ScheduledDay_hour',
           'ScheduledDay_minute', 'ScheduledDay_dayofweek',
           'AppointmentDay_day', 'AppointmentDay_dayofweek', 'delta_days']

x_training = df_training[col2use].values
x_validation = df_validation[col2use].values

y_training = df_training['OUTPUT_LABEL'].values
y_validation = df_validation['OUTPUT_LABEL'].values

rf = RandomForestClassifier(max_depth = 5, n_estimators = 100, random_state = 42)
rf.fit(x_training,y_training)
y_training_preds = rf.predict_proba(x_training)[:,1]
y_validation_preds = rf.predict_proba(x_validation)[:,1]

threshold = 0.201
print('Random Forest')
print('Training')
print_report(y_training, y_training_preds, threshold)
print('Validation')
print_report(y_validation, y_validation_preds, threshold)

fpr_train, tpr_train, thresholds_train = roc_curve(y_training, y_training_preds)
auc_train = roc_auc_score(y_training, y_training_preds)
fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_validation, y_validation_preds)
auc_valid = roc_auc_score(y_validation, y_validation_preds)
plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)
plt.plot([0,1],[0,1],'k-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()