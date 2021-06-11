#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator,TransformerMixin): 
    
 
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    

    def transform(self, X, y=None, copy=None):
        
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    
class absenteeism_model():
    
    def __init__(self, model_file, scalar_file):
        with open('model', 'rb') as model_file, open('scalar', 'rb') as scalar_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scalar_file)
            self.data = None
            
    def load_and_clean_data(self, data_file):
        
        df = pd.read_csv(data_file, delimiter=',')
        self.df_with_predictions = df.copy()
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'
        
        reasons_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        reasons_columns_type1 = reasons_columns.loc[:,1:14].max(axis=1)
        reasons_columns_type2 = reasons_columns.loc[:,15:17].max(axis=1)
        reasons_columns_type3 = reasons_columns.loc[:,18:21].max(axis=1)
        reasons_columns_type4 = reasons_columns.loc[:,22:].max(axis=1)
        
        df = df.drop(['Reason for Absence'], axis=1)
        df = pd.concat([df, reasons_columns_type1, reasons_columns_type2, reasons_columns_type3, reasons_columns_type4], axis=1)
        
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_', 'Reason_4']
        
        df.columns = column_names
        
        columns_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[columns_names_reordered]
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        
        def date_to_week(date_value):
            return date_value.weekday()
        
        df['Day of week'] = df['Date'].apply(date_to_week)
        
        df = df.drop(['Date'], axis=1)
        
        columns_names_upd = [ 'Reason_1', 'Reason_2', 'Reason_', 'Reason_4', 'Day of week','Transportation Expense', 'Distance to Work', 'Age',
                               'Daily Work Load Average', 'Body Mass Index', 'Education',
                               'Children', 'Pets', 'Absenteeism Time in Hours']
        
        df = df[columns_names_upd]
        
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        df = df.fillna(value=0)
        
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        self.preprocessed_data = df.copy()
        
        self.data = self.scaler.transform(df)
        
    def predicted_probablity(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
        
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probablity'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

