import pandas as pd 
import plotly.express as px 
import streamlit as st
import sys
sys.path.append(r'Finalproject.py')
import Finalproject as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate , train_test_split , StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, confusion_matrix, classification_report
import joblib

df2=fp.y
st.subheader('First:Label encoding for the target and drop the target from the data frame')
le=LabelEncoder()
df2['label']=le.fit_transform(df2['label'])
y=df2['label']
x=df2.drop(['label'],axis=1)
st.write('The data frame after removing the target')
st.write(x)
st.write('The target after encoding')
st.write(y)
st.subheader('Cross validation to identify the best model for implementation (model which has lowest train error and least overfit)')  
models=[LogisticRegression(multi_class='ovr'), KNeighborsClassifier(),
        RandomForestClassifier(),DecisionTreeClassifier()]
for model in models:
    st.write(model)
    pl=make_pipeline(KNNImputer(n_neighbors=5),RobustScaler(), model)
    scores = cross_validate(estimator = pl , X = x , y = y , cv = StratifiedKFold(n_splits=5) ,
                        scoring='accuracy' , return_train_score=True )
    st.write(scores['train_score'].mean())
    st.write(scores['test_score'].mean())
    
    st.write('-------------------------------------')
            
st.subheader('Hyperparameter tuning')
st.write(' 1-KNeighbors classifier')
pl3=make_pipeline(KNNImputer(),RobustScaler(), KNeighborsClassifier())
params=[{'knnimputer__n_neighbors':list(range(1,25)),
    'kneighborsclassifier__n_neighbors':list(range(1,25,2))
}]
grid_search=GridSearchCV(estimator=pl3,param_grid=params,cv=StratifiedKFold(n_splits=5),scoring='accuracy')
grid_search.fit(x,y)
model1=grid_search.best_estimator_
scores = cross_validate(estimator = model1 , X = x , y = y , cv = StratifiedKFold(n_splits=5) ,
                scoring='accuracy' , return_train_score=True )
st.write('The train score of KNeigbors classifier is ',scores['train_score'].mean())
st.write('The average test score for KNeighbors classifier is ',scores['test_score'].mean())

st.write('2-Decision tree')
pl4=make_pipeline(KNNImputer(),RobustScaler(), DecisionTreeClassifier())
params=[{'knnimputer__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
    'decisiontreeclassifier__criterion':['gini','entropy'], 'decisiontreeclassifier__max_depth':list(range(1,20))
}]
grid_search=GridSearchCV(estimator=pl4,param_grid=params,cv=StratifiedKFold(n_splits=5),scoring='accuracy')
grid_search.fit(x,y)
pl4=make_pipeline(KNNImputer(n_neighbors=1),RobustScaler(), DecisionTreeClassifier(max_depth=18,criterion='gini'))
scores = cross_validate(estimator = pl4 , X = x , y = y , cv = StratifiedKFold(n_splits=5) ,
                scoring='accuracy' , return_train_score=True )
print(scores['train_score'].mean())
print(scores['test_score'].mean())

st.subheader('3-Random forest model')
pl5=make_pipeline(KNNImputer(),RobustScaler(), RandomForestClassifier())
params=[{'randomforestclassifier__n_estimators':list(range(100,105)),
     'randomforestclassifier__max_depth':list(range(1,15))}]
grid_search=GridSearchCV(estimator=pl5,param_grid=params,cv=StratifiedKFold(n_splits=5),scoring='accuracy')
grid_search.fit(x,y)
model=grid_search.best_estimator_
scores = cross_validate(estimator = model , X = x , y = y , cv = StratifiedKFold(n_splits=5) ,
                scoring='accuracy' , return_train_score=True )
print(scores['train_score'].mean())
print(scores['test_score'].mean())
joblib.dump(model, "model.pkl")
joblib.dump(x.columns, "Inputs.pkl")




             
    