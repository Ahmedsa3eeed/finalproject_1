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

##Functions to be used for plot and analysis
def pie1(x,y,z):
    """
    The function is used to plot pie charts for any feature
    x: represents the data frame to be given
    y: string represents the feature to be plotted
    z: string represents the title of the chart required 
    """
    fig=px.pie(x,str(y),title=str(z))
    return fig

def func1(x):
    """
    This function is used to unify the values of target that have same meaning but different letters(eg. wether small or capital)
    x: The class of the target
    """
    if x=='Grapes':
        return 'grapes'
    elif x=='Pomegranate':
        return 'pomegranate'
    else:
        return x
    
def func2(x):
    """
    The function is used to replace the comma by dot and remove the celsius, finally converting the data type to float
    """
    return float(x.replace(',','.').strip()[0:-3])

def hist(x,y,z):
    """
    This function is used to plot histogram for a univariate features along with the type of representing the marginal in outliers
    x: represents the data frame
    y: represents the univariate feature to be ploted
    z: represents the outlier detection appearance wether box or violin
    """
    fig2=px.histogram(df,x=str(y),marginal=str(z))
    return fig2

def func3(x):
    """
    This function is considered as a filter such that any value for temperature greater than 45 or less than zero will 
    be placed as null value
    """
    if x<0 or x>45:
        return np.nan
    else:
        return x
    
def func4(x):
    """
    The function is used to replace the comma by dot and remove the percentage, finally converting the data type to float
    """
    return float(x.replace(',','.').strip()[0:-1])


df=pd.read_csv(r"crop_recommend.csv")
df.dtypes

## First step: Read data sets, check duplicates, nulls and gather statsistical information about each feature

###### Observations: 
##For the temperature feature, humidity, ph and rainfall, the comma to be replaced with '.'  and for the temperature feature the celsius to be ##removed and for humidity the percentage to be removed.

##Check duplicates
df.duplicated().sum()## no duplicates is included in the data

##Check nulls
df.isnull().sum()
df.isnull().mean()*100

#### Observations: There are null values to be filled for features except for the target the nulls will be dropped

##Check the data type in each column
df.info()

#### Observations: The humidity, temperature, ph and rainfall features data type shall be float

##Gather brief statistical information for numerical columns
df.select_dtypes('number').describe()

#### Observations: 
#* Other features to be analyzed after converting the object into integer
#* It is expected to find outliers in both nitrogen and potassium columns as the value of median is much different than mean.

## Second step: Univariate analysis and feature engineering:
##for the target columns
df['label'].value_counts()

#### Observations: Target classes are balanced, therefore no need for undersampling or oversampling, but there are some value classes written in different letter, therefore it shall be modified

##Label column:

df['label']=df['label'].apply(func1)
df['label'].value_counts()

##second step drop the rows that have null value for target
x=list(df[df['label'].isnull()== True].index)  ##the index of the rows that have null values for the target
df.drop(x,axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

##Temperature column
##First remove the celsius sign as well as replace the ',' to with '.' 

df['temperature']=df['temperature'].apply(func2)
##plot to detect any outliers in temperature

#### Observations:Temperatures below zero celsius and above 45 degree celsius are considered as outliers and will be put as null values and Then in the preprocessing phase to be dealt with.
df['temperature']=df['temperature'].apply(func3)

##Observations: Temperatures represented as outliers in the graph are real and true values, as for example rice is planted in high temperature about 37 degree celsius

##Humidity column
##First remove the percentage sign as well as replace the ',' to with '.' 
df['humidity']=df['humidity'].apply(func4)

##plot to detect any outliers in the humidity

#### Observation:Humidity values are real and true as low values of humidity are indeication for dry areas, although some of them are represented as outliers on the graph.

##ph column
##First replace the ',' to with '.' 
df['ph']=df['ph'].str.replace(',','.')
df['ph']=df['ph'].astype('float')

#### Observation:PH values are real as the PH value can range from 0 to 14

##rain fall column
##First replace the ',' to with '.' 
df['rainfall']=df['rainfall'].str.replace(',','.')
df['rainfall']=df['rainfall'].astype('float')
df['rainfall'].dtype
px.histogram(df,x='rainfall',color_discrete_sequence=['green'],marginal='violin', title='rainfall values ')  

#### Observation:In reality the values presented on the graph as outliers are indication for rainy regions, therefore the outliers will be remained as they are and in case the accuracy of the model is not satisfactory the values will be dealth with

##Nitrogen column
df['Nitrogen'].isnull().sum()  ##there are null values in the nitrogen and will be dealt with in preprocessing phase
#### Observation:No outliers in the nitrogen column

##phosphorous feature
df['phosphorus'].isnull().sum()   ##there are null values and will be dealt with in the preprocessing phase

##potassium column
df['potassium'].isnull().sum()  ##no null values

##Bivariate analysis:
y=df