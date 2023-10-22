import numpy as np
import pandas as pd
import plotly.express as px
import warnings
import sys
sys.path.append(r'..\Finalproject.py')
import Finalproject as fp
import streamlit as st

tab_overall, tab_univariate=st.tabs(['Overall view and priliminary checks','Univariate and feature engineering'])

with tab_overall:
    ##read data
    df = pd.read_csv(r"crop_recommend.csv", encoding= 'unicode_escape')
    st.write(df)
    st.write('Observations:')
    st.markdown("For the temperature feature, humidity, ph and rainfall, the comma to be replaced with '.'  and for the temperature feature the celsius to be removed and for humidity the percentage to be removed")              
    st.subheader('First step: Read data sets, check duplicates, nulls and gather statsistical information about each feature')

    ##Check duplicates
    st.write(' 1-No duplicates are found')

    ##Check nulls
    st.write(' 2- Check the nulls:')
    st.write(df.isnull().sum())
    st.write('Observations:')
    st.write('There are null values to be filled for features except for the target the nulls will be dropped')

    ##Check information for type of featrures
    st.write(' 3-Check data type of each column')
    st.write(df.dtypes)
    st.write('Observations:')
    st.write("For the temperature feature, humidity, ph and rainfall, the comma to be replaced with '.'  and for the temperature feature the celsius to be removed and for humidity the percentage to be removed.")

    ##Brief explanation to numerical columns
    st.write(' 4-Some stastitical information for numerical columns: ')
    st.write(df.select_dtypes('number').describe())
    st.write('Observations: ')
    st.write('* Other features to be analyzed after converting the object into integer')
    st.write('* It is expected to find outliers in both nitrogen and potassium columns as the value of median is much different than mean.')

with tab_univariate:
    option=st.selectbox('Please select the desired column for feature engineering and univariate analysis preview',('label','temperature','humidity','ph','rainfall','Nitrogen','phosphorus','potassium'))
    if option=='label':
        st.subheader('Check the unique values and the balancing of the target: ')
        st.write(df['label'].value_counts())
        st.write('Observations')
        st.write('Target classes are balanced, therefore no need for undersampling or oversampling, but there are some value classes written in different letter, therefore it shall be modified')
        

        df['label']=df['label'].apply(fp.func1)
        df['label'].value_counts()
        st.subheader('Check for the null values')
        st.write('The number of null values in the target are:', df['label'].isnull().sum(),'accordingly the values to be dropped')
        st.write(fp.pie1(df,'label','Distribution of crops in data'))
       
    if option=='temperature':
        st.write(df['temperature'])
        st.write('Observations:')
        st.write("First remove the celsius sign as well as replace the ',' to with '.' ")
        df['temperature']=df['temperature'].apply(fp.func2)
        ##plot to detect any outliers in temperature
        st.subheader('Distribution of temperature and outliers')
        st.write(px.histogram(df,x='temperature',marginal='box',color_discrete_sequence=['black']))
        st.write('Observations:')
        st.write('Temperatures below zero celsius and above 45 degree celsius are considered as outliers and will be put as null values and Then in the preprocessing phase to be dealt with.')
        df['temperature']=df['temperature'].apply(fp.func3)
        st.subheader('Distribution of temperature and outliers after removing unreal values')
        st.write(px.histogram(df,x='temperature',marginal='box',color_discrete_sequence=['green']))
        st.write('Observations')
        st.write('Temperatures represented as outliers in the graph are real and true values, as for example rice is planted in high temperature about 37 degree celsius')
        st.subheader('Convert the temperature value to log scale and verify wether the log scale dealt with outlier values')
        st.write(px.histogram(x=np.log(df['temperature']),marginal='box'))
        st.write('Observations')
        st.write('No difference in outliers on graph when converting the feature to log scale')
        st.subheader('Statistical information for temperature feature')
        st.write(df['temperature'].describe())

        
    if option=='humidity':
        st.write(df['humidity'])
        st.write('Observations')
        st.write("remove the percentage sign as well as replace the ',' to with '.' ")
        df['humidity']=df['humidity'].apply(fp.func4)
        st.subheader('Distribution for humidity and outliers')
        st.write(fp.hist(df,'humidity','violin'))
        st.write('Observations')
        st.write('Humidity values are real and true as low values of humidity are indeication for dry areas, although some of them are represented as outliers on the graph.')
        st.subheader('Distribution of humidity with log scale')
        st.write(px.histogram(df,x='humidity',marginal='box',color_discrete_sequence=['red']))
        st.write('Observations')
        st.write('No difference in outliers on graph when converting the feature to log scale')
        st.subheader('Statistical information for humidity feature')
        st.write(df['humidity'].describe())
        
    if option=='ph':
        st.write(df['ph'])
        st.write('Observations')
        st.write("replace the ',' to with '.' ")
        df['ph']=df['ph'].str.replace(',','.') 
        df['ph']=df['ph'].astype('float')
        st.subheader('Distribution for ph and outliers')
        st.write(px.histogram(df,x='ph',color_discrete_sequence=['yellow'],marginal='box', title='ph values '))
        st.write('Observations')
        st.write('PH values are real as the PH value can range from 0 to 14.')
        st.subheader('Statistical information for PH feature')
        st.write(df['ph'].describe())
        
    if option=='rainfall':
        st.write(df['rainfall'])
        st.write('Observations')
        st.write("replace the ',' to with '.' ")
        df['rainfall']=df['rainfall'].str.replace(',','.') 
        df['rainfall']=df['rainfall'].astype('float')
        st.subheader('Distribution for rainfall and outliers')
        st.write(px.histogram(df,x='ph',color_discrete_sequence=['violet'],marginal='box', title='rainfall values '))
        st.write('Observations')
        st.write('In reality the values presented on the graph as outliers are indication for rainy regions, therefore the outliers will be remained as they are and in case the accuracy of the model is not satisfactory the values will be dealth with')
        st.subheader('Statistical information for rainfall feature')
        st.write(df['rainfall'].describe())
        
    if option=='Nitrogen':
        st.write(df['Nitrogen'])
        st.write(px.histogram(df,x='Nitrogen',color_discrete_sequence=['purple'],marginal='box', title='Distribution of nitrogen values ') )
        st.write('Observations')
        st.write('No outliers in nitrogen feature')
        st.subheader('Statistical information for nitrogen feature')
        st.write(df['Nitrogen'].describe())
        
    if option=='phosphorus':
        st.write(df['phosphorus'])
        st.write(px.histogram(df,x='phosphorus',color_discrete_sequence=['black'],marginal='box', title='phosphorous values ')) 
        st.subheader('Statistical information for phosphorous feature')
        st.write(df['phosphorus'].describe())
        
    if option=='potassium':
        st.write(df['potassium'])
        st.write(px.histogram(df,x='potassium',color_discrete_sequence=['red'],marginal='box', title='potassium values ')) 
        st.subheader('Statistical information for potassium feature')
        st.write(df['potassium'].describe())
        
        
    
        
           
        
        

        


                 

        
        
          
 
    
             
    


     
