
import streamlit as st
import pandas as pd 
import sys

st.title('Crop recommendation')
st.header('About the data')
st.write('This dataset contains information on the levels of nitrogen, phosphorus, and potassium in soil, as well as temperature, humidity, pH, and rainfall, and their impact on the growth of crops.The main aim is to recommend the most convenient crop according to the type of soil with respect to the value of mentioned features')
st.write('The data consists of 7 features as follow')
st.write(' 1-Nitrogen:This column represents the amount of nitrogen (in kg/ha) present in the soil for the crop. Nitrogen is an essential nutrient required')
st.write(' 2-Phosphorus:This column represents the amount of phosphorus (in kg/ha) present in the soil for the crop')
st.write(' 3-Potassium:This column represents the amount of potassium (in kg/ha) present in the soil for the crop')
st.write(" 4-Temperature:This column represents the average temperature (in Celsius) during the crop's growing period")
st.write(' 5-Humidity:This column represents the average relative humidity (in percentage) during the crop\'s growing period')
st.write(' 6-PH:This column represents the soil pH during the crop\'s growing period. pH is a measure of the acidity or alkalinity of the soil')
st.write(' 7-rainfall:This column represents the amount of rainfall (in mm) received during the crop\'s growing period.')
st.write(' 8-Label:The label column is used to identify the type of crop')
df_source = pd.read_csv(r"C:\Data science course\Final project\final project including new data set\multipages\Sources\crop_recommend.csv", encoding= 'unicode_escape')
st.write(df_source)
