import pandas as pd 
import plotly.express as px 
import streamlit as st
import sys
sys.path.append(r'Finalproject.py')
import Finalproject as fp

df2=fp.y
##First question:
st.header('Bivariate analysis')
st.subheader(' 1- What is the effect of each feature individually on the target?')
fact=st.radio('please select the required feature to view with the target',['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph','rainfall'],horizontal=True)
st.write(px.scatter(df2,x=fact,color='label'))
if fact=='phosphorus':
    st.write('At low phosphorous in soil it is clear that crops to be planted are jute and orange,whereas at high values of phosphorous in soil only apple and oranges can be planted')

elif fact=='potassium':
    st.write('At low potassium values only orange can be planted, at high potassium values, grapes and apples can be planted and in range from 77 to 85 only chickypea can be planted')
    
elif fact=='temperature':
    st.write('At low temperature values only orange and grapes can be planted, at high temperature values, papaya can be planted')

elif fact=='humidity':
    st.write('At low humidity  values only chickpea,kidneybeans,pigeonbeans and mothbeans can be planted, at high humidity values, coconut can be planted')

elif fact=='ph':
    st.write('It is noticed that at high and low significant of ph (pure alkaline or acidic soils), mothbeans can only be planted')

elif fact=='rainfall':
    st.write('It is noticed that rice requires high rainfall in other words high amount of water, whereas muskmelon can be planted in non rainy areas')

elif fact=='Nitrogen':
    st.write('It is obvious that cotton requires high nitrogen in soil')
    
##Second question:
st.subheader(' 2-Provide recommendation for a convenient crop using two features by using scatter?')
con_pie = st.container()
col1_p, col2_p = con_pie.columns(2)

with col1_p:
    first = st.radio("Select the first feature",['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph','rainfall'])
                       
with col2_p:
    classes=['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph','rainfall']
    classes.remove(first)  ##in order to repeat the feature chosen again
    second = st.radio("Select the second feature",classes)

st.write(px.scatter(df2,x=first,y=second,color='label'))

##Third question
st.subheader(' 3-Is there any correlation between features?')
st.write(px.imshow(df2.corr(),text_auto=True))
st.write('There is a strong correlation between phosphorous and nitrogen, whereas no other correlations are found')



             
    