#NewApp

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Import Data and Clean
s_df = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

#Subset Dataframe
ss_df = pd.DataFrame({
    'sm_li': clean_sm(s_df['web1h']),
    "income":np.where(s_df["income"] > 9, np.nan, s_df['income']),
    "education":np.where(s_df["educ2"] > 8, np.nan, s_df['educ2']),
    "parent":np.where(s_df["par"] == 1, 1, 0),
    'married':np.where(s_df['marital'] == 1, 1, 0),
    "gender":np.where(s_df["gender"] == 2, 1, 0),
    "age":np.where(s_df["age"] > 98, np.nan, s_df['age'])
})

#Drop Missing Values
ss_df = ss_df.dropna()

# Target (y) and feature(s) selection (X)
y = ss_df["sm_li"]
X = ss_df[["income", "education", "parent", 'married', "gender", "age"]]

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X.values, y)

#StreamLit App Build Out
st.title("Dom's Basic LinkedIn Predictor App")

#Income Question Setup
income= st.selectbox('Income Level', options=["< 10,000", 
                                      '10,000 - 20,000',
                                       '20,000 - 30,000',
                                       '30,000 - 40,000',
                                       '40,000 - 50,000',
                                       '50,000 - 75,000',
                                       '75,000 - 100,000',
                                       '100,000 - 150,000',
                                       '150,000 >'
                                       ])

if income =="< 10,000" : income = 1
elif income == '10,000 - 20,000': income = 2
elif income == '20,000 - 30,000': income = 3
elif income == '30,000 - 40,000': income = 4
elif income == '40,000 - 50,000': income = 5
elif income == '50,000 - 75,000': income = 6
elif income == '75,000 - 100,000': income = 7
elif income == '100,000 - 150,000': income = 8
else: income = 9


# Education Question Setup
education = st.selectbox("Education level",
    options = ["< High School",
               "High School Diploma",
               "High School Graduate",
               "Some College, No Degree",
               "2-year Associate Degree",
               "4-year College",
               "Some Post Grad or professional schooling, no postgraduate degree",
               "Postgraduate or professional schooling, including master's, doctorate, medical, or law degree"])

if education == "< High School" :education = 1
elif education == "High School Diploma": education = 2
elif education == "High School Graduate": education = 3
elif education == "Some College, No Degree" : education = 4
elif education == "2-Year Associate Degree" : education = 5
elif education == "4-Year College": education = 6
elif education == "Some Post Grad or professional schooling, no postgraduate degree": education = 7
else: education = 8

#Parent Question Setup
parent = st.radio('Are you a Parent?', options= ['Yes',
                                        'No'])

if parent == "Yes" : parent = 1
else: parent = 0

#Married Question Setup
married = st.radio('Are you married?', options= ['Yes',
                                        'No'])
if married == "Yes" : married = 1
else: married = 0

#Gender Question Setup
gender= st.radio('Gender', options= ['Male', 
                             'Female'])

if gender == "Female" : gender = 1
else: gender = 0

#Age Question Slider
age = st.slider( 'Age', 1, 97, 18)

#Output Final Probability
user= [income, education, parent, married, gender, age]

st.write(f'The Probability of you being on LinkedIn is: {round(lr.predict_proba([user])[0][1]*100, 2)}')

st.write(f' Are you a User on LinkedIn? 1= Yes 0= No: {lr.predict([user])}')