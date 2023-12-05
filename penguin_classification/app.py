import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example/csv)
""")
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill length (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass(g)', 2700.0, 6300.0, 4207.0)

        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
        }

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
# print(input_df)
length = input_df.shape[0]
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)
# df = input_df
# print(df)
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]
print(df)

df = df[:length] #select only the first row (the user input data)
print(df)
st.subheader('User Input features')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make prediction
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
print(prediction_proba)
st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
data_predict_species = penguins_species[prediction]
column_name = ['Predict_species']
df_predict = pd.DataFrame(data = data_predict_species,
                          columns = column_name)
df_predict_overall = pd.concat([df, df_predict], axis = 1)
st.write(df_predict_overall)

st.subheader('Prediction Probability')
column_name = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba = pd.DataFrame(data = prediction_proba,
                          columns = column_name)
df_prediction_proba_overall = pd.concat([df, df_prediction_proba, df_predict], axis = 1)
st.write(df_prediction_proba_overall)


