# 
import streamlit as st
import pickle
import joblib
import numpy as np

def load_model(path: str):
    with open(path,'rb') as f:
        model,scaler, contry_freq_map = joblib.load(f)
    return model ,scaler,contry_freq_map
    
model,scaler, contry_freq_map = load_model("autism_model.pkl")



st.title("Autism Prediction Web App")
st.write("Enter the details below to predict whether an individual may be on the autism spectrum.")


st.header("Behavioral Scores (A1 to A10)")

scores=[]
for i in range(1,11):
     
    score= st.radio(f"A{i} Score",[0,1])
    scores.append(score)


col1, col2  = st.columns(2)


with col1:
    age = st.number_input("Age", 1, 100, 5)
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    Relation =st.selectbox("Relation",["Others","Parent","Relative","Self"])
    jaundice= st.selectbox("Jaundice",['1', "0"])
    jaundice= int(jaundice)




relation_col= [
      "relation_Others",
      "relation_Parent",
      "relation_Relative",
      
      "relation_Self",
]
relation_encoded= {col: False for col in relation_col}
selected_rel = f"relation_{Relation}"
relation_encoded[selected_rel] =True

gender_map ={"Male":0, "Female":1,"Other":2}
gender_encoded= gender_map[gender]

country = st.selectbox(
      "Country of Residence",
      contry_freq_map.index.tolist()
)

country_freq =contry_freq_map.get(
      country,
      contry_freq_map.mean()
)

features = np.array(
      [
            *scores,
            age,
            gender_encoded,
            jaundice,
            relation_encoded["relation_Others"],
            relation_encoded["relation_Parent"],
            relation_encoded["relation_Relative"],
            relation_encoded["relation_Self"],
            
            country_freq
]).reshape(1,-1)

features = scaler.transform(features)
if st.button('Predict'):
      prediction= model.predict(features)
      if prediction[0] ==1:
            st.success("The model predicts a likelihood of Autism")
      else:
            st.success("The model predicts no Autism.")  

