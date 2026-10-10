import streamlit as st
import joblib
from sklearn.datasets import load_iris

# load model
model = joblib.load("iris_model.pk1")

#load flower names
iris = load_iris()

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

 
st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower  measurements below to predict the Iris species.")

st.sidebar.header("🌸 Flower Measurements")

sepal_lenght = st.sidebar.number_input("Sepal Lenght", value=5.1)
sepal_width = st.sidebar.number_input("Sepal Width", value=3.5)
petal_lenght = st.sidebar.number_input("Petal lenght", value=1.4)
petal_width  = st.sidebar.number_input("Petal Width", value=0.2)

if st.button("Predict"):
    data = [[
        sepal_lenght,
        sepal_width,
        petal_lenght,
        petal_width
    ]]

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    flower_name = iris.target_names[prediction[0]]
    comfidence = probability[0] [prediction[0]] * 100

    st.success (f"🌸 Flower:{flower_name}")
    st.info("f🎯 Confidence:{confidence:.2f}%")

    