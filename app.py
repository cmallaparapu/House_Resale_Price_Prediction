import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

model = pickle.load(open("iris_model.pkl","rb"))
st.title("Iris Data classification")

st.write(model)
#input fields
sepal_length=st.number_input("sepal_length(cm",4.0,8.0,5.9)
sepal_width=st.number_input("sepal_width(cm)",2.0,7.0,4.9)
petal_length=st.number_input("petal_length(cm)",2.0,9.0,4.9)
petal_width = st.number_input("petal_width(cm)",0.1,9.1,2.1)

if st.button("Predict"):
    features=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    species=["setosa","versicolor","virginica"]
    prediction=model.predict(features)
    print("pred",prediction)
    st.write(prediction)
    st.success(f"The prediction is {species[prediction[0]]}")

