import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 #load the data
iris=load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
Y=iris.target
print(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train, y_train)
#pickling and save the data in file
pickle.dump(model,open('iris_model.pkl','wb'))
print("model trained and saved the data")

