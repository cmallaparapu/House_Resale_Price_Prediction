from io import BytesIO
import pandas as pd
import streamlit as st
import re
import time
import numpy as np
from click import option
from joblib  import  load
import pickle
import gdown
import os
import requests



FILE_ID = "1o-v4eoUged74SIMHVyOt4aYdmIKw1f2f"
@st.cache_resource
def load_model_from_drive():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    output = "RandomForestRegressor_small.joblib"

    # ✅ Use gdown to properly fetch large files
    gdown.download(url, output, quiet=False)

    models = load(output)
    return models

rand_model = load_model_from_drive()

st.title("🏠 House Resale Price Prediction")
model=st.sidebar.selectbox('Select model to use',options=(['RandomForestRegressor']))
# User inputs
town = st.text_input("Town",key="town")
flat_type = st.selectbox("Flat-type",['select here',
    '1 room', '2 room', '3 room', '4 room', '5 room', 'executive',
    'multi generation', 'multi-generation'
],key="flat_type")
street_name = st.text_input("Street Name",key="street_name")
storey_range = st.text_input("Storey Range (e.g., 01 TO 03)",key="storey_range")
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=200.0, value=28.0,key="floor_area_sqm")
lease_commence_date = st.number_input("Lease Commence Year", min_value=1960, max_value=2030, value=1966,key="lease_commence_date")
month = st.text_input("Month (YYYY-MM)",key="month")




@st.cache_resource
def load_encoder(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# 🚀 Load resources only once
with st.spinner("Loading model and encoders..."):
    rand_model = rand_model
    flat_type_encoder = load_encoder('flat_type_encoder.pkl')
    town_encoder = load_encoder('town_encoder.pkl')
    street_name_encoder = load_encoder('street_name_encoder.pkl')


# Encode categorical features safely
try:
    flat_type_encoded = flat_type_encoder.fit_transform([flat_type.lower()])
except:
    st.error("⚠️ Unknown flat type entered.")
    st.stop()

try:
    town_encoded = town_encoder.fit_transform([town])
except:
    st.error("⚠️ Unknown town entered.")
    st.stop()

try:
    street_name_encoded = street_name_encoder.fit_transform([street_name])
except:
    st.error("⚠️ Unknown street name entered.")
    st.stop()

# Process month input
year_month = 0
if month:
    try:
        month_dt = pd.to_datetime(month, format="%Y-%m")
        year = month_dt.year
        month_num = month_dt.month
        year_month = round(year + month_num / 12, 2)
    except:
        st.error("❌ Invalid month format. Please use YYYY-MM{1990-02}.")
        st.stop()

# Process storey range
avg_storey = None
if storey_range:
    match = re.findall(r'\d+', storey_range)
    if len(match) == 2:
        lower_storey = int(match[0])
        upper_storey = int(match[1])
        avg_storey = round((lower_storey + upper_storey) / 2, 1)
    else:
        st.error("❌ Invalid storey range format. Example: '01 TO 03'")
        st.stop()

# Prepare input dataframe
Data = pd.DataFrame([{
    'town': town_encoded,
    'flat_type': flat_type_encoded,
    'street_name': street_name_encoded,
    'storey_range': avg_storey,
    'floor_area_sqm': floor_area_sqm,
    'lease_commence_date': lease_commence_date,
    'year_month': year_month
}])


# Prediction
if st.button("Predict Resale Price"):
    predicted = rand_model.predict(Data)
    st.success(f"💰 Estimated Resale Price: ${predicted[0]:,.2f}")

    with st.spinner("⏳ Resetting form in 1 minute..."):
        time.sleep(60)

    for key in [
        "town_input", "flat_type_input", "street_input", "storey_input",
        "floor_input", "model_input", "lease_input", "month_input"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()
