import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('model.h5')

tenure = st.slider("Tenure", 1,72,23, step = 1)
monCharges = st.number_input("Charge", 20.5, 200.5, 50, step=0.05)
totChargers = st.number_input("Total Charges", 20.5, 10000.5, 2000.0, step = 0.05)
contract = st.radio("Contract", ("Month-to-Month", "One Year",'Two Year'))
onlineSecurity = st.radio("Online Security", ("No", "Yes","No Internet Service"))
techSupport = st.radio("Technician Support", ("No", "Yes","No Internet Service"))

myDict = {"monCharges" : , "totCharges", 'tenure', }