import streamlit as st
import os

# Sidebar navigation
st.sidebar.title("App Selection")
option = st.sidebar.radio("Chọn lựa chọn phù hợp:", ["Pre Processing", "Linear Regression"])

if(option == 'Pre Processing'):
    import pre_processing
    pre_processing.display()
elif(option == 'Linear Regression'):
    import linear_regression
    linear_regression.display()