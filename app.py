import streamlit as st
import os
from src import linear_regression 
from src import pre_processing

# Sidebar navigation
st.sidebar.title("App Selection")
option = st.sidebar.radio("Chọn lựa chọn phù hợp:", ["Pre Processing", "Linear Regression"])

if(option == 'Pre Processing'):
    pre_processing.display()
elif(option == 'Linear Regression'):
    linear_regression.display()