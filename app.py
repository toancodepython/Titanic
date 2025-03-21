import streamlit as st
import os
from src import linear_regression 
from src import pre_processing
from src import svm_mnist
from src import decision_tree_mnist
from src import clustering
from src import mlflow_web
from src import neural

# Sidebar navigation
option = st.sidebar.selectbox("Lựa chọn", ["Titanic Data", "Linear Regression", "SVM Mnist", "Decision Tree Mnist",  "Clustering", "Neural Network", "ML-Flow"])

if(option == 'Titanic Data'):
    pre_processing.display()
elif(option == 'Linear Regression'):
    linear_regression.display()
elif(option == 'SVM Mnist'):
    svm_mnist.display()
elif(option == 'Decision Tree Mnist'):
    decision_tree_mnist.display()
elif(option == 'Clustering'):
    clustering.display()
elif(option == 'ML-Flow'):
    mlflow_web.display()
elif(option == 'Neural Network'):
    neural.display()
