import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
def load_data():
    train_data = pd.read_csv("data/mnist/train.csv")
    X = train_data.iloc[:, 1:].values / 255.0
    y = train_data.iloc[:, 0].values
    return train_data, X, y

def show_sample_images():
    train_data = pd.read_csv("data/mnist/train.csv")
    unique_labels = train_data.iloc[:, 0].unique()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    label_count = 0
    
    for i, ax in enumerate(axes.flat):
        if label_count >= len(unique_labels):
            break
        sample = train_data[train_data.iloc[:, 0] == unique_labels[label_count]].iloc[0, 1:].values.reshape(28, 28)
        ax.imshow(sample, cmap='gray')
        ax.set_title(f"Label: {unique_labels[label_count]}", fontsize=10)
        ax.axis("off")
        label_count += 1
    st.pyplot(fig)

def plot_label_distribution(y):
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Label Distribution in Dataset")
    ax.set_xlabel("Digit Label")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def display():
    st.title("🖼️ MNIST Classification using SVM")
    st.header("📌 Step 1: Understanding Data")
    st.write("Below are some sample images from the dataset:")

    show_sample_images()

    st.write("🔹 The pixel values are normalized by dividing by 255 to scale them between 0 and 1, which helps improve model performance and convergence speed.")
    train_data, X, y = load_data()
    st.write("📊 First few rows of the dataset:")
    st.dataframe(train_data.head())
    st.write(f"📏 Dataset Shape: {train_data.shape}")

    st.write("📊 Label Distribution:")
    plot_label_distribution(y)

    if st.button("Proceed to Training 🚀"):
        st.session_state['train_ready'] = True

    if 'train_ready' in st.session_state:
        st.header("📌 Step 2: Training Model")
        
        with st.form(key='train_form'):
            col1, col2 = st.columns(2)
            with col1:
                train_ratio = st.number_input("🔢 Enter training size ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
            with col2:
                val_ratio = st.number_input("🔢 Enter validation size ratio", min_value=0.05, max_value=0.9, value=0.15, step=0.05)
            
            test_ratio = 1 - train_ratio - val_ratio
            submit_button = st.form_submit_button("Train Model 🎯")
        
        if test_ratio <= 0:
            st.error("❌ Invalid split! The sum of train, validation, and test sizes must be 1.")
        elif submit_button:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)
            
            st.write("⏳ Training SVM Model...")
            model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            with open("svm_mnist_model.pkl", "wb") as model_file:
                pickle.dump(model, model_file)
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            st.success(f"✅ Model Accuracy: {accuracy:.4f}")
            st.subheader("📊 Classification Report")
            st.dataframe(pd.DataFrame(class_report).transpose())

    if 'model' in st.session_state:
        
        st.header("📌 Step 4: Predict Custom Digit")
        uploaded_file = st.file_uploader("📤 Upload  grayscale image of a digit", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            from PIL import Image
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            image_array = np.array(image) / 255.0
            image_flatten = image_array.flatten().reshape(1, -1)
            
            st.image(image, caption="Uploaded Image", width=100)
            prediction = st.session_state['model'].predict(image_flatten)[0]
            st.success(f"🔮 Predicted Label: {prediction}")
