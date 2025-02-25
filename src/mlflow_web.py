import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt

# Kết nối với MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Thay URL của MLflow Server
client = MlflowClient()

st.title("🚀 MLflow Model Logging & Registry")

# Chọn thí nghiệm
experiments = client.search_experiments()
experiment_names = [exp.name for exp in experiments]
selected_experiment = st.selectbox("📊 Chọn thí nghiệm", experiment_names)
experiment_id = next(exp.experiment_id for exp in experiments if exp.name == selected_experiment)

# Hiển thị các mô hình đã log
def list_logged_models():
    runs = client.search_runs(experiment_ids=[experiment_id])
    gmt7 = pytz.timezone("Asia/Bangkok")
    df = pd.DataFrame([{ 
        "Run ID": r.info.run_id, 
        "Run Name": r.data.tags.get("mlflow.runName", "N/A"), 
        "Start Time": pd.to_datetime(r.info.start_time, unit='ms').tz_localize('UTC').tz_convert(gmt7).strftime('%Y-%m-%d %H:%M:%S'), 
        "Accuracy": r.data.metrics.get("accuracy", None),
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi"
    } for r in runs])
    return df

st.subheader("📌 Các mô hình đã log")
models_df = list_logged_models()
st.dataframe(models_df.style.set_properties(**{"background-color": "#f0f2f6", "color": "black"}), use_container_width=True)

# Chọn các Run Name để so sánh
st.subheader("📈 So sánh các mô hình")
available_run_names = models_df["Run Name"].tolist()
selected_run_names = st.multiselect("🔍 Chọn Run Name để so sánh", available_run_names)

if selected_run_names:
    comparison_data = []
    for run_name in selected_run_names:
        run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
        run_id = run_info["Run ID"]
        run = client.get_run(run_id)
        comparison_data.append({
            "Run ID": run_id,
            "Run Name": run.data.tags.get("mlflow.runName", "N/A"),
            "Validation Accuracy": run.data.metrics.get("Validation Accuracy", None),
            "Validation Precision": run.data.metrics.get("Validation Precision", None),
            "Validation Recall": run.data.metrics.get("Validation Recall", None),
            "Validation F1-Score": run.data.metrics.get("Validation F1-Score", None),
            "Test Accuracy": run.data.metrics.get("Test Accuracy", None),
            "Test Precision": run.data.metrics.get("Test Precision", None),
            "Test Recall": run.data.metrics.get("Test Recall", None),
            "Test F1-Score": run.data.metrics.get("Test F1-Score", None),
        })
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.set_properties(**{"background-color": "#e8f5e9", "color": "black"}), use_container_width=True)
    
    # Vẽ biểu đồ so sánh các metric
    st.subheader("📊 Biểu đồ so sánh các metric")
    selected_metric = st.selectbox("📌 Chọn metric để vẽ biểu đồ", ["Validation Accuracy", "Validation Precision", "Validation Recall", "Validation F1-Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"])
    
    if selected_metric:
        fig, ax = plt.subplots()
        ax.bar(comparison_df["Run Name"], comparison_df[selected_metric], color='skyblue')
        ax.set_xlabel("Run Name", fontsize=8)
        ax.set_ylabel(selected_metric, fontsize=8)
        ax.set_title(f"So sánh {selected_metric}", fontsize=8)
        ax.tick_params(axis='x', rotation=0, labelsize=5)
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig)

# Huấn luyện và log một mô hình mới
st.subheader("🛠️ Huấn luyện mô hình mới")
num_trees = st.slider("🌲 Số cây trong Random Forest", 10, 200, 50)
log_model = st.button("🚀 Huấn luyện và log mô hình")

if log_model:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=num_trees, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag("mlflow.runName", f"RF_{num_trees}_trees")
        mlflow.log_param("n_estimators", num_trees)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        st.success(f"✅ Mô hình được log với Run Name: RF_{num_trees}_trees")

# Đăng ký mô hình vào Model Registry
st.subheader("📌 Đăng ký mô hình vào MLflow Registry")
run_name = st.text_input("✏️ Nhập Run Name của mô hình muốn đăng ký")
model_name = st.text_input("🏷️ Nhập tên mô hình")
register_model = st.button("📂 Đăng ký mô hình")

if register_model and run_name and model_name:
    run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
    run_id = run_info["Run ID"]
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    st.success(f"🎉 Mô hình đã được đăng ký với tên: {model_name}")