import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

# Paths (Ensure correctness)
MODEL_PATH = "artifacts/model.pkl"
TRAIN_DATA_PATH = "artifacts/train_transformed.npy"
TEST_DATA_PATH = "artifacts/test_transformed.npy"

# App Title
st.set_page_config(page_title="TrendSeeker - AI Stock Insights", layout="wide")
st.title("📊 TrendSeeker - AI-Powered Stock Trend Analysis")
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

# Load Data
@st.cache_data
def load_data():
    if not os.path.exists(TEST_DATA_PATH):
        st.error(f"❌ Test data file not found: {TEST_DATA_PATH}")
        st.stop()
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    return train_data, test_data

# Load model & data
model = load_model()
train_arr, test_arr = load_data()

# Prepare Data
X_test = test_arr[:, :-1]  # Features
y_test = test_arr[:, -1]    # Target labels
y_pred = model.predict(X_test)
columns = ["Open", "High", "Low", "Close", "Volume", "EMA_5", "Stoch_RSI_5"]
df = pd.DataFrame(X_test, columns=columns)
df["Actual Trend"] = y_test
df["Predicted Trend"] = y_pred

# Sidebar Navigation
st.sidebar.header("🔍 Explore TrendSeeker")
page = st.sidebar.radio("Navigate", ["📈 Overview", "📊 Predictions", "📉 Performance Metrics"])

if page == "📈 Overview":
    st.subheader("📌 Stock Prices with Predicted Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.7)
    ax.scatter(df.index[df["Predicted Trend"] == 1], df["Close"][df["Predicted Trend"] == 1],
               color='green', label="Uptrend", marker='^')
    ax.scatter(df.index[df["Predicted Trend"] == 0], df["Close"][df["Predicted Trend"] == 0],
               color='red', label="Downtrend", marker='v')
    ax.set_xlabel("Time")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

elif page == "📊 Predictions":
    st.subheader("🔹 Prediction Results")
    trend_filter = st.radio("Filter by Trend", ["All", "Uptrend", "Downtrend"], horizontal=True)
    
    filtered_df = df.copy()
    if trend_filter == "Uptrend":
        filtered_df = df[df["Predicted Trend"] == 1]
    elif trend_filter == "Downtrend":
        filtered_df = df[df["Predicted Trend"] == 0]
    
    st.dataframe(filtered_df)
    
elif page == "📉 Performance Metrics":
    st.subheader("✅ Model Performance")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("🔹 Prediction Accuracy", f"{accuracy:.2%}")
    st.write("### Confusion Matrix")
    cm_data = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    st.dataframe(cm_data.style.background_gradient(cmap="Blues"))
