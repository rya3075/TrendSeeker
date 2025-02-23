import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model and preprocessed data
model_path = "artifacts/model.pkl"
train_data_path = "artifacts/train_transformed.npy"
test_data_path = "artifacts/test_transformed.npy"

st.title("Stock Trend Prediction Dashboard")

@st.cache_data
def load_model():
    return joblib.load(model_path)

@st.cache_data
def load_data():
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    return train_data, test_data

# Load model and data
model = load_model()
train_arr, test_arr = load_data()

# Separate features and target
X_test = test_arr[:, :-1]  # Features
y_test = test_arr[:, -1]    # Target labels

# Make predictions
y_pred = model.predict(X_test)

# Convert to DataFrame for visualization
df = pd.DataFrame(X_test, columns=["Open", "High", "Low", "Close", "Volume", "EMA_5", "Stoch_RSI_5"])
df["Actual Trend"] = y_test
df["Predicted Trend"] = y_pred

# Sidebar
st.sidebar.header("Stock Trend Analysis")
st.sidebar.write("Prediction Distribution:")
st.sidebar.write(df["Predicted Trend"].value_counts())

# Visualization
st.subheader("Stock Prices with Predicted Trends")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.7)
ax.scatter(df.index[df["Predicted Trend"] == 1], df["Close"][df["Predicted Trend"] == 1], color='green', label="Uptrend", marker='^')
ax.scatter(df.index[df["Predicted Trend"] == 0], df["Close"][df["Predicted Trend"] == 0], color='red', label="Downtrend", marker='v')
ax.set_xlabel("Time")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)


st.write("📌 **Prediction Accuracy:**", model.score(X_test, y_test))
