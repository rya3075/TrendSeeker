import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

# Paths (Make sure they are correct)
MODEL_PATH = "artifacts/model.pkl"
TRAIN_DATA_PATH = "artifacts/train_transformed.npy"
TEST_DATA_PATH = "artifacts/test_transformed.npy"

st.title("📊 Stock Trend Prediction Dashboard")

@st.cache_resource  # Use cache_resource for joblib (since it's not purely data)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data  # Use cache_data for numpy arrays
def load_data():
    if not os.path.exists(TEST_DATA_PATH):
        st.error(f"❌ Test data file not found: {TEST_DATA_PATH}")
        st.stop()
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
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
columns = ["Open", "High", "Low", "Close", "Volume", "EMA_5", "Stoch_RSI_5"]
df = pd.DataFrame(X_test, columns=columns)
df["Actual Trend"] = y_test
df["Predicted Trend"] = y_pred

# Sidebar
st.sidebar.header("📈 Stock Trend Analysis")
st.sidebar.write("🔹 **Prediction Distribution:**")
st.sidebar.write(df["Predicted Trend"].value_counts())

# Visualization
st.subheader("📌 Stock Prices with Predicted Trends")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.7)
ax.scatter(df.index[df["Predicted Trend"] == 1], df["Close"][df["Predicted Trend"] == 1], 
           color='green', label="Uptrend", marker='^')
ax.scatter(df.index[df["Predicted Trend"] == 0], df["Close"][df["Predicted Trend"] == 0], 
           color='red', label="Downtrend", marker='v')
ax.set_xlabel("Time")
ax.set_ylabel("Close Price")
ax.legend()
st.pyplot(fig)

# Prediction Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ **Prediction Accuracy:** {accuracy:.2%}")
