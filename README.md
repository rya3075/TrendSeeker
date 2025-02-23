# Stock Trend Prediction 📈  

## Overview  
Stock Trend Prediction is a machine learning-based application that forecasts stock price movements using historical data. The project leverages machine learning models and data visualization techniques to provide insights into market trends.  

## Features  
- 📊 **Historical Data Analysis**: Utilizes past stock data for trend identification.  
- 🤖 **Machine Learning Model**: Predicts stock trends based on trained algorithms.  
- 📈 **Interactive Visualizations**: Displays stock trends through dynamic charts.  
- 🚀 **Streamlit Deployment**: A user-friendly web interface for easy interaction.  

## Technologies Used  
- **Python 3.12**  
- **Streamlit**  
- **scikit-learn**  
- **pandas**  
- **NumPy**  
- **Matplotlib & Seaborn**  
- **Joblib** (for model persistence)  

## Installation & Setup  
To run this project locally, follow these steps:  

```bash
# Clone the repository
git clone https://github.com/your-username/stock-trend-prediction.git
cd stock-trend-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run TrendSeeker.py
