import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
st.set_page_config(page_title="Stock Price Prediction",page_icon="📈",layout="wide")
menu = st.sidebar.radio("📌 Navigation",["🏠 Home","📂 Dataset","📊 Visualization","🤖 Train Model","🔮 Prediction","📈 Results","ℹ️ About"])
if menu == "🏠 Home":
    st.title("📈 Stock Price Prediction System")
    st.markdown("""
    ## Welcome
    This project predicts the closing price of a stock using Machine Learning.
    ### Objectives
    - Analyze historical stock market data
    - Train a Machine Learning model
    - Predict stock closing prices
    - Compare Actual and Predicted Prices
    ### Selected Stock
    Apple (AAPL)
    ### Technologies Used
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    """)
    st.write("""This application analyzes historical stock market data, trains a machine learning model, and predicts the closing price based on user-provided stock information.""")
elif menu == "📂 Dataset":
    st.title("📂 Dataset")
    df = pd.read_csv("stock_data.csv")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    st.subheader("Dataset Information")
    st.write(df.dtypes)
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Summary Statistics")
    st.write(df.describe())
elif menu == "📊 Visualization":
    st.title("📊 Data Visualization")
    # Load Dataset
    df = pd.read_csv("stock_data.csv")
    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])
    # ----------------------------
    # Closing Price Graph
    # ----------------------------
    st.subheader("📈 Closing Price Over Time")
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df["Date"], df["Close"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Price")
    ax1.set_title("Closing Price Over Time")
    st.pyplot(fig1)
    # ----------------------------
    # Trading Volume Graph
    # ----------------------------
    st.subheader("📊 Trading Volume Over Time")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(df["Date"], df["Volume"])
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.set_title("Trading Volume")
    st.pyplot(fig2)
    # ----------------------------
    # Moving Averages
    # ----------------------------
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    st.subheader("📉 20-Day & 50-Day Moving Average")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(df["Date"], df["Close"], label="Close Price")
    ax3.plot(df["Date"], df["MA20"], label="20-Day MA")
    ax3.plot(df["Date"], df["MA50"], label="50-Day MA")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.set_title("Moving Average Analysis")
    ax3.legend()
    st.pyplot(fig3)
elif menu == "🤖 Train Model":
    st.title("🤖 Train Model")
    # Load Dataset
    df = pd.read_csv("stock_data.csv")
    # Data Preprocessing
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    # Features and Target
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    st.write("Training Data:", X_train.shape)
    st.write("Testing Data:", X_test.shape)
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save Model
    joblib.dump(model, "model.pkl")
    st.success("✅ Model trained successfully!")
    st.write("Model has been saved as **model.pkl**")
elif menu == "🔮 Prediction":
    st.title("🔮 Stock Price Prediction")
    if not os.path.exists("model.pkl"):
        st.warning("Please train the model first by visiting the 🤖 Train Model page.")
    else:
        model = joblib.load("model.pkl")
        st.subheader("Enter Stock Details")
        open_price = st.number_input("Open Price", min_value=0.0, format="%.4f")
        high_price = st.number_input("High Price", min_value=0.0, format="%.4f")
        low_price = st.number_input("Low Price", min_value=0.0, format="%.4f")
        volume = st.number_input("Volume", min_value=0.0)
        if st.button("Predict Closing Price"):
            input_data = [[open_price, high_price, low_price, volume]]
            prediction = model.predict(input_data)
            st.success(f"Predicted Closing Price: ${prediction[0]:.2f}")
elif menu == "📈 Results":
    st.title("📈 Results")
    # Load Dataset
    df = pd.read_csv("stock_data.csv")
    # Data Preprocessing
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    # Features and Target
    X = df[["Open", "High", "Low", "Volume"]]
    y = df["Close"]
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Prediction
    prediction = model.predict(X_test)
    # Evaluation
    r2 = r2_score(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    mae = mean_absolute_error(y_test, prediction)
    st.subheader("📊 Model Evaluation")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("MAE", f"{mae:.4f}")
    st.subheader("📈 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.values,label="Actual Price")
    ax.plot(prediction,label="Predicted Price")
    ax.set_xlabel("Test Samples")
    ax.set_ylabel("Closing Price")
    ax.set_title("Actual vs Predicted Stock Prices")
    ax.legend()
    st.pyplot(fig)
    st.info("""**Interpretation of Results**
    • **R² Score:** A value closer to 1 indicates that the model explains most of the variation in the stock's closing price.
    • **RMSE:** Lower RMSE values indicate smaller prediction errors.
    • **MAE:** Lower MAE values indicate that the predicted closing prices are closer to the actual closing prices.""")
elif menu == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""## 📌 Project Overview    
    This project predicts the closing price of a stock using Machine Learning. Historical stock market data is analyzed to train a Linear Regression model, which estimates the closing price based on stock features.
    
    ## 📂 Dataset
    Source: Kaggle
                
    The dataset contains:
    - Date
    - Open Price
    - High Price
    - Low Price
    - Close Price
    - Adjusted Close Price
    - Volume
    
    ## 🤖 Machine Learning Algorithm
    Linear Regression
    Linear Regression predicts the stock's closing price using:
    - Open Price
    - High Price
    - Low Price
    - Volume
    
    ## 🛠 Technologies Used
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Scikit-learn
    - Streamlit
    - Joblib
    
    ## 🚀 Future Enhancements
    - Support multiple company stocks
    - Use Random Forest Regression
    - Use LSTM Deep Learning model
    - Display live stock prices
    - Improve prediction accuracy
    - Add interactive charts
    
    ## 🎯 Conclusion
    This project demonstrates how Machine Learning can be applied to historical stock market data to estimate stock closing prices. It also showcases data preprocessing, visualization, model training, evaluation, and predictionusing a user-friendly Streamlit interface.""")