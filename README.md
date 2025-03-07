# 📈 Stock Market Analysis - FTSE100 & S&P500

A **Python-based** stock market analysis project that examines **FTSE100 and S&P500 indices** using **statistical models and financial time series analysis**.

This project includes **risk modeling, co-integration tests, and volatility forecasting using GARCH models**.

---

## 📌 Project Overview

- **Dataset**: FTSE100 & S&P500 historical price data  
- **Programming Language**: Python  
- **Libraries Used**: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `arch`, `scipy`  
- **Key Techniques**:  
  - Time Series Cleaning & Transformation  
  - Return Computation & Statistical Analysis  
  - Unit Root Tests (ADF) for Stationarity  
  - Co-integration Testing for Market Relationships  
  - CAPM & Beta Calculation  
  - ARCH/GARCH Modeling for Volatility Forecasting  
  
---

## 🚀 Features

✅ **Data Preprocessing** – Cleans and structures stock market data  
✅ **Return Calculation** – Computes daily log returns for FTSE100 & S&P500  
✅ **Descriptive Statistics** – Analyzes distributional properties of returns  
✅ **Unit Root Testing (ADF Test)** – Determines stationarity of price/returns  
✅ **Co-integration Test** – Evaluates long-term relationships between FTSE100 & S&P500  
✅ **Beta Calculation (CAPM Model)** – Measures systematic risk of assets  
✅ **ARCH/GARCH Volatility Models** – Forecasts financial market volatility  
✅ **Graphical Representation** – Plots histograms, ACF/PACF, and volatility trends  

---

## 🏗️ System Architecture

This project is structured as follows:

📂 **proiect_python.py** – Main script for stock market analysis  
📂 **FTSE100_data.csv** – Historical price data for FTSE100 index  
📂 **SP500_data.csv** – Historical price data for S&P500 index  

---

## 📥 Installation & Setup

### 🔹 **Prerequisites**
Ensure you have the following installed:  

```sh
pip install pandas numpy matplotlib statsmodels arch scipy
```

### 🔹 **Running the Project**
1️⃣ Clone the repository:  
```sh
git clone https://github.com/your-repo/stock-analysis.git
```
2️⃣ Navigate to the project directory:  
```sh
cd stock-analysis
```
3️⃣ Run the analysis:  
```sh
python proiect_python.py
```

---

## 📊 Data Analysis Steps

### **📌 Data Processing**
- Reads historical stock market data from CSV files  
- Cleans missing values and converts `Close` prices to numerical format  
- Computes **daily log returns** for FTSE100 and S&P500  

### **📍 Key Analysis Tasks**
1. **Descriptive Statistics & Normality Tests**  
   - Compute **mean, standard deviation, skewness, and kurtosis**  
   - Apply **Jarque-Bera test** to assess normality  
2. **Beta Calculation & CAPM Model**  
   - Estimate **systematic risk (Beta) using OLS regression**  
   - Compute **Jensen's Alpha** to evaluate portfolio performance  
3. **Time Series Analysis**  
   - Perform **Auto-correlation (ACF) & Partial Auto-correlation (PACF) plots**  
   - Conduct **Augmented Dickey-Fuller (ADF) tests** for stationarity  
4. **Market Co-integration Test**  
   - Conduct **Engle-Granger Co-integration Test** for FTSE100 & S&P500  
5. **ARCH & GARCH Volatility Modeling**  
   - Fit **ARCH(1) model** for short-term volatility effects  
   - Fit **GARCH(1,1) model** for long-term volatility forecasting  

---

## 📊 Visualization & Outputs

📌 **Histogram of Returns** – Visualizes return distributions  
📌 **ACF/PACF Plots** – Examines auto-correlation in returns  
📌 **Co-integration Test Results** – Evaluates FTSE100-S&P500 relationships  
📌 **GARCH Model Conditional Volatility** – Forecasts stock market risk  

---

## 🎯 Future Enhancements

🔮 **Next Steps:**  
- Integrate **Machine Learning for stock price prediction**  
- Expand **multi-asset risk modeling** for portfolio optimization  
- Implement **Monte Carlo simulations for market forecasting**  

---

👥 Contributors:  
- **Ariton Alexandru**  

---
