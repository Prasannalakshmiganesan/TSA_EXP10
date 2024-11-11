## Developed By: Prasannalakshmi G
## Reg No: 212222240075
## Date: 

# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL


### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions

   
### PROGRAM:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'Goodreads_books.csv'
data = pd.read_csv(file_path, index_col='publication_date', parse_dates=True)

# Focus on 'ratings_count' for time series analysis
data = data['ratings_count'].resample('MS').sum()  # Sum ratings per month

# Plot the data
data.plot(figsize=(10, 5))
plt.title("Ratings Count Over Time")
plt.ylabel("Ratings Count")
plt.xlabel("Date")
plt.show()

# Define the ADF test function for stationarity
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # True if stationary

# Check if the series is stationary
is_stationary = adf_test(data)

# Plot ACF and PACF
plot_acf(data.dropna(), lags=20)
plt.show()

plot_pacf(data.dropna(), lags=20)
plt.show()

# Define ARIMA order parameters
p, d, q = 1, 1, 1       # ARIMA terms
P, D, Q, S = 1, 1, 1, 12 # Seasonal terms with seasonality cycle S=12 (monthly data)

# Fit the SARIMA model
model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, S))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 12 months
forecast_steps = 12
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

# Plot historical data and forecast
plt.plot(data, label="Historical Data")
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("Ratings Count Forecast")
plt.show()

# Use Auto ARIMA to determine the best model
auto_model = auto_arima(data, seasonal=True, m=12)
print(auto_model.summary())

# Train-test split for model evaluation
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Fit the model on the training data
model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, S))
fitted_train_model = model.fit()

# Make predictions on the test data
start = len(train_data)
end = len(data) - 1
predictions = fitted_train_model.predict(start=start, end=end)

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print("RMSE:", rmse)

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/8dd4804f-f3f6-4024-9ea6-4eb296a38062)





### RESULT:
Thus the program run successfully based on the SARIMA model.
