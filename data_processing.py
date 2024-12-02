import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

# Load data
file_path = 'DataBahanPokok.csv'
data = pd.read_csv(file_path)

# Ganti nama bulan dari Bahasa Indonesia ke Bahasa Inggris
month_mapping = {
    "Januari": "January",
    "Februari": "February",
    "Maret": "March",
    "April": "April",
    "Mei": "May",
    "Juni": "June",
    "Juli": "July",
    "Agustus": "August",
    "September": "September",
    "Oktober": "October",
    "November": "November",
    "Desember": "December"
}

# Ganti nama bulan ke Bahasa Inggris
data['Bulan'] = data['Bulan'].map(month_mapping)

# Convert 'Bulan' dan 'Tahun' ke datetime
data['Date'] = pd.to_datetime(data['Tahun'].astype(str) + '-' + data['Bulan'], format='%Y-%B')
data.set_index('Date', inplace=True)

# Create SQLite database and save data
conn = sqlite3.connect('data_warehouse.db')
data.to_sql('bahan_pokok', conn, if_exists='replace', index=True)

# Replikasi data (copy data to another table)
conn.execute('CREATE TABLE IF NOT EXISTS bahan_pokok_copy AS SELECT * FROM bahan_pokok')
conn.commit()

# Filter data
items_of_interest = ['Daging Ayam Kampung', 'Kacang Kedelai Lokal', 'Telur Ayam Kampung', 'Garam Beryodium Halus']
data_filtered = data[data['Nama_Bahan'].isin(items_of_interest)]
data_filtered.sort_index(inplace=True)

def fit_sarima_model(data, item, order=(1,1,1), seasonal_order=(1,1,1,12)):
    item_data = data[data['Nama_Bahan'] == item]['Harga_Rata_Rata'].dropna()

    # Split data untuk training dan test sets data
    train_size = int(len(item_data) * 0.8)
    train, test = item_data[:train_size], item_data[train_size:]

    # Fit SARIMA model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Prediksi dari test set
    test_predictions = model_fit.forecast(steps=len(test))

    # Melakukan Evaluasi pada Model
    error = mean_squared_error(test, test_predictions)
    rmse = np.sqrt(error)

    # Fit untuk model lagi pada seluruh dataset untuk prediksi dataset selanjutnya
    full_model = SARIMAX(item_data, order=order, seasonal_order=seasonal_order)
    full_model_fit = full_model.fit(disp=False)

    return full_model_fit, rmse

models = {}
rmses = {}

# Fit models dan input kedalam dictionary
for item in items_of_interest:
    model_fit, rmse = fit_sarima_model(data_filtered, item)
    models[item] = model_fit
    rmses[item] = rmse

# Prediksi harga pada (June 2024 - December 2026)
future_predictions = {}
start_date = '2024-06-01'
end_date = '2026-12-01'
steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 30 + 1  # Kemungkinan jumlah bulan

for item in items_of_interest:
    model_fit = models[item]
    future_forecast = model_fit.get_forecast(steps=steps)
    future_predictions[item] = future_forecast.predicted_mean

    # Kombinasi dari data yang sudah ada dengan prediksi dari model
    full_data = pd.concat([data_filtered[data_filtered['Nama_Bahan'] == item]['Harga_Rata_Rata'], future_predictions[item]])

    # Kombinasi dari data yang sudah ada
    plt.figure(figsize=(10, 6))
    plt.plot(full_data.index, full_data, label='Historical and Forecasted')
    plt.axvline(x=data_filtered.index[-1], color='r', linestyle='--')  # Garis dimana prediksi dimulai
    plt.title(f'{item} Price Forecast (June 2024 - December 2026)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.show()

    print(f'RMSE for {item}: {rmses[item]}')

# Menampilkan hasil prediksi
for item in items_of_interest:
    print(f'Future Predictions for {item} (June 2024 - December 2026):')
    print(future_predictions[item])

# Data Mining: Linear Regression for simple prediction
for item in items_of_interest:
    item_data = data_filtered[data_filtered['Nama_Bahan'] == item]['Harga_Rata_Rata'].dropna().reset_index()
    item_data['Month'] = item_data['Date'].dt.month
    item_data['Year'] = item_data['Date'].dt.year
    X = item_data[['Year', 'Month']]
    y = item_data['Harga_Rata_Rata']

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.plot(item_data['Date'], y, label='Actual')
    plt.plot(item_data['Date'], predictions, label='Predicted')
    plt.title(f'{item} Price Prediction using Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.show()