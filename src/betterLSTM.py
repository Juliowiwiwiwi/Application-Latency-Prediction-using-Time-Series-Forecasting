import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, target, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : (i + lookback)])
        y.append(target[i + lookback + horizon])
    return np.array(X), np.array(y)

def run_optimized_lstm(csv_path):
    print("Loading dataset")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    features = ['request_count', 'cpu_usage', 'db_query_time_ms', 'p50_latency_ms', 'p99_latency_ms']
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_X = scaler_X.fit_transform(df[features])
    scaled_y = scaler_y.fit_transform(df[['p99_latency_ms']])

    #Look back 2 hours 
    LOOKBACK = 24 
    HORIZON = 6   

    X, y = create_sequences(scaled_X, scaled_y, LOOKBACK, HORIZON)
    
    
    #The Test Set Fix
    # 7 days * 24 hours * 12 intervals = 2016

    test_size = 2016 
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    test_dates = df.index[-test_size:]

    print("Building the LSTM Model")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(LOOKBACK, len(features))),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("Training Model")
    history = model.fit(
        X_train, y_train, 
        epochs=100,             
        batch_size=64,          
        validation_split=0.15, 
        callbacks=[early_stop], 
        verbose=1
    )

    print("\nForecasting 30-minutes ahead for the Test Set")
    predictions = model.predict(X_test)

    predictions_ms = scaler_y.inverse_transform(predictions)
    actual_ms = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(actual_ms, predictions_ms)
    
    
    print(f"OPTIMIZED LSTM 30-Min Ahead MAE: {mae:.2f} ms")
    

    #Plot the Results
   
    plt.figure(figsize=(15, 6))
    

    plt.plot(test_dates, actual_ms, label='Actual p99 Latency', color='purple', alpha=0.5, linewidth=2)
    plt.plot(test_dates, predictions_ms, label='Optimized LSTM Forecast', color='blue', linestyle='--', linewidth=2)
    
    # Highlight the Day 25 Anomaly area
    plt.axvspan(pd.to_datetime('2026-04-25'), pd.to_datetime('2026-04-26'), color='red', alpha=0.1, label='Deployment Bug Anomaly')

    plt.title(f'Optimized AI Prediction vs Actual Latency (MAE: {mae:.2f} ms)', fontsize=14)
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_optimized_lstm('DataSets/latency_data.csv')