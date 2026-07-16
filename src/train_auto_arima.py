import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pmdarima as pm
import warnings



def run_optimized_baseline(csv_path):
    print("Loading and Optimizing Dataset")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)


    #Downsampling
    #Convert 5min noisy data into 30min smooth averages
    print("  Downsampling data to 30-minute intervals to isolate the signal")
    df_resampled = df[['p99_latency_ms']].resample('30min').mean()
    
    
    SEASONAL_PERIOD = 48 

    train_data = df_resampled.loc['2026-04-01':'2026-04-23', 'p99_latency_ms']
    test_data = df_resampled.loc['2026-04-24':'2026-04-30', 'p99_latency_ms']

    print(f"  Training points: {len(train_data)} | Testing points: {len(test_data)}")

    
    #Auto-ARIMA
    print("\n Running Auto ARIMA Grid Search to find best params")
    
    auto_model = pm.auto_arima(
        train_data,
        m=SEASONAL_PERIOD,    
        seasonal=True,        
        stepwise=True,       
        trace=True,         
        error_action='ignore',
        suppress_warnings=True,
        n_jobs=-1           
    )

    print(f"\nBest Model Found: {auto_model.summary().tables[0].data[0][1]}")
  
  
    # Forecast and Evaluate
    print("\nforcasitng next 7 days")
    forecast = auto_model.predict(n_periods=len(test_data))
    
    mae = mean_absolute_error(test_data, forecast)
    
    print(f"AUTO ARIMA MAE: {mae:.2f} ms")

    
    # Plot the Results
    plt.figure(figsize=(15, 6))
    

    plt.plot(test_data.index, test_data, label='Actual p99 Latency (30m Avg)', color='purple', alpha=0.7, linewidth=2)
    plt.plot(test_data.index, forecast, label='Auto-ARIMA Forecast', color='orange', linestyle='--', linewidth=2.5)
    
   
    plt.axvspan(pd.to_datetime('2026-04-25'), pd.to_datetime('2026-04-26'), color='red', alpha=0.1, label='Deployment Bug Anomaly')

    plt.title(f'Optimized Statistical Baseline vs Actual Latency (MAE: {mae:.2f} ms)', fontsize=14)
    plt.ylabel('Milliseconds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_optimized_baseline('DataSets/latency_data.csv')