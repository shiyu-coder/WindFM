import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import WindFM, WindFMTokenizer, WindFMPredictor


def plot_probabilistic_forecast(
        ground_truth,
        pred_df,
):
    pred_samples = pred_df.values
    pred_len = len(pred_df)
    q5 = np.quantile(pred_samples, 0.05, axis=1)
    q25 = np.quantile(pred_samples, 0.25, axis=1)
    q50_median = np.quantile(pred_samples, 0.50, axis=1)
    q75 = np.quantile(pred_samples, 0.75, axis=1)
    q95 = np.quantile(pred_samples, 0.95, axis=1)

    forecast_idx = np.arange(pred_len)

    c_truth = 'darkblue'
    c_forecast = '#1F77B4'

    plt.figure(figsize=(8, 5))

    plt.fill_between(forecast_idx, q5, q95, color=c_forecast, alpha=0.15, label='90% Confidence Interval', zorder=1)
    plt.fill_between(forecast_idx, q25, q75, color=c_forecast, alpha=0.3, label='50% Confidence Interval', zorder=2)

    plt.plot(forecast_idx, ground_truth, color=c_truth, linestyle='-', linewidth=1.0, label='Ground Truth', zorder=4)
    plt.plot(forecast_idx, q50_median, color=c_forecast, linestyle='-', linewidth=1.0, label='Median Forecast', zorder=5)

    plt.xlabel("Forecast Horizon (Time Steps)")
    plt.ylabel("Power (MW)")

    plt.grid(True, axis='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = WindFMTokenizer.from_pretrained("NeoQuasar/WindFM-Tokenizer")
model = WindFM.from_pretrained("NeoQuasar/WindFM")

# 2. Instantiate Predictor
predictor = WindFMPredictor(model, tokenizer, device="cuda:0", max_context=512, clip=5)

# 3. Prepare Data
df = pd.read_csv("./data/121522.csv")
df['time'] = pd.to_datetime(df['time'], utc=True)

lookback = 240
pred_len = 80

x_df = df.loc[:lookback-1, ['wind_speed', 'wind_direction', 'power', 'density', 'temperature', 'pressure']]
x_timestamp = df.loc[:lookback-1, 'time']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'time']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=1.0,
    sample_count=100,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
y_power = df.loc[lookback:lookback+pred_len-1, 'power'].values

# visualize
plot_probabilistic_forecast(y_power, pred_df)

