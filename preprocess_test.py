import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the files and symbols
files = [
    { "filename": "./test/USDJPY_Ticks_01.11.2024-01.11.2024.csv", "symbol": "USD/JPY" },
    { "filename": "./test/AUDUSD_Ticks_01.11.2024-01.11.2024.csv", "symbol": "AUD/USD" },
    { "filename": "./test/EURUSD_Ticks_01.11.2024-01.11.2024.csv", "symbol": "EUR/USD" },
    { "filename": "./test/GBPUSD_Ticks_01.11.2024-01.11.2024.csv", "symbol": "GBP/USD" },
    { "filename": "./test/NZDUSD_Ticks_01.11.2024-01.11.2024.csv", "symbol": "NZD/USD" },
    { "filename": "./test/USDCAD_Ticks_01.11.2024-01.11.2024.csv", "symbol": "USD/CAD" },
    { "filename": "./test/USDCHF_Ticks_01.11.2024-01.11.2024.csv", "symbol": "USD/CHF" },
]

# Map each symbol to a unique identifier
symbol_map = {symbol["symbol"]: idx for idx, symbol in enumerate(files)}

def normalize_volume(value):
    """Normalize volume to a range of 0 to 1 based on a maximum volume."""
    return round(value / 10000000.0, 3)

def normalize_time(timestamp):
    """Normalize time based on a predefined start and end timestamp."""
    return timestamp / 86400.0

def calculate_bollinger_bands(df, window=1000, std_multiplier=2):
    """Calculate the Bollinger Bands and return differences."""
    sma = df['Ask'].rolling(window=window).mean()
    rolling_std = df['Ask'].rolling(window=window).std()

    upper_band = sma + (rolling_std * std_multiplier)
    lower_band = sma - (rolling_std * std_multiplier)

    # Calculate the difference between the current 'Ask' price and the bands
    df['Ask_Bollinger_Center'] = ((df['Ask'] - sma) * 1000).round(6) 
    df['Ask_Bollinger_Upper'] = ((df['Ask'] - upper_band) * 1000).round(6) 
    df['Ask_Bollinger_Lower'] = ((df['Ask'] - lower_band) * 1000).round(6) 

    # Fill NaNs with zeros for the first few rows where SMA and std aren't available
    df.fillna(0, inplace=True)
    return df

def calculate_rsi(df, window=180):
    """Calculate RSI based on a window of ticks."""
    # Calculate the differences between consecutive 'Ask' prices
    delta = df['Ask'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    # Calculate the average gain and average loss
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN values with zero for the initial rows where RSI can't be calculated
    rsi.fillna(0, inplace=True)
    df['RSI'] = (rsi / 100).round(6)
    return df

def calculate_stochastic(df, window=2000):
    """Calculate Stochastic Oscillator %K and %D."""
    # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    df['Lowest_Low'] = df['Bid'].rolling(window=window, min_periods=1).min()
    df['Highest_High'] = df['Ask'].rolling(window=window, min_periods=1).max()
    
    # %K calculation
    df['%K'] = ((df['Ask'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])).round(6)
    
    # %D is the 3-period moving average of %K
    df['%D'] = df['%K'].rolling(window=10, min_periods=1).mean().round(6)
    
    # Drop temporary columns used for calculation
    df.drop(columns=['Lowest_Low', 'Highest_High'], inplace=True)
    
    # Fill NaNs with 0 for the initial rows where the window isn't complete
    df[['%K', '%D']] = df[['%K', '%D']].fillna(0)
    
    return df

def calculate_atr_percentage(df, window=180):
    """Calculate ATR as a percentage based on a window of ticks."""
    # Calculate True Range (TR)
    df['Previous_Close'] = df['Ask'].shift(1)
    df['High-Low'] = df['Ask'] - df['Bid']
    df['High-PrevClose'] = (df['Ask'] - df['Previous_Close']).abs()
    df['Low-PrevClose'] = (df['Bid'] - df['Previous_Close']).abs()

    # True Range is the maximum of the three calculated ranges
    df['True_Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate the Average True Range (ATR) using a rolling window
    df['ATR'] = df['True_Range'].rolling(window=window, min_periods=1).mean()

    # Convert ATR to percentage of the current Ask price
    df['ATR_Percentage'] = ((df['ATR'] / df['Ask']) * 1000).round(6)

    # Drop temporary columns used for calculation
    df.drop(columns=['Previous_Close', 'High-Low', 'High-PrevClose', 'Low-PrevClose', 'True_Range', 'ATR'], inplace=True)

    # Fill NaNs with 0 for the initial rows where the window isn't complete
    df['ATR_Percentage'] = df['ATR_Percentage'].fillna(0)
    
    return df

def calculate_macd(df, short_window=100, long_window=350, signal_window=75):
    """Calculate MACD line, signal line, and MACD histogram."""
    # Calculate the short and long EMAs
    df['EMA_Short'] = df['Ask'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    df['EMA_Long'] = df['Ask'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    # Calculate the MACD line
    df['MACD_Line'] = df['EMA_Short'] - df['EMA_Long']
    
    # Calculate the signal line
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    # Calculate the MACD histogram
    df['MACD_Histogram'] = ((df['MACD_Line'] - df['Signal_Line']) * 10000).round(6)
    
    # Drop temporary columns used for calculation
    df.drop(columns=['EMA_Short', 'EMA_Long', 'MACD_Line', 'Signal_Line'], inplace=True)
    
    # Ensure the MACD histogram is not in scientific notation
    df['MACD_Histogram'] = df['MACD_Histogram'].astype(float)
    
    return df


def process_file(file_info):
    filename = file_info["filename"]
    symbol = file_info["symbol"]
    symbol_id = symbol_map[symbol]
    
    # Read the data
    df = pd.read_csv(filename)
    
    # Clean up the 'Local time' column and convert it to datetime
    df['Local time'] = df['Local time'].str.replace(' GMT', '', regex=False)
    df['time'] = pd.to_datetime(df['Local time'], dayfirst=True, errors='coerce')
    df.set_index('time', inplace=True)
    df.drop(columns=['Local time'], inplace=True)

    # Downsample if necessary
    target_size = 50000
    if len(df) > target_size:
        indices = np.linspace(0, len(df) - 1, target_size, dtype=int)
        df = df.iloc[indices]

    # Apply normalization based on symbol
    normalizations = {
        "USD/JPY": 300, "AUD/USD": 1.3, "EUR/USD": 2.2,
        "GBP/USD": 2.6, "NZD/USD": 1.1, "USD/CAD": 2.8, "USD/CHF": 1.7
    }
    factor = normalizations.get(symbol, 1)
    df['Ask'] = (df['Ask'] / factor).round(6)
    df['Bid'] = (df['Bid'] / factor).round(6)

    # Calculate Bollinger Bands and SMA differences
    df = calculate_bollinger_bands(df)

    # Calculate RSI using a 180-tick window
    df = calculate_rsi(df, window=180)

    # Calculate Stochastic Oscillator %K and %D using a 2000-tick window
    df = calculate_stochastic(df, window=2000)

    # Calculate ATR using a 180-tick window
    df = calculate_atr_percentage(df, window=180)

    # Calculate MACD Histogram using short=100 ticks, long=350 ticks, signal=75 ticks
    df = calculate_macd(df, short_window=100, long_window=350, signal_window=75)

    # Step 1: Calculate the 1-minute ROC for each tick
    ask_roc = []
    bid_roc = []
    window_start = 0
    times = df.index
    for i in range(len(df)):
        current_time = times[i]
        target_time = current_time - pd.Timedelta(minutes=1)
        while window_start < i and times[window_start] < target_time:
            window_start += 1
        if window_start > 0:
            previous_row = df.iloc[window_start - 1]
            current_row = df.iloc[i]
            ask_roc_value = ((current_row['Ask'] - previous_row['Ask']) / previous_row['Ask']) * 100
            bid_roc_value = ((current_row['Bid'] - previous_row['Bid']) / previous_row['Bid']) * 100
        else:
            ask_roc_value = 0
            bid_roc_value = 0
        ask_roc.append(ask_roc_value)
        bid_roc.append(bid_roc_value)

    df['Ask_ROC_1'] = ask_roc
    df['Bid_ROC_1'] = bid_roc

    # Remove the first 30 minutes
    start_time = df.index[0]
    threshold_time = start_time + pd.Timedelta(minutes=15)
    df = df[df.index > threshold_time].copy()

    df['Symbol_ID'] = symbol_id
    df.reset_index(inplace=True)
    df['AskVolume'] = df['AskVolume'].apply(normalize_volume)
    df['BidVolume'] = df['BidVolume'].apply(normalize_volume)
    df['time'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second
    df['time'] = df['time'].apply(normalize_time)

    # Save DataFrame to text file for verification
    text_filename = f"./test/{symbol.replace('/', '')}_data.txt"
    with open(text_filename, "w") as file:
        file.write(df.to_string())
    print(f"Saved {symbol} data to {text_filename} for verification. Symbol ID: {symbol_id}")

    # Save the tensor
    tensor_data = torch.tensor(df.values, dtype=torch.float32)
    tensor_filename = f"./test/{symbol.replace('/', '')}_data.pt"
    torch.save(tensor_data, tensor_filename)
    print(f"Saved {symbol} data to {tensor_filename}")

# Process all files
for file_info in files:
    process_file(file_info)
