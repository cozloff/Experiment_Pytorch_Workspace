import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from glob import glob

# Define the symbols and their file patterns
symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]
file_pattern = "./train/{}_Ticks_*.csv"

# Map each symbol to a unique identifier
symbol_map = {symbol: idx for idx, symbol in enumerate(symbols)}

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


def process_symbol(symbol):
    # Load all CSV files for the symbol
    all_files = sorted(glob(file_pattern.format(symbol)))
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list)

    # Ensure 'Ask' and 'Bid' columns are floats
    df['Ask'] = pd.to_numeric(df['Ask'], errors='coerce')
    df['Bid'] = pd.to_numeric(df['Bid'], errors='coerce')

    # Drop rows with NaN values in 'Ask' or 'Bid'
    df.dropna(subset=['Ask', 'Bid'], inplace=True)
    
    # Apply normalization based on symbol
    normalizations = {
        "USDJPY": 300, "AUDUSD": 1.3, "EURUSD": 2.2,
        "GBPUSD": 2.6, "NZDUSD": 1.1, "USDCAD": 2.8, "USDCHF": 1.7
    }
    factor = normalizations.get(symbol, 1)
    
    # Debugging: Check the normalization factor
    print(f"Normalizing {symbol} with factor {factor}")

    # Apply normalization
    df['Ask'] = (df['Ask'] / factor).round(6)
    df['Bid'] = (df['Bid'] / factor).round(6)
    
    # Clean up the 'Local time' column and convert it to datetime
    df['Local time'] = df['Local time'].str.replace(' GMT', '', regex=False)
    df['time'] = pd.to_datetime(df['Local time'], dayfirst=True, errors='coerce')
    df.set_index('time', inplace=True)
    df.drop(columns=['Local time'], inplace=True)

    # Downsample if necessary
    target_size = 150000
    if len(df) > target_size:
        indices = np.linspace(0, len(df) - 1, target_size, dtype=int)
        df = df.iloc[indices]

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

    import torch

    # Move data to PyTorch tensors on GPU
    ask_prices = torch.tensor(df['Ask'].values, device='cuda')
    bid_prices = torch.tensor(df['Bid'].values, device='cuda')
    times = torch.tensor(pd.to_datetime(df.index).view(int), device='cuda')  # Convert datetime index to int

    labels = torch.zeros(len(df), dtype=torch.int32, device='cuda')  # Default labels set to 0

    five_min_window = torch.tensor(5 * 60 * 1e9, device='cuda')  # 5 minutes in nanoseconds

    # Vectorized approach
    for i in range(len(df)):
        current_time = times[i]
        current_ask = ask_prices[i]
        current_bid = bid_prices[i]

        # Define the future window for the next 5 minutes
        future_mask = (times > current_time) & (times <= current_time + five_min_window)
        if future_mask.sum() == 0:
            continue

        future_asks = ask_prices[future_mask]
        future_bids = bid_prices[future_mask]

        # Calculate percentage changes in a vectorized manner
        upward_changes = ((future_bids - current_ask) / current_ask) * 100
        downward_changes = ((future_asks - current_bid) / current_bid) * 100

        # Check for the first upward spike
        if torch.any(upward_changes >= 0.03):
            labels[i] = 1
            continue

        # Check for the first downward spike
        if torch.any(downward_changes <= -0.03):
            labels[i] = 2

    # Move labels back to CPU and convert to a numpy array
    labels_np = labels.cpu().numpy()
    df['Label'] = labels_np

    # Get the timezone of the DataFrame index
    df_timezone = df.index.tz

    # Remove the first 60 minutes of the data after processing
    first_day = df.index[0].date()
    # Make the timestamp timezone-aware using the same timezone as the DataFrame
    threshold_time = pd.Timestamp(first_day) + pd.Timedelta(minutes=60)
    threshold_time = threshold_time.tz_localize(df_timezone)

    # Filter the DataFrame using the timezone-aware threshold
    df = df[df.index > threshold_time]


    df['Symbol_ID'] = symbol_map[symbol]
    df.reset_index(inplace=True)
    df['AskVolume'] = df['AskVolume'].apply(normalize_volume)
    df['BidVolume'] = df['BidVolume'].apply(normalize_volume)
    df['time'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second
    df['time'] = df['time'].apply(normalize_time)

    # Save DataFrame to text file for verification
    text_filename = f"./train6/{symbol.replace('/', '')}_data.txt"
    with open(text_filename, "w") as file:
        file.write(df.to_string())
    print(f"Saved {symbol} data to {text_filename} for verification. Symbol ID: {symbol_map[symbol]}")

    # Save the tensor
    tensor_data = torch.tensor(df.values, dtype=torch.float32)
    tensor_filename = f"./train6/{symbol.replace('/', '')}_data.pt"
    torch.save(tensor_data, tensor_filename)
    print(f"Saved {symbol} data to {tensor_filename}")

# Process all symbols
for symbol in symbols:
    process_symbol(symbol)
