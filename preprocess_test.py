import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Define the files and symbols
files = [
    { "filename": "./test/AUDUSD_Ticks_30.10.2024-30.10.2024.csv", "symbol": "AUD/USD" },
    { "filename": "./test/EURUSD_Ticks_30.10.2024-30.10.2024.csv", "symbol": "EUR/USD" },
    { "filename": "./test/GBPUSD_Ticks_30.10.2024-30.10.2024.csv", "symbol": "GBP/USD" },
    { "filename": "./test/NZDUSD_Ticks_30.10.2024-30.10.2024.csv", "symbol": "NZD/USD" },
    { "filename": "./test/USDCAD_Ticks_30.10.2024-30.10.2024.csv", "symbol": "USD/CAD" },
    { "filename": "./test/USDCHF_Ticks_30.10.2024-30.10.2024.csv", "symbol": "USD/CHF" },
    { "filename": "./test/USDJPY_Ticks_30.10.2024-30.10.2024.csv", "symbol": "USD/JPY" },
]

# Map each symbol to a unique identifier
symbol_map = {symbol["symbol"]: idx for idx, symbol in enumerate(files)}  

def normalize_volume(value):
    """Normalize volume to a range of 0 to 1 based on a maximum volume."""
    return round(value / 10000000.0, 6)

def normalize_time(timestamp):
    """Normalize time based on a predefined start and end timestamp."""
    return timestamp / 8640.0

def process_file(file_info):
    filename = file_info["filename"]
    symbol = file_info["symbol"]
    symbol_id = symbol_map[symbol]  # Get the unique identifier for the symbol
    
    # Read the data
    df = pd.read_csv(filename)
    
    # Clean up the 'Local time' column and convert it to datetime
    df['Local time'] = df['Local time'].str.replace(' GMT', '', regex=False)
    df['time'] = pd.to_datetime(df['Local time'], dayfirst=True, errors='coerce')
    df.set_index('time', inplace=True)

    df.drop(columns=['Local time'], inplace=True)
    
    # Step 1: Calculate the 1-minute ROC for each tick
    ask_roc = []
    bid_roc = []
    window_start = 0
    times = df.index
    for i in range(len(df)):
        current_time = times[i]
        target_time = current_time - pd.Timedelta(minutes=1)
        
        # Advance the window_start pointer until it's as close as possible to target_time
        while window_start < i and times[window_start] < target_time:
            window_start += 1
        
        # Calculate ROC based on the closest time within the sliding window
        if window_start > 0:
            previous_row = df.iloc[window_start - 1]
            current_row = df.iloc[i]
            
            # Calculate the ROC for Ask and Bid prices
            ask_roc_value = ((current_row['Ask'] - previous_row['Ask']) / previous_row['Ask']) * 100
            bid_roc_value = ((current_row['Bid'] - previous_row['Bid']) / previous_row['Bid']) * 100
        else:
            ask_roc_value = None
            bid_roc_value = None
        
        ask_roc.append(ask_roc_value)
        bid_roc.append(bid_roc_value)
    
    # Assign ROC values to DataFrame
    df['Ask_ROC_1'] = ask_roc
    df['Bid_ROC_1'] = bid_roc

    # Step 3: Remove the first 30 minutes after ROC and labeling
    start_time = df.index[0]
    threshold_time = start_time + pd.Timedelta(minutes=30)
    df = df[df.index > threshold_time].copy()

    # Step 4: Add symbol ID and time as columns
    df['Symbol_ID'] = symbol_id  # Add the unique identifier for the currency pair
    df.reset_index(inplace=True)  # Reset index to make 'time' a column

    # Step 5: Normalize columns
    df['AskVolume'] = df['AskVolume'].apply(lambda x: normalize_volume(x))
    df['BidVolume'] = df['BidVolume'].apply(lambda x: normalize_volume(x))

    df['time'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second + df['time'].dt.microsecond / 1_000_000
    df['time'] = (df['time'].astype('int64')).apply(lambda x: normalize_time(x))

    # Save DataFrame to text file for verification
    text_filename = f"./test/{symbol.replace('/', '')}_data.txt"
    with open(text_filename, "w") as file:
        file.write(df.to_string())
    print(f"Saved {symbol} data to {text_filename} for verification. Symbol ID: {symbol_id}")
    
    # Convert to tensor after verification, ensuring Label is included in the tensor data
    tensor_data = torch.tensor(df.values, dtype=torch.float32)
    
    # Save the tensor
    tensor_filename = f"./test/{symbol.replace('/', '')}_data.pt"
    torch.save(tensor_data, tensor_filename)
    print(f"Saved {symbol} data excluding first 30 minutes to {tensor_filename}")

# Process each file and confirm each symbol ID is unique
for file_info in files:
    process_file(file_info)
