import os
import pandas as pd
import traceback

# Normalization factors for each currency pair
normalizations = {
    "USDJPY": 300, "AUDUSD": 1.3, "EURUSD": 2.2,
    "GBPUSD": 2.6, "NZDUSD": 1.1, "USDCHF": 1.7
}

def load_predictions(symbols):
    """Load and de-normalize predictions for all symbols into a single dataset."""
    try:
        combined_data = []
        for symbol in symbols:
            file_path = f"{symbol}_predictions.txt"
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Loading data from: {file_path}")
            with open(file_path, "r") as file:
                file.readline()  # Skip header
                for line in file:
                    parts = line.strip().split("\t")
                    tick_index = int(parts[0])
                    prediction = int(parts[1])
                    confidence = float(parts[2])
                    features = eval(parts[3])
                    normalized_time = features[0]
                    ask_price = features[1]
                    bid_price = features[2]

                    # De-normalize prices using the correct factor
                    factor = normalizations.get(symbol, 1)
                    ask_price *= factor
                    bid_price *= factor

                    combined_data.append({
                        "symbol": symbol,
                        "tick_index": tick_index,
                        "prediction": prediction,
                        "confidence": confidence,
                        "time": normalized_time,
                        "ask_price": ask_price,
                        "bid_price": bid_price
                    })
        
        combined_df = pd.DataFrame(combined_data)
        print(f"Total rows loaded: {len(combined_df)}")
        
        if combined_df.empty:
            print("DataFrame is empty after loading predictions.")
        else:
            print("First few rows of the combined DataFrame:")
            print(combined_df.head())

        combined_df.sort_values(by="time", inplace=True)

        # Save to CSV for inspection
        if not combined_df.empty:
            combined_df.to_csv("combined_predictions.csv", index=False)
            print("Combined dataset saved to combined_predictions.csv")
        
        return combined_df

    except Exception as e:
        print("Error during loading predictions:")
        traceback.print_exc()
        return pd.DataFrame()

def save_trading_results(results, total_trades, account_balance):
    """Save trading results to a CSV file."""
    try:
        with open("trading_results_detailed.csv", "w") as f:
            f.write("Symbol,Direction,Entry_Price,Exit_Price,Reason,Close_Tick,Entry_Time,Close_Time,Price_Change,Price_Change_Percent,Balance\n")
            for result in results:
                f.write(",".join(map(str, result)) + "\n")
        print("Trading results saved to trading_results_detailed.csv")
        print(f"Total Trades: {total_trades}, Final Balance: {account_balance}")
    except Exception as e:
        print("Error saving trading results:", str(e))
