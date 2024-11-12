import os
import pandas as pd
import torch
from glob import glob

# Define the path to the `train5` directory
train5_path = "./train5"
symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]

def analyze_label_distribution(symbol):
    """Analyze the distribution of labels in the processed dataset."""
    # Load the corresponding text file for the symbol
    text_filename = os.path.join(train5_path, f"{symbol}_data.txt")
    
    if not os.path.exists(text_filename):
        print(f"File not found: {text_filename}")
        return
    
    # Read the data into a DataFrame
    df = pd.read_csv(text_filename, delim_whitespace=True)

    # Check if the 'Label' column exists
    if 'Label' not in df.columns:
        print(f"Label column not found in {text_filename}")
        return

    # Get the distribution of labels
    label_counts = df['Label'].value_counts().sort_index()

    # Print the distribution
    print(f"Label distribution for {symbol}:")
    print(label_counts)
    print("\n")

# Analyze label distribution for each symbol
for symbol in symbols:
    analyze_label_distribution(symbol)
