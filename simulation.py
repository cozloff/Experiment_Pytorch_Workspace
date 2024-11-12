import os
import pandas as pd
import traceback
from helpers import *

def simulate_trading(symbols, initial_balance=10000):
    data = load_predictions(symbols)

    account_balance = initial_balance
    five_minute_window = 5 / 1440  # 5 minutes in normalized time

    # Initialize position tracking for each symbol
    positions = {symbol: {
        "open": False,
        "entry_price": 0,
        "entry_time": 0,
        "direction": None
    } for symbol in symbols}

    # Global flag to indicate if any position is open
    global_position_open = False

    total_trades = 0
    results = []

    # Process the combined dataset
    for _, row in data.iterrows():
        symbol = row["symbol"]
        tick_index = row["tick_index"]
        prediction = row["prediction"]
        confidence = row["confidence"]
        tick_time = row["time"]
        ask_price = row["ask_price"]
        bid_price = row["bid_price"]

        position = positions[symbol]
        current_price = bid_price if position["direction"] == 'BUY' else ask_price

        # Open a new position if none is currently open for this symbol
        if not global_position_open and not position["open"] and confidence > 0.9 and prediction != 0:
            total_trades += 1
            position["entry_price"] = ask_price if prediction == 1 else bid_price
            position["entry_time"] = tick_time
            position["direction"] = 'BUY' if prediction == 1 else 'SELL'
            position["open"] = True
            global_position_open = True
            print(f"[{symbol}] Opened {position['direction']} position at tick {tick_index}, price: {position['entry_price']}")

        # If a position is open, track its performance
        if position["open"]:
            trade_amount = account_balance

            # Determine the current price based on the position direction
            if position["direction"] == 'BUY':
                current_price = bid_price  # Closing a BUY uses the bid price
                price_change = current_price - position["entry_price"]
            elif position["direction"] == 'SELL':
                current_price = ask_price  # Closing a SELL uses the ask price
                price_change = position["entry_price"] - current_price
            else:
                # If direction is not set, skip calculation (this shouldn't normally happen)
                continue

            # Calculate the percentage change
            price_change_percent = (price_change / position["entry_price"]) * 100


            # Timeout condition with balance adjustment
            if tick_time - position["entry_time"] >= five_minute_window:
                profit_loss = trade_amount * (price_change_percent / 100)
                account_balance += profit_loss
                results.append([
                    symbol, position["direction"], position["entry_price"], current_price, 
                    "TIMEOUT", tick_index, position["entry_time"], tick_time, 
                    price_change, price_change_percent, account_balance
                ])
                position["open"] = False
                global_position_open = False

            # Check take profit and stop loss conditions
            if position["open"]:
                # Take profit condition
                if position["direction"] == 'BUY' and price_change_percent >= 0.05:
                    profit = trade_amount * 0.0005
                    account_balance += profit
                    results.append([
                        symbol, position["direction"], position["entry_price"], current_price, 
                        "TAKE_PROFIT", tick_index, position["entry_time"], tick_time, 
                        price_change, price_change_percent, account_balance
                    ])
                    position["open"] = False
                    global_position_open = False
                elif position["direction"] == 'SELL' and price_change_percent >= 0.05:
                    profit = trade_amount * 0.0005
                    account_balance += profit
                    results.append([
                        symbol, position["direction"], position["entry_price"], current_price, 
                        "TAKE_PROFIT", tick_index, position["entry_time"], tick_time, 
                        price_change, price_change_percent, account_balance
                    ])
                    position["open"] = False
                    global_position_open = False

                # Stop loss condition
                if position["direction"] == 'BUY' and price_change_percent <= -0.05:
                    loss = trade_amount * 0.0005
                    account_balance -= loss
                    results.append([
                        symbol, position["direction"], position["entry_price"], current_price, 
                        "STOP_LOSS", tick_index, position["entry_time"], tick_time, 
                        price_change, price_change_percent, account_balance
                    ])
                    position["open"] = False
                    global_position_open = False
                elif position["direction"] == 'SELL' and price_change_percent <= -0.05:
                    loss = trade_amount * 0.0005
                    account_balance -= loss
                    results.append([
                        symbol, position["direction"], position["entry_price"], current_price, 
                        "STOP_LOSS", tick_index, position["entry_time"], tick_time, 
                        price_change, price_change_percent, account_balance
                    ])
                    position["open"] = False
                    global_position_open = False

    save_trading_results(results, total_trades, account_balance)

def main():
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]
    simulate_trading(symbols)

if __name__ == "__main__":
    main()
