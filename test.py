import torch
import numpy as np
from heehee import TemporalTransformerWithCrossPairAttention

# Define symbols and paths for loading test data
symbols = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
test_file_paths = [f"./test/{symbol}_data.pt" for symbol in symbols]

# Load test data into a dictionary for each currency pair
test_data_dict = {}
for symbol, file_path in zip(symbols, test_file_paths):
    test_data = torch.load(file_path)
    print(f"{symbol} test data length: {test_data.shape[0]}")
    test_data_dict[symbol] = test_data

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters (should match the training script)
input_dim = test_data_dict[symbols[0]].shape[1] - 1  # Exclude Symbol_ID
print(f"Input dimension: {input_dim}")
d_model = 64
nhead = 4
num_layers = 2 
dim_feedforward = 128
output_dim = 3  
num_pairs = len(symbols)
input_len = 300  # Same as window_length

try:
    # Instantiate the model with input_dim=6
    model = TemporalTransformerWithCrossPairAttention(
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        output_dim,
        num_pairs,
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load('./trained_classification_model.pth', map_location=device)

    # Adjust the input_layer's weight dimensions in the checkpoint
    # because the checkpoint was saved with input_dim=7
    state_dict = checkpoint['model_state_dict']

    # Load the adjusted state_dict into the model
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode

    # Define the inference function for sequential processing
    def batch_inference(model, test_data_dict, device, input_len=300):
        model.eval()
        predictions = {symbol: [] for symbol in symbols}
        min_length = min(test_data.shape[0] for test_data in test_data_dict.values())
        print(f"min_length: {min_length}, input_len: {input_len}")

        with torch.no_grad():
            # Initialize buffers for each symbol
            buffers = {}
            for idx, symbol in enumerate(symbols):
                test_data = test_data_dict[symbol][:, :-1]  # ExcludeSymbol_ID column
                buffer = test_data[:input_len].to(device)  # Initial buffer for each symbol
                buffers[symbol] = buffer

            # Process each timestep sequentially to simulate real-world data flow
            for i in range(input_len, min_length):
                input_seqs = []
                pair_ids = []
                for idx, symbol in enumerate(symbols):
                    buffer = buffers[symbol]
                    input_seqs.append(buffer.unsqueeze(0))  # Shape: (1, input_len, input_dim)
                    pair_ids.append(idx)
                input_seqs = torch.cat(input_seqs, dim=0)  # Shape: (num_symbols, input_len, input_dim)
                pair_ids = torch.tensor(pair_ids, dtype=torch.long, device=device)

                # Forward pass
                output = model(input_seqs, pair_ids)  # Output shape: (num_symbols, output_dim)
                prediction = torch.argmax(output, dim=1).cpu().numpy()  # Shape: (num_symbols,)

                # Store predictions for each symbol
                for idx, symbol in enumerate(symbols):
                    predictions[symbol].append(prediction[idx])

                # Update buffers with new ticks
                for idx, symbol in enumerate(symbols):
                    test_data = test_data_dict[symbol][:, :-1]  # Exclude Symbol_ID columns
                    new_tick = test_data[i].unsqueeze(0).to(device)  # Shape: (1, input_dim)
                    buffers[symbol] = torch.cat((buffers[symbol][1:], new_tick), dim=0)

        return predictions

    # Run batch inference on all symbols
    predictions = batch_inference(model, test_data_dict, device, input_len=input_len)

    # Write the final evaluation to a text file
    with open("evaluation.txt", "w") as f:
        for symbol in symbols:
            pred_counts = np.bincount(predictions[symbol], minlength=output_dim)
            f.write(f"Predictions for {symbol}:\n")
            for label in range(output_dim):
                f.write(f"  Label {label}: {pred_counts[label]} times\n")
            f.write("\n")

    # Simulate trading strategy
    # Assuming we have test data with actual prices
    # Let's assume that the first column in test_data is the price
    # Implement take profit and stop loss
    # Labels:
    #   0: No change (no open position)
    #   1: Positive - Open long position
    #   2: Negative - Open short position
    #   3: Both - Open long position

    # # Parameters for the simulation
    # initial_capital = 10000.0  # Starting money in USD
    # take_profit = 0.0005       # 0.05%
    # stop_loss = 0.0005         # 0.05%
    # max_hold_time = 5          # 5 minutes (assuming 1-minute intervals)
    # position_size = 1000.0     # Amount to trade per position in USD

    # # Simulate trading and write results to the text file
    # with open("trading_simulation.txt", "w") as f:
    #     account_balance = {symbol: initial_capital for symbol in symbols}
    #     for symbol in symbols:
    #         f.write(f"Trading simulation for {symbol}:\n")
    #         prices = test_data_dict[symbol][:, 0].numpy()  # Assuming price is the first column
    #         preds = predictions[symbol]
    #         for i in range(len(preds)):
    #             label = preds[i]
    #             current_price = prices[i + input_len]  # Adjust for input_len offset
    #             # Open positions based on the label
    #             if label == 1 or label == 3:
    #                 # Open long position
    #                 entry_price = current_price
    #                 exit_price = None
    #                 profit = 0.0
    #                 for j in range(1, max_hold_time + 1):
    #                     if i + j >= len(preds):
    #                         break
    #                     future_price = prices[i + input_len + j]
    #                     price_change = (future_price - entry_price) / entry_price
    #                     if price_change >= take_profit:
    #                         exit_price = future_price
    #                         profit = take_profit * position_size
    #                         break
    #                     elif price_change <= -stop_loss:
    #                         exit_price = future_price
    #                         profit = -stop_loss * position_size
    #                         break
    #                 if exit_price is None:
    #                     # Close the position after max_hold_time
    #                     exit_price = prices[min(i + input_len + max_hold_time, len(prices) - 1)]
    #                     price_change = (exit_price - entry_price) / entry_price
    #                     profit = price_change * position_size
    #                 account_balance[symbol] += profit
    #                 f.write(f"  Long position opened at {entry_price:.5f}, closed at {exit_price:.5f}, profit: {profit:.2f}, balance: {account_balance[symbol]:.2f}\n")
    #             elif label == 2:
    #                 # Open short position
    #                 entry_price = current_price
    #                 exit_price = None
    #                 profit = 0.0
    #                 for j in range(1, max_hold_time + 1):
    #                     if i + j >= len(preds):
    #                         break
    #                     future_price = prices[i + input_len + j]
    #                     price_change = (entry_price - future_price) / entry_price
    #                     if price_change >= take_profit:
    #                         exit_price = future_price
    #                         profit = take_profit * position_size
    #                         break
    #                     elif price_change <= -stop_loss:
    #                         exit_price = future_price
    #                         profit = -stop_loss * position_size
    #                         break
    #                 if exit_price is None:
    #                     # Close the position after max_hold_time
    #                     exit_price = prices[min(i + input_len + max_hold_time, len(prices) - 1)]
    #                     price_change = (entry_price - exit_price) / entry_price
    #                     profit = price_change * position_size
    #                 account_balance[symbol] += profit
    #                 f.write(f"  Short position opened at {entry_price:.5f}, closed at {exit_price:.5f}, profit: {profit:.2f}, balance: {account_balance[symbol]:.2f}\n")
    #             else:
    #                 # No action for label 0
    #                 continue
    #         total_profit = account_balance[symbol] - initial_capital
    #         f.write(f"Final account balance for {symbol}: {account_balance[symbol]:.2f}\n")
    #         f.write(f"Total profit for {symbol}: {total_profit:.2f}\n\n")

except Exception as e:
    print(f"An error occurred: {e}")
