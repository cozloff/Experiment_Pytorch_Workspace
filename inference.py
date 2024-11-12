import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Define the model class
class TemporalTransformerWithCrossPairAttention(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs):
        super(TemporalTransformerWithCrossPairAttention, self).__init__()
        self.pair_embedding = nn.Embedding(num_pairs, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        self.input_layer = nn.Linear(input_dim, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, pair_id):
        pair_embed = self.pair_embedding(pair_id).unsqueeze(1).expand(-1, src.size(1), -1)
        src = self.input_layer(src) + pair_embed
        transformer_output = self.transformer_encoder(src)
        final_output = transformer_output[:, -1, :]
        output = self.output_layer(final_output)
        return output

def main():
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]
    num_pairs = len(symbols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model checkpoint
    checkpoint = torch.load("./fourth_model_checkpoints/model_epoch_4.pth", map_location=device)
    input_len = checkpoint["config"]["input_len"]

    d_model = 128
    nhead = 4
    num_layers = 4
    dim_feedforward = 256
    output_dim = 3

    test_data_dir = "./test/"
    data_dict = {}
    for symbol in symbols:
        file_path = os.path.join(test_data_dir, f"{symbol}_data.pt")
        data = torch.load(file_path)
        data_dict[symbol] = data

    input_dim = data_dict[symbols[0]].shape[1] - 1

    # Initialize the model with the same input_dim used during training
    model = TemporalTransformerWithCrossPairAttention(
        input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs
    ).to(device)

    # Load the state dict from the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model.pair_embedding.load_state_dict(checkpoint["pair_embeddings"])
    model.eval()

    prediction_counts = {i: 0 for i in range(output_dim)}
    prediction_stats = {symbol: {"0": 0, "1": 0, "2": 0} for symbol in symbols}

    sequences = {}
    indices = {}
    done_symbols = set()

    # Initialize sequences with the first `input_len` ticks for each symbol
    for symbol in symbols:
        data = data_dict[symbol]

        # Ensure features are extracted using the saved input dimension
        features = data[:, :input_dim]

        if features.shape[0] < input_len + 1:
            done_symbols.add(symbol)
            continue

        sequences[symbol] = features[:input_len, :].unsqueeze(0)
        indices[symbol] = input_len

    output_files = {symbol: open(f"{symbol}_predictions.txt", "w") for symbol in symbols}
    for symbol, file in output_files.items():
        file.write("TickIndex\tPrediction\tConfidence\tFeatures\n")

    while len(done_symbols) < len(symbols):
        batch_sequences = []
        batch_pair_ids = []
        symbols_in_batch = []

        for idx, symbol in enumerate(symbols):
            if symbol in done_symbols:
                continue

            seq = sequences[symbol]
            index = indices[symbol]
            data = data_dict[symbol]

            # Extract features with the correct input dimension
            features = data[:, :input_dim]

            if index >= features.shape[0]:
                done_symbols.add(symbol)
                continue

            new_tick = features[index, :].unsqueeze(0).unsqueeze(0)
            seq = torch.cat([seq[:, 1:, :], new_tick], dim=1)
            sequences[symbol] = seq
            indices[symbol] += 1

            batch_sequences.append(seq)
            batch_pair_ids.append(torch.tensor(idx))
            symbols_in_batch.append(symbol)

        if len(batch_sequences) == 0:
            break

        batch_sequences = torch.cat(batch_sequences, dim=0).to(device)
        batch_pair_ids = torch.stack(batch_pair_ids).to(device)

        with torch.no_grad():
            output = model(batch_sequences, batch_pair_ids)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        for i, symbol in enumerate(symbols_in_batch):
            tick_index = indices[symbol] - 1
            pred = predictions[i].item()
            probs = probabilities[i].cpu().numpy()
            confidence = probs[pred]
            features = sequences[symbol][0, -1].cpu().numpy()

            output_files[symbol].write(
                f"{tick_index}\t{pred}\t{confidence:.4f}\t{features.tolist()}\n"
            )
            prediction_stats[symbol][str(pred)] += 1
            prediction_counts[pred] += 1

    for file in output_files.values():
        file.close()

    print("\nFinal prediction statistics:")
    for symbol, stats in prediction_stats.items():
        total = sum(int(count) for count in stats.values())
        if total > 0:
            print(f"\n{symbol}:")
            for pred, count in stats.items():
                print(f"Class {pred}: {count} ({(int(count) / total) * 100:.2f}%)")

if __name__ == "__main__":
    main()
