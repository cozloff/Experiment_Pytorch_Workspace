import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Define the Temporal Transformer model with pair embeddings
class TemporalTransformerWithCrossPairAttention(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs):
        super(TemporalTransformerWithCrossPairAttention, self).__init__()
        self.pair_embedding = nn.Embedding(num_pairs, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1
            ),
            num_layers=num_layers,
        )
        self.input_layer = nn.Linear(input_dim, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, pair_id):
        # Print out pair IDs in the batch
        print(f"Forward Pass: Pair IDs - {pair_id.unique().cpu().numpy()}")
        
        # Print out pair embeddings for each unique ID in the batch
        unique_pair_ids = pair_id.unique().cpu().numpy()
        for pid in unique_pair_ids:
            print(f"Embedding for Pair ID {pid}: {self.pair_embedding(torch.tensor([pid], device=src.device)).detach().cpu().numpy()}")
        
        pair_embed = self.pair_embedding(pair_id).unsqueeze(1).expand(-1, src.size(1), -1)
        src = self.input_layer(src) + pair_embed
        transformer_output = self.transformer_encoder(src)
        final_output = transformer_output[:, -1, :]
        output = self.output_layer(final_output)
        return output


def create_sequences(data, input_len, pair_id):
    inputs, labels, pair_ids = [], [], []
    for i in range(len(data) - input_len):
        input_seq = data[i : i + input_len, :-2]
        label = data[i + input_len, -2].long()
        inputs.append(input_seq)
        labels.append(label)
        pair_ids.append(pair_id)
    # Logging sequence creation details for each pair
    print(f"Created {len(inputs)} sequences for pair ID {pair_id}")
    return torch.stack(inputs), torch.tensor(labels), torch.tensor(pair_ids)

def main():
    symbols = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    file_paths = [f"./train2/{symbol}_data.pt" for symbol in symbols]

    data_dict = {symbol: torch.load(file_path) for symbol, file_path in zip(symbols, file_paths)}
    print("Loaded data for symbols:")
    for symbol, data in data_dict.items():
        print(f"{symbol}: {data.shape}")

    input_dim = data_dict[symbols[0]].shape[1] - 2
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    output_dim = 3
    num_pairs = len(symbols)
    input_len = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformerWithCrossPairAttention(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    # Test embedding dimensions
    print(f"Embedding weights shape: {model.pair_embedding.weight.shape}")

    # Sequence creation
    input_seqs, label_seqs, pair_ids = [], [], []
    for idx, symbol in enumerate(symbols):
        inputs, labels, ids = create_sequences(data_dict[symbol], input_len=input_len, pair_id=idx)
        if inputs.numel() == 0:
            print(f"Warning: No data generated for symbol {symbol} (ID {idx})")
        input_seqs.append(inputs)
        label_seqs.append(labels)
        pair_ids.append(ids)

    input_seqs = torch.cat(input_seqs)
    label_seqs = torch.cat(label_seqs)
    pair_ids = torch.cat(pair_ids)

    print(f"Total sequences: {len(input_seqs)}")
    print(f"Unique Pair IDs in dataset: {set(pair_ids.numpy())}")

    # DataLoader
    dataset = TensorDataset(input_seqs, label_seqs, pair_ids)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    # Training loop with batch inspection
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        all_preds, all_targets = [], []

        for batch_idx, (x_batch, y_batch, id_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            id_batch = id_batch.to(device)




            # Log unique pair IDs in each batch
            print(f"Batch {batch_idx}: Pair IDs - {id_batch.unique().cpu().numpy()}")

            optimizer.zero_grad()
            with autocast():
                output = model(x_batch, id_batch)
                loss = nn.CrossEntropyLoss()(output, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

                        # Log the output distribution for each pair ID
            for pid in id_batch.unique():
                pid_outputs = output[id_batch == pid]
                print(f"Outputs for Pair ID {pid.item()} - Mean: {pid_outputs.mean().item()}, Std Dev: {pid_outputs.std().item()}")

            for i in range(num_pairs):
                grad = model.pair_embedding.weight.grad[i].cpu().numpy()
                print(f"Gradient for Pair ID {i}: Mean {grad.mean()}, Std Dev {grad.std()}")


        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader)}")

        


# Run main
if __name__ == "__main__":
    main()
