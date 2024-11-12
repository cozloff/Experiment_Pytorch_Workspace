import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
import json


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
        pair_embed = self.pair_embedding(pair_id).unsqueeze(1).expand(-1, src.size(1), -1)
        src = self.input_layer(src) + pair_embed
        transformer_output = self.transformer_encoder(src)
        final_output = transformer_output[:, -1, :]
        output = self.output_layer(final_output)
        return output

def create_sequences(data, input_len):
    inputs, labels, pair_ids = [], [], []
    for i in range(len(data) - input_len):
        input_seq = data[i : i + input_len, :-2]
        label = data[i + input_len, -2].long()
        pair_id = data[i + input_len, -1].long()
        inputs.append(input_seq)
        labels.append(label)
        pair_ids.append(pair_id)
    return torch.stack(inputs), torch.tensor(labels), torch.tensor(pair_ids)

def embedding_regularization(model):
    pair_embeddings = model.pair_embedding.weight
    mean_embedding = pair_embeddings.mean(dim=0, keepdim=True)
    variance = ((pair_embeddings - mean_embedding) ** 2).mean()
    return variance

def main():
    symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]
    file_paths = [f"./train5/{symbol}_data.pt" for symbol in symbols]

    data_dict = {symbol: torch.load(file_path) for symbol, file_path in zip(symbols, file_paths)}

    print("Loaded data for symbols:")
    for symbol, data in data_dict.items():
        print(symbol, data.shape)

    input_dim = data_dict[symbols[0]].shape[1] - 2
    d_model = 128  
    nhead = 4
    num_layers = 4  
    dim_feedforward = 256 
    output_dim = 3  
    num_pairs = len(symbols)
    input_len = 300

    print("num_pairs: ", num_pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TemporalTransformerWithCrossPairAttention(
        input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs
    ).to(device)
    optimizer = optim.Adam([
        {'params': model.input_layer.parameters()},
        {'params': model.transformer_encoder.parameters()},
        {'params': model.output_layer.parameters()},
        {'params': model.pair_embedding.parameters(), 'lr': 1e-3}  # Higher LR for embeddings
    ], lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    # Initialize counters
    pair_id_counter = {i: 0 for i in range(num_pairs)}  # Tracks appearance of each pair ID in batches
    prediction_counter = {i: 0 for i in range(num_pairs)}  # Tracks predictions made per pair

    input_seqs_list, label_seqs_list, pair_ids_list = [], [], []
    sample_counts = {}
    for symbol in symbols:
        inputs, labels, ids = create_sequences(
            data_dict[symbol], input_len=input_len
        )
        input_seqs_list.append(inputs)
        label_seqs_list.append(labels)
        pair_ids_list.append(ids)

        # Count samples per pair
        sample_counts[symbol] = len(labels)

    print("Sample counts per pair:")
    for symbol, count in sample_counts.items():
        print(f"{symbol}: {count}")

    input_seqs = torch.cat(input_seqs_list)
    label_seqs = torch.cat(label_seqs_list)
    pair_ids = torch.cat(pair_ids_list)

    print(f"Total sequences: {len(input_seqs)}")
    print(f"Total labels: {len(label_seqs)}")
    print(f"Total pairs: {len(pair_ids)}")
    unique_ids = set(pair_ids.numpy())
    print(f"Unique Pair IDs in dataset: {unique_ids}")

    # Recalculate class weights using sklearn
    labels_np = label_seqs.cpu().numpy()
    class_weights = torch.tensor([2, 3, 3], dtype=torch.float).to(device)
    print(f"Computed class weights: {class_weights}")

    # Use Cross-Entropy Loss with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Create weights for sampling
    weights = []
    for idx in range(num_pairs):
        num_samples = (pair_ids == idx).sum().item()
        weight = 1.0 / num_samples
        weights.extend([weight] * num_samples)
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(weights, len(weights))

    dataset = TensorDataset(input_seqs, label_seqs, pair_ids)
    dataloader = DataLoader(dataset, batch_size=512, sampler=sampler, pin_memory=True)

    save_dir = "./fifth_model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    model.train()
    num_epochs = 5
    reg_weight = 0.01  # Regularization weight
    for epoch in range(num_epochs):
        epoch_loss = 0
        all_preds = []
        all_targets = []
        all_pair_ids = []

        for x_batch, y_batch, id_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            id_batch = id_batch.to(device)

            # Update pair ID counters within the batch
            unique_ids_in_batch = id_batch.unique().cpu().numpy()
            for pair_id in unique_ids_in_batch:
                pair_id_counter[pair_id] += (id_batch == pair_id).sum().item()  # Count occurrences

            optimizer.zero_grad()
            with autocast():
                output = model(x_batch, id_batch)
                loss = loss_fn(output, y_batch)
                # Add embedding regularization
                reg_loss = reg_weight * embedding_regularization(model)
                total_loss = loss + reg_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            all_pair_ids.append(id_batch.cpu())

            # Update prediction counts per pair ID
            for pair_id in unique_ids_in_batch:
                prediction_counter[pair_id] += (preds[id_batch == pair_id].numel())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_pair_ids = torch.cat(all_pair_ids)

        # Calculate metrics
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        classification_report_str = classification_report(all_targets, all_preds, digits=4, zero_division=0)

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader)}")
        print("Overall Classification Report:")
        print(classification_report(all_targets.numpy(), all_preds.numpy(), digits=4, zero_division=0))

        # Save model checkpoint for each epoch
        checkpoint_filename = f"{save_dir}/model_epoch_{epoch + 1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pair_embeddings": model.pair_embedding.state_dict(),
            "loss": epoch_loss / len(dataloader),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "config": {"input_len": input_len}
        }, checkpoint_filename)
        print(f"Model saved to {checkpoint_filename}")

        # Generate the classification report as a dictionary
        classification_report_dict = classification_report(
            all_targets.numpy(), all_preds.numpy(), digits=4, zero_division=0, output_dict=True
        )

        # Save the classification report to a separate JSON file
        report_filename = f"{save_dir}/classification_report_epoch_{epoch + 1}.json"
        with open(report_filename, "w") as f:
            json.dump(classification_report_dict, f, indent=4)
        print(f"Classification report saved to {report_filename}")


        binary_targets = (all_targets.numpy() != 0).astype(int)
        binary_preds = (all_preds.numpy() != 0).astype(int)

        precision = precision_score(binary_targets, binary_preds, zero_division=0)
        recall = recall_score(binary_targets, binary_preds, zero_division=0)
        f1 = f1_score(binary_targets, binary_preds, zero_division=0)

        print(f"Spike Detection Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Per-pair performance
        for idx, symbol in enumerate(symbols):
            pair_indices = (all_pair_ids == idx).numpy()
            symbol_preds = all_preds.numpy()[pair_indices]
            symbol_targets = all_targets.numpy()[pair_indices]

            if len(symbol_targets) == 0:
                print(f"No data for {symbol} (Pair ID {idx})")
                continue

            report = classification_report(symbol_targets, symbol_preds, digits=4, zero_division=0)
            print(f"Classification Report for {symbol} (Pair ID {idx}):\n{report}\n")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pair_embeddings": model.pair_embedding.state_dict(),
            "config": {"input_len": input_len},
        },
        "./trained_classification_model.pth",
    )
    print("Model saved to './trained_classification_model.pth'")

    # Log final counts to a file
    with open("pair_id_usage_counts.txt", "w") as f:
        f.write("Pair ID Usage and Prediction Counts:\n")
        for idx, symbol in enumerate(symbols):
            f.write(f"{symbol} (ID {idx}):\n")
            f.write(f"  Samples used: {pair_id_counter[idx]} occurrences\n")
            f.write(f"  Predictions made: {prediction_counter[idx]} times\n\n")
    print("Saved pair ID usage counts to 'pair_id_usage_counts.txt'")

if __name__ == "__main__":
    main()
