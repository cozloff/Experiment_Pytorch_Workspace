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
    return torch.stack(inputs), torch.tensor(labels), torch.tensor(pair_ids)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Tensor of shape [num_classes]
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply CrossEntropyLoss without reduction to calculate loss per class
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        
        # Calculate pt by applying the exponential to the negative cross-entropy loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    symbols = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    file_paths = [f"./train/{symbol}_data.pt" for symbol in symbols]

    data_dict = {symbol: torch.load(file_path) for symbol, file_path in zip(symbols, file_paths)}

    input_dim = data_dict[symbols[0]].shape[1] - 2
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    output_dim = 4
    num_pairs = len(symbols)
    input_len = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TemporalTransformerWithCrossPairAttention(
        input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, num_pairs
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    input_seqs, label_seqs, pair_ids = [], [], []
    for idx, symbol in enumerate(symbols):
        inputs, labels, ids = create_sequences(
            data_dict[symbol], input_len=input_len, pair_id=idx
        )
        input_seqs.append(inputs)
        label_seqs.append(labels)
        pair_ids.append(ids)

    input_seqs = torch.cat(input_seqs)
    label_seqs = torch.cat(label_seqs)
    pair_ids = torch.cat(pair_ids)

    # Set custom class weights with a higher weight for label 0
    class_weights = torch.tensor([4.5, 1.0, 1.0, 1.0], dtype=torch.float).to(device)
    print(f"Using custom class weights: {class_weights}")

    # Initialize FocalLoss with the custom class weights
    loss_fn = FocalLoss(gamma=2, alpha=class_weights)


    dataset = TensorDataset(input_seqs, label_seqs, pair_ids)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)

    # Training loop with modified thresholding for predictions
    model.train()
    for epoch in range(10):  # Adjust the number of epochs as needed
        epoch_loss = 0
        all_preds = []
        all_targets = []

        for x_batch, y_batch, id_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            id_batch = id_batch.to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(x_batch, id_batch)
                loss = loss_fn(output, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Prediction with adjusted threshold
            probabilities = torch.softmax(output, dim=1)
            threshold = 0.5

            # Start with label 0 as default prediction
            preds = torch.zeros(output.size(0), dtype=torch.long, device=device)
            for label in [1, 2, 3]:  # Apply threshold to spike labels only
                preds_label = (probabilities[:, label] >= threshold).long() * label
                preds = torch.where(preds_label == label, preds_label, preds)


            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_preds_np = all_preds.numpy()
        all_targets_np = all_targets.numpy()

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(dataloader)}")
        print("Classification Report:")
        print(classification_report(all_targets_np, all_preds_np, digits=4, zero_division=0))

        # Spike Detection Metrics
        binary_targets = (all_targets_np != 0).astype(int)
        binary_preds = (all_preds_np != 0).astype(int)

        precision = precision_score(binary_targets, binary_preds, zero_division=0)
        recall = recall_score(binary_targets, binary_preds, zero_division=0)
        f1 = f1_score(binary_targets, binary_preds, zero_division=0)

        print(f"Spike Detection Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pair_embeddings": model.pair_embedding.state_dict(),
            "config": {"input_len": input_len},
        },
        "./trained_classification_model.pth",
    )
    print("Model saved to './trained_classification_model.pth'")

# Only execute main if this script is run directly
if __name__ == "__main__":
    main()
