import torch
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define the symbols you're working with
symbols = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF"]
file_paths = [f"./train5/{symbol}_data.pt" for symbol in symbols]

# Load the data from .pt files and extract labels
labels_list = []

for file_path in file_paths:
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        data = torch.load(file_path)
        labels = data[:, -2].long()  # Assuming the second-to-last column is the label
        labels_list.append(labels)
    else:
        print(f"File {file_path} not found. Skipping...")

# Concatenate all labels into a single tensor
if len(labels_list) > 0:
    all_labels = torch.cat(labels_list)
else:
    raise ValueError("No labels found in the loaded data.")

# Convert PyTorch tensor to numpy array for sklearn
labels_np = all_labels.cpu().numpy()

# Compute class weights using sklearn
unique_classes = np.unique(labels_np)
class_weights_sklearn = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=labels_np
)

# Convert the class weights to a PyTorch tensor
class_weights = torch.tensor(class_weights_sklearn, dtype=torch.float32)
print(f"Computed class weights using sklearn: {class_weights}")
