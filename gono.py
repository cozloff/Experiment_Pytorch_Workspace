import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Display information about the GPU if available
if device.type == "cuda":
    print("GPU available:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
    print("Memory Cached:", torch.cuda.memory_reserved(0) / 1024**3, "GB")
else:
    print("No GPU detected. Please ensure your machine has a CUDA-compatible GPU and the necessary drivers installed.")

# Test a simple tensor operation on the GPU if available
try:
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y
    print("Tensor operation result:", z)
    print("Tensor is on device:", z.device)
except Exception as e:
    print("An error occurred while trying to use the GPU:", e)
