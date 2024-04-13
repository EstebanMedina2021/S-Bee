import torch

"""
This script checks if a GPU (Graphics Processing Unit) is available 
and uses it for tensor computations if possible. Otherwise, it falls back 
to using the CPU.

A tensor is a multi-dimensional data structure similar to a NumPy array 
commonly used in deep learning. This script creates a random tensor and 
moves it to the chosen device (GPU or CPU) for potential performance benefits.
"""

# Check for GPU availability
if torch.cuda.is_available():
    print("Great news! We can leverage the power of a GPU for computations.")
    device = torch.device('cuda')  # Use GPU for tensor operations
else:
    print("No GPU detected. We'll use the CPU instead.")
    device = torch.device('cpu')  # Use CPU for tensor operations

# Create a random tensor with size (2, 3)
tensor = torch.randn(2, 3)

# Move the tensor to the chosen device (GPU or CPU)
tensor = tensor.to(device)

# Print information about the tensor's device
print(f"The tensor is now on device: {tensor.device}")
