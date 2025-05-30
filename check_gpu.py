import torch
print("Using GPU" if torch.cuda.is_available() else "CPU only")
