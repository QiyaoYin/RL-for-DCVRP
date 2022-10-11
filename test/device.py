import torch

print(torch.cuda.is_available())
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)