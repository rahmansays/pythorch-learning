import torch

#1 Setup the device toggle
# This check is the first thing every PyTorch dev writes
if torch.cuda.is_available():
    my_device = torch.device("cuda")
    print("🚀 GPU detected! We are going fast.")
else:
    my_device = torch.device("cpu")
    print("💻 No GPU found. Using CPU.")
#2 Create a tensor on the Cpu (default)
x = torch.tensor([1.0, 2.0, 3.0])
print(f"current location: {x.device}")

#3 move to the gpu
x_gpu = x.to(my_device)
print(f"New location: {x_gpu.device}")

