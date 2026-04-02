import torch

#1. Manuel pytorch (directly from a list)
x = torch.tensor([1.0,2.0,3.0])

#2. Random tensor (commonly used to initialize AI weights)
# creates a 3x3 grid or random numbers
y = torch.rand(3, 3)

#3. zeros or ones (used for placeholders)
z = torch.zeros(2, 2)

print("Manual vector:", x)
print("Random matrix:\n", y)
print("zeros matrix:\n", z)

# Element-wise addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# this happens in parallel across your cpu/gpu cores 
result = a + b
print ("Addition result:", result)

print(f"shape of y: {y.shape}")
print(f"Data type of y: {y.dtype}")

first_column = y[:, 0]
# Get all rows (:), but only the first column (0)
print("\nJust the fisrt column:\n", first_column) # slicing
