import torch

# 1. Create a tensor and tell PyTorch to 'watch' it
# requires_grad=True means "Track every math operation I do to this"
x = torch.tensor([2.0, 3.0], requires_grad=True)

# some math
# y = x^2 +5
y = x**2 + 5

#3 calculate the slope 
#this is what AI uses to see if the answer is right
z= y.mean()
z.backward()

# 4. Check the result
# This tells us: "If I change x, how much does z change?"
print(f"gradiesnt of x: {x.grad}")
