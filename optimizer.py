import torch
import torch.optim as optim

# 1. The variable we want to optimize (Starting at 10)
x = torch.tensor([10.0], requires_grad=True)

#2 setup the optimizer( Adam is faster than SDG)
# lr = learning rate how big the step is

optimizer = optim.Adam([x], lr = 0.1)

print(f"Starting value: {x.item()}")
# 3 the loop(Running 50 'steps' to find the bottom)

for i in range(50):
    optimizer.zero_grad()  # clear old notes

    loss = (x**2)  # the math ( we want x to be 0)

    loss.backward() # calculus

    optimizer.step()  # the actual movement

    if i % 10 == 0:
     print(f"Step {i}: x = {x.item():.4f}, Loss = {loss.item():.4f}")

print(f"Final value of x: {x.item():.4f}")
