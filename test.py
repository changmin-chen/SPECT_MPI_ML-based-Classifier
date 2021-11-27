import torch

weights = torch.ones(4, requires_grad=True)

output = (weights*3).mean()
output.backward()

print(weights.grad)

weights.grad.zero_()

print(weights.grad)