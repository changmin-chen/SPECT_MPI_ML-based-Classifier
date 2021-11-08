import torch

print('Is GPU Available?: ', torch.cuda.is_available())
print('The index of currently selected device: ',torch.cuda.current_device())
print('Number of GPU you have: ',torch.cuda.device_count())
print('Name of your GPU: ',torch.cuda.get_device_name(0))

# run tensor in CPU
x = torch.tensor([1,2,3,4])
print(x.is_cuda)

# throw tensor to the selected GPU ('cuda0' = first visible GPU)
device = torch.device('cuda:0')
x = torch.tensor([1,2,3,4]).to(device)
print(x.is_cuda)

# throw tensor to the first visible GPU
y = torch.tensor([1,2,3,4]).cuda()
print(y.is_cuda)

# with
with torch.cuda.device(0):
    a = torch.tensor([1,2,3,4]).cuda()
    b = torch.tensor([1,2,3,4]).cuda()
    c = a + b
print(c.cpu())