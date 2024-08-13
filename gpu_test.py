import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Returns the index of the current GPU device
print(torch.cuda.get_device_name(0))  # Returns the name of the GPU device