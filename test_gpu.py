import torch, os, sys
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("Device", i, "name:", torch.cuda.get_device_name(i))
