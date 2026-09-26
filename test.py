import torch
from models.VGG_models import vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg16(num_classes=10).to(device)
model.T = 2

x = torch.rand(2, 3, 32, 32).to(device)
y = model(x)
print("Output shape:", y.shape)
