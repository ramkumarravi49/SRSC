
import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.tensorboard import SummaryWriter
#import pynvml
import matplotlib.pyplot as plt
from models.VGG_models import *
from models.resnet_models import *
import data_loaders
from functions import TET_loss, seed_all, get_logger
from datetime import datetime

def test_with_spikes(model, test_loader, device):


    model.eval()
    correct, total = 0, 0

    number_of_neurons = []
    spike_sum_over_samples = []
    layer_index = {}
    hooks = []

    current_batch_size = 0
    total_samples = 0

    def hook_fn(module, inputs, out):
        nonlocal current_batch_size

        spk = (out > 0).float()  # binary spikes
        if spk.size(0) == current_batch_size:
            spk = spk.transpose(0, 1)  # [T, N, ...] if needed

        # Flatten spatial dims
        feature_dims = tuple(range(2, spk.dim()))
        num_neurons = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)

        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons

        idx = layer_index.get(module)
        if idx is None:
            idx = len(number_of_neurons)
            layer_index[module] = idx
            number_of_neurons.append(num_neurons)
            spike_sum_over_samples.append(spikes_per_sample.sum().item())
        else:
            spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

    # Hook every LIFSpike
    for m in model.modules():
        if m.__class__.__name__ == "LIFSpike":
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples += current_batch_size

            outputs = model(inputs)

            if outputs.dim() == 3:
                mean_out = outputs.mean(1)
            else:
                mean_out = outputs

            _, predicted = mean_out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    for h in hooks:
        h.remove()

    number_of_spikes = [s / total_samples for s in spike_sum_over_samples]

    print("number_of_neurons:", number_of_neurons)
    print("number_of_spikes:", number_of_spikes)

    final_acc = 100.0 * correct / total
    return final_acc, number_of_neurons, number_of_spikes



train_dataset, val_dataset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                               num_workers=16, pin_memory=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                              shuffle=False, num_workers=16, pin_memory=True)

# model = resnet19(num_classes=10)
# device = 'cuda'
# model = torch.load('/data/cs24m037/TET_with_stats/checkpoints_lr_1e2/best_model.pth', weights_only=False).to(device)
# #/data/cs24m037/TET_with_stats/checkpoints_lr_1e2/best_model.pth


device = 'cuda'
# Rebuild the model architecture
model = resnet19(num_classes=10)
# Load checkpoint dict
checkpoint = torch.load('/data/cs24m037/TET_with_stats/checkpoints_lr_1e2_nocutout/best_model.pth', map_location=device)
# Restore weights
model.load_state_dict(checkpoint['model_state_dict'])
# Move to GPU
model = model.to(device)





model.T = 8 
#parallel_model = torch.nn.DataParallel(model)
#parallel_model.to(device)
parallel_model = model.to(device)
def get_all_spike_layers(model):
    spike_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LIFSpike":
            spike_layers.append(name)
    return spike_layers

# Usage
layers = get_all_spike_layers(model)
for i, name in enumerate(layers):
    print(f"[{i}] {name}")
print(f"Total LIFSpike layers: {len(layers)}")
facc,number_of_neurons, number_of_spikes = test_with_spikes(parallel_model, test_loader, device)

print(facc)
plt.figure(figsize=(8, 5))
plt.bar(range(len(number_of_spikes)), number_of_spikes, color='skyblue', edgecolor='black')

plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Average Number of Spikes", fontsize=12)
plt.title("Spike Counts per Layer", fontsize=14)


plt.tight_layout()
plt.savefig("TET_spike_histogram_lr1e2_cutout_T8.png", dpi=300)
plt.close()


#number_of_spikes = [2.78761484375, 1.5282626953125, 2.2151650390625, 2.120008203125, 1.93587578125, 2.31426015625, 2.3278568359375, 1.969434765625, 2.1311421875, 0.784743310546875, 0.137127]
# plt.figure(figsize=(8, 5))
# plt.bar(range(len(number_of_spikes)), number_of_spikes, color='skyblue', edgecolor='black')

# plt.xlabel("Layer Index", fontsize=12)
# plt.ylabel("Average Number of Spikes", fontsize=12)
# plt.title("Spike Counts per Layer", fontsize=14)
# plt.ylim(0, 7)

# plt.tight_layout()
# plt.savefig("Event_spike_histogram.png", dpi=300)
# plt.close()
