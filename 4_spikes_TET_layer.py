# spikes_TET_layer.py

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
import matplotlib.pyplot as plt
from models.VGG_models import *
from models.resnet_models import *
import data_loaders
from functions import TET_loss, seed_all, get_logger
from datetime import datetime

# ----------------------------
# Runtime arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Spike DIT layer-wise inference")

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to trained model checkpoint (.pth)"
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["resnet18", "resnet19", "vgg16"],
    help="Model architecture to use (resnet18, resnet19, vgg16)"
)

[]

parser.add_argument(
    "--T",
    type=int,
    required=True,
    help="Temporal length T used during inference"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for inference"
)

parser.add_argument(
    "--workers",
    type=int,
    default=16,
    help="Number of DataLoader workers"
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run inference on"
)

args = parser.parse_args()

# ----------------------------
# Function to standardize layer names (QCFS-style)
# ----------------------------
def standardize_layer_name(name: str):
    replacements = {
        "layer1": "block1",
        "layer2": "block2",
        "layer3": "block3",
        "layer4": "block4",
        "spike": "act"
    }
    parts = name.split(".")
    for k, v in replacements.items():
        parts = [p.replace(k, v) for p in parts]
    return ".".join(parts)



def test_with_spikes(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    number_of_neurons = []
    spike_sum_over_samples = []
    layer_index = {}
    hooks = []
    module_call_counter = {}
    current_batch_size = 0
    total_samples = 0
    module_to_name = {m: n for n, m in model.named_modules()}

    raw_spike_total = 0   # <── NEW: global accumulator

    def hook_fn(module, inputs, out):
        nonlocal current_batch_size, raw_spike_total

        spk = (out > 0).float()

        # raw spikes across entire layer
        raw_spike_total += spk.sum().item()

        # normalize per-neuron for your existing metric
        if spk.dim() >= 3 and spk.size(0) == current_batch_size:
            spk = spk.transpose(0, 1)

        feature_dims = tuple(range(2, spk.dim()))
        num_neurons = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)

        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons

        cnt = module_call_counter.get(module, 0) + 1
        module_call_counter[module] = cnt
        mod_name = module_to_name[module]
        key = (mod_name, cnt)

        idx = layer_index.get(key)
        if idx is None:
            idx = len(number_of_neurons)
            layer_index[key] = idx
            number_of_neurons.append(num_neurons)
            spike_sum_over_samples.append(spikes_per_sample.sum().item())
        else:
            spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

    # Register hooks
    for m in model.modules():
        if m.__class__.__name__ == "LIFSpike":
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples += current_batch_size
            module_call_counter.clear()

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
    inv_map = {v: k for k, v in layer_index.items()}

    layer_names = []
    for i in range(len(number_of_spikes)):
        mod_name, call_idx = inv_map[i]
        if call_idx > 1:
            layer_names.append(f"{mod_name}:{call_idx}")
        else:
            layer_names.append(mod_name)

    layer_names = [standardize_layer_name(name) for name in layer_names]
    final_acc = 100.0 * correct / total

    # existing metrics
    print("number_of_neurons:", number_of_neurons)
    print("number_of_spikes:", number_of_spikes)
    print("layer_names:", layer_names)
    print(f"Accuracy: {final_acc:.2f}%")
    network_avg_spikes = sum(number_of_spikes) / len(number_of_spikes)
    print("Network average spikes:", network_avg_spikes)

    # NEW METRIC
    avg_raw_spikes_per_image = raw_spike_total / total_samples
    print( "Total Sample: " , total_samples )
    print("Average TOTAL spikes per image (raw count):", avg_raw_spikes_per_image)

    return final_acc, number_of_neurons, number_of_spikes, layer_names


# ----------------------------
# Main Execution
# ----------------------------
# train_dataset, val_dataset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
# ----------------------------
# Data loading
# ----------------------------
train_dataset, val_dataset = data_loaders.build_cifar(
    cutout=True,
    use_cifar10=True,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)

device = 'cuda'
#model = resnet19(num_classes=10)
model_map = {
    "resnet18": resnet18,
    "resnet19": resnet19,
    "vgg16":    vgg16,
}
model = model_map[args.model](num_classes=10)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
#model.T = 8
model.T = args.T
parallel_model = model.to(device)

# Run test
facc, number_of_neurons, number_of_spikes, layer_names = test_with_spikes(parallel_model, test_loader, device)

# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(12, 5))
bars = plt.bar(range(len(number_of_spikes)), number_of_spikes, color='skyblue', edgecolor='black')
plt.xticks(range(len(number_of_spikes)), layer_names, rotation=45, ha="right", fontsize=8)
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Average Spikes per Neuron", fontsize=12)
plt.title(f"resnet18 | cifar10 | T=8 | Acc={facc:.2f}%", fontsize=14)

# Add spike values on top of bars
for i, v in enumerate(number_of_spikes):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("TET_LT_SR_1e3_resnet18_histogram.png", dpi=300)
plt.close()














# ----------------------------
# Test + spike collection
# ----------------------------
# def test_with_spikes(model, test_loader, device):
#     model.eval()
#     correct, total = 0, 0

#     number_of_neurons = []
#     spike_sum_over_samples = []
#     layer_index = {}
#     hooks = []
#     module_call_counter = {}
#     current_batch_size = 0
#     total_samples = 0
#     module_to_name = {m: n for n, m in model.named_modules()}

#     raw_spike_total = 0


#     def hook_fn(module, inputs, out):
#         nonlocal current_batch_size
#         spk = (out > 0).float()
#         raw_spike_total += spk.sum().item()
#         if spk.dim() >= 3 and spk.size(0) == current_batch_size:
#             spk = spk.transpose(0, 1)

#         feature_dims = tuple(range(2, spk.dim()))
#         num_neurons = 1
#         for d in feature_dims:
#             num_neurons *= spk.size(d)

#         spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons
#         cnt = module_call_counter.get(module, 0) + 1
#         module_call_counter[module] = cnt
#         mod_name = module_to_name[module]
#         key = (mod_name, cnt)

#         idx = layer_index.get(key)
#         if idx is None:
#             idx = len(number_of_neurons)
#             layer_index[key] = idx
#             number_of_neurons.append(num_neurons)
#             spike_sum_over_samples.append(spikes_per_sample.sum().item())
#         else:
#             spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

#     # Register hooks
#     for m in model.modules():
#         if m.__class__.__name__ == "LIFSpike":
#             hooks.append(m.register_forward_hook(hook_fn))

#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             current_batch_size = inputs.size(0)
#             total_samples += current_batch_size
#             module_call_counter.clear()

#             outputs = model(inputs)
#             if outputs.dim() == 3:
#                 mean_out = outputs.mean(1)
#             else:
#                 mean_out = outputs
#             _, predicted = mean_out.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     for h in hooks:
#         h.remove()

#     number_of_spikes = [s / total_samples for s in spike_sum_over_samples]
#     inv_map = {v: k for k, v in layer_index.items()}

#     layer_names = []
#     for i in range(len(number_of_spikes)):
#         mod_name, call_idx = inv_map[i]
#         if call_idx > 1:
#             layer_names.append(f"{mod_name}:{call_idx}")
#         else:
#             layer_names.append(mod_name)

#     # Standardize names
#     layer_names = [standardize_layer_name(name) for name in layer_names]
#     final_acc = 100.0 * correct / total

#     # ---- Print output (to keep same as before) ----
#     print("number_of_neurons:", number_of_neurons)
#     print("number_of_spikes:", number_of_spikes)
#     print("layer_names:", layer_names)
#     print(f"Accuracy: {final_acc:.2f}%")
#     network_avg_spikes = sum(number_of_spikes) / len(number_of_spikes)
#     print("Network average spikes:", network_avg_spikes)
#     avg_raw_spikes_per_image = raw_spike_total / total_samples
#     print("Average TOTAL spikes per image (raw count):", avg_raw_spikes_per_image)


#     return final_acc, number_of_neurons, number_of_spikes, layer_names