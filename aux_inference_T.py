# aux_inference_T.py
# Based on spikes_TET_layer.py
# Changes from original:
#   1. Added --use_cifar10 arg
#   2. num_classes derived from use_cifar10 (was hardcoded 10)
#   3. Switched to data_loaders_qcfs (was data_loaders)
#   4. FIX: clear forward_spike_traces before each forward pass

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.VGG_models import *
from models.resnet_models import *
import data_loaders_qcfs as data_loaders
from functions import seed_all

parser = argparse.ArgumentParser(description="Spike layer-wise inference")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet19", "vgg16"])
parser.add_argument("--T", type=int, required=True)
parser.add_argument("--use_cifar10", default=True, type=lambda x: x.lower() != 'false')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--run_name", type=str, default="inference")
args = parser.parse_args()

device = 'cuda'


def standardize_layer_name(name: str):
    replacements = {
        "layer1": "block1", "layer2": "block2",
        "layer3": "block3", "layer4": "block4",
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
    raw_spike_total = 0

    def hook_fn(module, inputs, out):
        nonlocal current_batch_size, raw_spike_total
        spk = (out > 0).float()
        raw_spike_total += spk.sum().item()
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

    for m in model.modules():
        if m.__class__.__name__ == "LIFSpike":
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples += current_batch_size
            module_call_counter.clear()
            # FIX: clear spike traces to prevent memory accumulation across batches
            for m in model.modules():
                if m.__class__.__name__ == 'LIFSpike':
                    if hasattr(m, 'forward_spike_traces'):
                        m.forward_spike_traces = []
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
        layer_names.append(f"{mod_name}:{call_idx}" if call_idx > 1 else mod_name)
    layer_names = [standardize_layer_name(n) for n in layer_names]

    final_acc = 100.0 * correct / total
    network_avg_spikes = sum(number_of_spikes) / len(number_of_spikes) if number_of_spikes else 0.0
    avg_raw_spikes_per_image = raw_spike_total / total_samples

    print(f"Accuracy: {final_acc:.2f}%")
    print(f"NetworkAvgSpikes: {network_avg_spikes:.4f}")
    print(f"AvgRawSpikesPerImage: {avg_raw_spikes_per_image:.2f}")
    print(f"TotalSamples: {total_samples}")

    return final_acc, network_avg_spikes, avg_raw_spikes_per_image, number_of_neurons, number_of_spikes, layer_names


# main
num_classes = 10 if args.use_cifar10 else 100

train_dataset, val_dataset = data_loaders.build_cifar_qcfs(
    cutout=False, use_cifar10=args.use_cifar10, download=True
)
test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True
)

model_map = {"resnet18": resnet18, "resnet19": resnet19, "vgg16": vgg16}
model = model_map[args.model](num_classes=num_classes)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.T = args.T
model = model.to(device)

print(f"\n{'='*60}")
print(f"Model: {args.model} | Dataset: {'CIFAR-10' if args.use_cifar10 else 'CIFAR-100'} | T={args.T}")
print(f"Checkpoint: {args.checkpoint}")
print(f"{'='*60}")

final_acc, network_avg_spikes, avg_raw_spikes, number_of_neurons, number_of_spikes, layer_names = test_with_spikes(model, test_loader, device)

dataset_str = 'cifar10' if args.use_cifar10 else 'cifar100'
plot_name = f"{args.run_name}_{args.model}_{dataset_str}_T{args.T}_acc{final_acc:.2f}.png"

plt.figure(figsize=(12, 5))
plt.bar(range(len(number_of_spikes)), number_of_spikes, color='skyblue', edgecolor='black')
plt.xticks(range(len(number_of_spikes)), layer_names, rotation=45, ha="right", fontsize=8)
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Average Spikes per Neuron", fontsize=12)
plt.title(f"{args.model} | {'CIFAR-10' if args.use_cifar10 else 'CIFAR-100'} | T={args.T} | Acc={final_acc:.2f}%", fontsize=14)
for i, v in enumerate(number_of_spikes):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(plot_name, dpi=300)
plt.close()
print(f"Plot saved: {plot_name}")