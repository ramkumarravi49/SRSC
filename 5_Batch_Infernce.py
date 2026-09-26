# 5

# #############################################################
# python batch_spike_inference.py 2>&1 | tee batch_results.log
################################################################
import os
import re
import torch
import data_loaders
import matplotlib.pyplot as plt
from models.VGG_models import *
from models.resnet_models import *
from functions import seed_all

# ============================================================
# USER CONFIGURATION — edit these two lists
# ============================================================

CHECKPOINT_BASE = "/data/cs24m037/TET_LT/CHECKPOINTS"

CHECKPOINT_FOLDERS = [
    "LT_resnet19_2",
    "LT_resnet18",
    "LT_SR_1e3_resnet18",
    "LT_SR_1e3_resnet19",
    "LT_VGG16_T_2_lr_1e3",

    # TEST - ResNet18
    "TEST_Resnet18_1e-6_sr_phy",
    "TEST_Resnet18_3e-7_sr_phy",
    "TEST_Resnet18_5e-7_sr_phy",
    "TEST_Resnet18_7e-7_sr_phy",

    # TEST - ResNet19
    "TEST_2_resnet19_0.00_sr_phy",
    "TEST_resnet19_0.00_sr_phy",
    "TEST_resnet19_1e-6_sr_phy",
    "TEST_resnet19_3e-7_sr_phy",
    "TEST_resnet19_5e-7_sr_phy",
    "TEST_resnet19_5e-7_sr_phy_T_8",
    "TEST_resnet19_7e-7_sr_phy",

    # TEST - VGG16
    "TEST_VGG16_0.0_sr_phy_5e-4_lr",
    "TEST_VGG16_1e-6_sr_phy_1e-3_lr",
    "TEST_VGG16_1e-6_sr_phy_5e-4_lr",
    "TEST_VGG16_5e-7_sr_phy_1e-3_lr",
    "TEST_VGG16_5e-7_sr_phy_5e-4_lr",

    # SRSC - ResNet18
    "SRSC_resnet18_sr_phy_1e-8__1e-6",
    "SRSC_resnet18_sr_phy_1e-8__3e-6",
    "SRSC_resnet18_sr_phy_1e-8__5e-6",
    "SRSC_resnet18_sr_phy_1e-8__5e-7",

    # SRSC - ResNet19
    "SRSC_resnet19_sr_phy_1e-8__1e-6",
    "SRSC_resnet19_sr_phy_1e-8__3e-6",
    "SRSC_resnet19_sr_phy_1e-8__5e-6",
    "SRSC_resnet19_sr_phy_1e-8__5e-7",

    # SRSC - VGG16
    "SRSC_vgg16_sr_phy_1e-8__1e-6",
    "SRSC_vgg16_sr_phy_1e-8__1e-6_acc_gate_87",
    "SRSC_vgg16_sr_phy_1e-8__1e-6_acc_gate_87_lr_5e-4",
    "SRSC_vgg16_sr_phy_1e-8__3e-6_acc_gate_87_lr_5e-4",
    "SRSC_vgg16_sr_phy_1e-8__3e-6_acc_gate_90_lr_5e-4",
    "SRSC_vgg16_sr_phy_1e-8__5e-6_acc_gate_87_lr_5e-4",
    "SRSC_vgg16_sr_phy_1e-8__5e-6_acc_gate_90_lr_5e-4",
    "SRSC_vgg16_sr_phy_1e-8__5e-7",
    "SRSC_vgg16_sr_phy_1e-8__5e-7_acc_gate_87",
    "SRSC_vgg16_sr_phy_1e-8__5e-7_acc_gate_87_lr_5e-4",
]

T_VALUES = [2]   # e.g. [2, 4, 8, 16]

BATCH_SIZE = 128
WORKERS    = 16
DEVICE     = "cuda"

# ============================================================
# Model registry — auto-detected from folder name
# ============================================================

MODEL_MAP = {
    "resnet18": resnet18,
    "resnet19": resnet19,
    "vgg16":    vgg16,
}

def detect_model(folder_name: str):
    """Return the model key by searching the folder name (case-insensitive)."""
    lower = folder_name.lower()
    for key in MODEL_MAP:
        if key in lower:
            return key
    raise ValueError(
        f"Cannot detect model from folder name '{folder_name}'. "
        f"Expected one of: {list(MODEL_MAP.keys())}"
    )

# ============================================================
# Layer name standardization (same as spikes_TET_layer.py)
# ============================================================

def standardize_layer_name(name: str):
    replacements = {
        "layer1": "block1",
        "layer2": "block2",
        "layer3": "block3",
        "layer4": "block4",
        "spike":  "act"
    }
    parts = name.split(".")
    for k, v in replacements.items():
        parts = [p.replace(k, v) for p in parts]
    return ".".join(parts)

# ============================================================
# Inference function (same logic as spikes_TET_layer.py)
# ============================================================

def test_with_spikes(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    number_of_neurons       = []
    spike_sum_over_samples  = []
    layer_index             = {}
    hooks                   = []
    module_call_counter     = {}
    current_batch_size      = 0
    total_samples           = 0
    module_to_name          = {m: n for n, m in model.named_modules()}
    raw_spike_total         = 0

    def hook_fn(module, inputs, out):
        nonlocal current_batch_size, raw_spike_total

        spk = (out > 0).float()
        raw_spike_total += spk.sum().item()

        if spk.dim() >= 3 and spk.size(0) == current_batch_size:
            spk = spk.transpose(0, 1)

        feature_dims = tuple(range(2, spk.dim()))
        num_neurons  = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)

        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons

        cnt      = module_call_counter.get(module, 0) + 1
        module_call_counter[module] = cnt
        mod_name = module_to_name[module]
        key      = (mod_name, cnt)

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
            total_samples     += current_batch_size
            module_call_counter.clear()

            outputs = model(inputs)
            if outputs.dim() == 3:
                mean_out = outputs.mean(1)
            else:
                mean_out = outputs
            _, predicted = mean_out.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    for h in hooks:
        h.remove()

    number_of_spikes = [s / total_samples for s in spike_sum_over_samples]
    inv_map          = {v: k for k, v in layer_index.items()}

    layer_names = []
    for i in range(len(number_of_spikes)):
        mod_name, call_idx = inv_map[i]
        layer_names.append(f"{mod_name}:{call_idx}" if call_idx > 1 else mod_name)

    layer_names = [standardize_layer_name(n) for n in layer_names]
    final_acc   = 100.0 * correct / total

    print("number_of_neurons:", number_of_neurons)
    print("number_of_spikes:", number_of_spikes)
    print("layer_names:", layer_names)
    print(f"Accuracy: {final_acc:.2f}%")
    print("Network average spikes:", sum(number_of_spikes) / len(number_of_spikes))
    print("Total Sample:", total_samples)
    print("Average TOTAL spikes per image (raw count):", raw_spike_total / total_samples)

    return final_acc, number_of_neurons, number_of_spikes, layer_names

# ============================================================
# Data loading (done once, reused for all runs)
# ============================================================

print("Loading dataset...")
train_dataset, val_dataset = data_loaders.build_cifar(
    cutout=True, use_cifar10=True, download=True
)
test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True
)
print("Dataset ready.\n")

# ============================================================
# Main batch loop
# ============================================================

for folder in CHECKPOINT_FOLDERS:
    model_key  = detect_model(folder)
    checkpoint_path = os.path.join(CHECKPOINT_BASE, folder, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[SKIP] Checkpoint not found: {checkpoint_path}\n")
        continue

    for T in T_VALUES:
        print("=" * 70)
        print(f"  Folder : {folder}")
        print(f"  Model  : {model_key}")
        print(f"  T      : {T}")
        print(f"  Ckpt   : {checkpoint_path}")
        print("=" * 70)

        # Build model
        model = MODEL_MAP[model_key](num_classes=10)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.T = T
        model    = model.to(DEVICE)

        # Run inference
        print(f"Test loader batches: {len(test_loader)}")
        print(f"Expected samples: {len(test_loader.dataset)}")
        test_with_spikes(model, test_loader, DEVICE)

        # Free GPU memory before next run
        del model
        torch.cuda.empty_cache()

        print()  # blank line between runs