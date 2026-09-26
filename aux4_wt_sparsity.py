import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from models.VGG_models import *
from models.resnet_models import *
import data_loaders_qcfs as data_loaders

# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--arch', type=str, required=True, choices=['resnet18', 'resnet19', 'vgg16'])
parser.add_argument('--use_cifar10', default=True, type=lambda x: x.lower() != 'false')
parser.add_argument('--T', default=2, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--weight_bits', default=8, type=int)
parser.add_argument('--gpu', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- LOAD MODEL -------------------- #
num_classes = 10 if args.use_cifar10 else 100
if args.arch == 'resnet18':
    model = resnet18(num_classes=num_classes)
elif args.arch == 'resnet19':
    model = resnet19(num_classes=num_classes)
elif args.arch == 'vgg16':
    model = vgg16(num_classes=num_classes)

model.T = args.T
ckpt = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Loaded {args.arch} from {args.checkpoint_path} | best_acc={ckpt.get('best_test_acc', 0.0):.3f}")


# -------------------- DATA -------------------- #
_, val_dataset = data_loaders.build_cifar_qcfs(
    cutout=False, use_cifar10=args.use_cifar10, download=False
)
test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True
)


# -------------------- WEIGHT SPARSITY -------------------- #
def print_weight_sparsity(model, label=""):
    if label:
        print(f"\n{'='*80}")
        print(f"  Weight Sparsity — {label}")
    print("=" * 80)
    print(f"{'Layer':<45} {'Zero':>8} {'Total':>8} {'Sparsity':>10}")
    print("=" * 80)
    total_zeros, total_params = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            continue
        if not (hasattr(module, 'weight') and module.weight is not None):
            continue
        w = module.weight.data
        zeros = (w == 0).sum().item()
        total = w.numel()
        total_zeros  += zeros
        total_params += total
        print(f"{name:<45} {zeros:>8} {total:>8} {zeros/total*100:>9.2f}%")
    print("-" * 80)
    overall = total_zeros / total_params * 100 if total_params else 0.0
    print(f"{'OVERALL':<45} {total_zeros:>8} {total_params:>8} {overall:>9.2f}%")
    print("=" * 80)
    return overall


# -------------------- INFERENCE -------------------- #
@torch.no_grad()
def run_inference(model, loader, label=""):
    correct, total = 0, 0
    loop = tqdm(loader, desc=f"Inferencing [{label}]", unit="batch")
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        for m in model.modules():
            if m.__class__.__name__ == 'LIFSpike':
                m.forward_spike_traces = []
        outputs = model(imgs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels.cpu()).sum().item()
        loop.set_postfix(acc=f"{100*correct/total:.2f}%")
    acc = 100 * correct / total
    print(f"  → Accuracy [{label}]: {acc:.3f}%")
    return acc


# -------------------- FAKE QUANTIZATION -------------------- #
@torch.no_grad()
def fake_quantize_weights(model, n_bits=8):
    """
    Per-tensor symmetric uniform fake-quantization of Conv2d and Linear weights.
    Steps:
      1. compute scale = max(|w|) / (2^(n_bits-1) - 1)
      2. w_int = clamp(round(w / scale), -qmax, qmax)
      3. w_fakeq = w_int * scale   ← back to FP32, zero-bin weights are exactly 0.0
    Skips BN, LIFSpike, and any layer without a weight.
    """
    qmax = (1 << (n_bits - 1)) - 1  # 127 for 8-bit
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            max_abs = w.abs().max().clamp_min(1e-12)
            scale   = max_abs / float(qmax)
            w_int   = torch.clamp(torch.round(w / scale), -qmax, qmax)
            w_fakeq = w_int * scale          # back to FP32
            module.weight.data.copy_(w_fakeq)
    return model


# ==================== MAIN ==================== #

# --- Step 1: baseline sparsity + accuracy ---
print("\n>>> BEFORE fake-quantization")
sparsity_before = print_weight_sparsity(model, label="FP32 (before)")
acc_before = run_inference(model, test_loader, label="FP32 before")

# --- Step 2: apply fake-quant IN-PLACE ---
print(f"\n>>> Applying {args.weight_bits}-bit symmetric fake-quantization to weights...")
fake_quantize_weights(model, n_bits=args.weight_bits)
print("    Done.")

# --- Step 3: post-quant sparsity + accuracy ---
print(f"\n>>> AFTER {args.weight_bits}-bit fake-quantization")
sparsity_after = print_weight_sparsity(model, label=f"{args.weight_bits}-bit fake-quant")
acc_after = run_inference(model, test_loader, label=f"{args.weight_bits}-bit fake-quant")

# --- Step 4: summary ---
print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
print(f"  Accuracy  : {acc_before:.3f}% → {acc_after:.3f}%  (drop={acc_before-acc_after:.3f}%)")
print(f"  Wt Sparsity: {sparsity_before:.2f}% → {sparsity_after:.2f}%  (+{sparsity_after-sparsity_before:.2f}%)")
print("=" * 50)

# # --- Step 5: save fake-quantized model ---
# save_dir  = os.path.dirname(args.checkpoint_path)
# save_name = f"best_model_fakeq{args.weight_bits}bit.pth"
# save_path = os.path.join(save_dir, save_name)
# torch.save({
#     'model_state_dict'  : model.state_dict(),
#     'best_test_acc'     : acc_after,
#     'weight_bits'       : args.weight_bits,
#     'acc_before_quant'  : acc_before,
#     'sparsity_before'   : sparsity_before,
#     'sparsity_after'    : sparsity_after,
# }, save_path)
# print(f"\n  Saved fake-quantized model → {save_path}")