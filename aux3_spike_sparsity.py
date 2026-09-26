import argparse
import torch
import torch.nn as nn
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
parser.add_argument('--num_batches', default=None, type=int,
                    help='Number of batches to evaluate (default: full test set)')
parser.add_argument('--gpu', type=str, default="0")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- MODEL -------------------- #
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
print(f"Loaded {args.arch} from {args.checkpoint_path} | best_acc={ckpt.get('best_test_acc', '?'):.3f}")


# -------------------- DATA -------------------- #
_, val_dataset = data_loaders.build_cifar_qcfs(
    cutout=False, use_cifar10=args.use_cifar10, download=False
)
test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True
)
num_batches = args.num_batches if args.num_batches is not None else len(test_loader)


# -------------------- SPIKE SPARSITY -------------------- #
def print_spike_sparsity(model, data_loader, device, T, num_batches):
    spike_stats = {}
    hooks = []

    def make_hook(layer_name, t):
        def hook(module, inp, out):
            tb = out.shape[0]
            b  = tb // t
            spatial = out.shape[1:]
            spikes = out.detach().view(t, b, *spatial)   # stay on GPU
            fired  = (spikes > 0).any(dim=0)
            dead   = int((~fired).sum().item())           # single scalar transfer
            total  = fired.numel()
            del spikes, fired
            if layer_name not in spike_stats:
                spike_stats[layer_name] = {"dead": 0, "total": 0}
            spike_stats[layer_name]["dead"]  += dead
            spike_stats[layer_name]["total"] += total
        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LIFSpike':
            h = module.register_forward_hook(make_hook(name, T))
            hooks.append(h)

    correct = 0
    total = 0
    from tqdm import tqdm
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=num_batches, desc="Inferencing", unit="batch")
        for i, (imgs, labels) in loop:
            if i >= num_batches:
                break
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # clear traces (consistent with 3.3)
            for m in model.modules():
                if m.__class__.__name__ == 'LIFSpike':
                    m.forward_spike_traces = []

            outputs = model(imgs)
            mean_out = outputs.mean(1)
            _, predicted = mean_out.cpu().max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels.cpu()).sum().item()
            del imgs

            loop.set_postfix(acc=f"{100*correct/total:.2f}%")

    for h in hooks:
        h.remove()

    acc = 100 * correct / total
    print(f"\nTest Accuracy over {i+1} batches: {acc:.3f}%")

    print("\n" + "=" * 70)
    print(f"{'IF Layer':<45} {'Dead':>8} {'Total':>8} {'Sparsity':>10}")
    print("=" * 70)
    total_dead, total_neurons = 0, 0
    for name, stats in spike_stats.items():
        dead, total = stats["dead"], stats["total"]
        total_dead    += dead
        total_neurons += total
        print(f"{name:<45} {dead:>8} {total:>8} {dead/total*100:>9.2f}%")
    print("-" * 70)
    overall = total_dead / total_neurons * 100 if total_neurons else 0.0
    print(f"{'OVERALL':<45} {total_dead:>8} {total_neurons:>8} {overall:>9.2f}%")
    print("=" * 70)


print_spike_sparsity(model, test_loader, device, args.T, num_batches)