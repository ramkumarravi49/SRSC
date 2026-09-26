import argparse
import torch
import torch.nn as nn

# Import your model + LIFSpike
from models.resnet_models import resnet19
from models.layers import LIFSpike


# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser(description="Print LIF thresholds")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (best_model.pth)")
parser.add_argument("--num_classes", type=int, default=10)
args = parser.parse_args()


# -------------------- LOAD MODEL -------------------- #
print("Loading model...")

model = resnet19(num_classes=args.num_classes)
checkpoint = torch.load(args.checkpoint, map_location="cpu")

state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict, strict=False)

model.eval()

print("\n======================================")
print("      LEARNED THRESHOLD VALUES        ")
print("======================================\n")


# -------------------- SCAN FOR THRESHOLDS -------------------- #
threshold_list = []

for name, module in model.named_modules():
    if isinstance(module, LIFSpike):
        value = float(module.thresh.data.item())
        threshold_list.append((name, value))

# Sort for clean output
threshold_list = sorted(threshold_list, key=lambda x: x[0])


# -------------------- PRINT RESULTS -------------------- #
for idx, (name, val) in enumerate(threshold_list):
    print(f"[{idx:02d}]  {name:<35}  thresh = {val:.6f}")

print("\n======================================")
print(f"Total LIFSpike modules found: {len(threshold_list)}")
print("======================================\n")
