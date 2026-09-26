"""
aux1_parse_logs.py
------------------
Parses all .log files in the Logs/ folder and extracts:
  - Final BestTest accuracy  (from "Final BestTest: XX.XXX" line)
  - NetworkAvgSpikes          (from "Epoch:[299/300]" line)

Output is grouped by dataset -> architecture -> sorted by spikes ascending.
Optionally filter by dataset and/or architecture.

Usage:
    python aux1_parse_logs.py
    python aux1_parse_logs.py --dataset cifar10
    python aux1_parse_logs.py --dataset cifar100 --arch resnet19
    python aux1_parse_logs.py --arch vgg16
    python aux1_parse_logs.py --logs_dir /path/to/Logs --dataset cifar10 --arch resnet18
"""

import os
import re
import argparse
import glob
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--logs_dir', default='Logs',
                    help='Path to Logs folder (default: Logs)')
parser.add_argument('--out', default=None,
                    help='Output txt path (default: <logs_dir>/results_summary.txt)')
parser.add_argument('--dataset', default=None,
                    choices=['cifar10', 'cifar100'],
                    help='Filter output to a specific dataset (cifar10 or cifar100)')
parser.add_argument('--arch', default=None,
                    choices=['resnet18', 'resnet19', 'vgg16'],
                    help='Filter output to a specific architecture')
args = parser.parse_args()

logs_dir = args.logs_dir
out_path = args.out or os.path.join(logs_dir, 'results_summary.txt')

DATASET_MAP = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
ARCH_MAP    = {'resnet18': 'ResNet18', 'resnet19': 'ResNet19', 'vgg16': 'VGG16'}

filter_dataset = DATASET_MAP[args.dataset] if args.dataset else None
filter_arch    = ARCH_MAP[args.arch]       if args.arch    else None

RE_BEST_TEST  = re.compile(r'Final BestTest:\s*([\d.]+)')
RE_EPOCH_299  = re.compile(r'Epoch:\[299/\d+\].*?NetworkAvgSpikes=([\d.]+)')
RE_EPOCH_LAST = re.compile(r'Epoch:\[\d+/300\].*?NetworkAvgSpikes=([\d.]+)')


def parse_log(filepath):
    best_test = None
    net_avg_spikes = None
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()
    for line in reversed(lines):
        m = RE_BEST_TEST.search(line)
        if m:
            best_test = float(m.group(1))
            break
    for line in lines:
        m = RE_EPOCH_299.search(line)
        if m:
            net_avg_spikes = float(m.group(1))
            break
    if net_avg_spikes is None:
        for line in lines:
            m = RE_EPOCH_LAST.search(line)
            if m:
                net_avg_spikes = float(m.group(2))
    return best_test, net_avg_spikes


def infer_dataset_arch(filename):
    name = filename.lower()
    dataset = 'CIFAR-100' if 'cifar100' in name else ('CIFAR-10' if 'cifar10' in name else 'Unknown Dataset')
    if 'resnet19' in name:
        arch = 'ResNet19'
    elif 'resnet18' in name:
        arch = 'ResNet18'
    elif 'vgg16' in name:
        arch = 'VGG16'
    else:
        arch = 'Unknown Arch'
    return dataset, arch


# ── Collect and parse ─────────────────────────────────────────────
log_files = sorted(glob.glob(os.path.join(logs_dir, '*.log')))

if not log_files:
    print(f"No .log files found in {logs_dir}")
    exit(1)

print(f"Found {len(log_files)} log files in {logs_dir}")
if filter_dataset or filter_arch:
    parts = []
    if filter_dataset: parts.append(f"dataset={filter_dataset}")
    if filter_arch:    parts.append(f"arch={filter_arch}")
    print(f"Filtering: {', '.join(parts)}")
print()

grouped = defaultdict(lambda: defaultdict(list))

for lf in log_files:
    name = os.path.basename(lf)
    best_test, net_spikes = parse_log(lf)
    dataset, arch = infer_dataset_arch(name)

    if filter_dataset and dataset != filter_dataset:
        continue
    if filter_arch and arch != filter_arch:
        continue

    grouped[dataset][arch].append((name, net_spikes, best_test))

ARCH_ORDER    = ['ResNet18', 'ResNet19', 'VGG16', 'Unknown Arch']
DATASET_ORDER = ['CIFAR-10', 'CIFAR-100', 'Unknown Dataset']

# ── Build output ──────────────────────────────────────────────────
output_lines = []

for dataset in DATASET_ORDER:
    if dataset not in grouped:
        continue

    output_lines.append(f"{'=' * 80}")
    output_lines.append(f"  DATASET: {dataset}")
    output_lines.append(f"{'=' * 80}")

    for arch in ARCH_ORDER:
        if arch not in grouped[dataset]:
            continue

        rows = grouped[dataset][arch]
        rows.sort(key=lambda r: (r[1] is None, r[1]))

        col1 = max(max(len(r[0]) for r in rows), 40)
        divider    = "  " + "-" * (col1 + 40)
        col_header = f"  {'Log File':<{col1}}  {'NetworkAvgSpikes':>18}  {'BestTest Acc (%)':>16}"

        output_lines.append("")
        output_lines.append(f"  [ {arch} ]")
        output_lines.append(divider)
        output_lines.append(col_header)
        output_lines.append(divider)

        for name, spikes, acc in rows:
            spikes_str = f"{spikes:.4f}" if spikes is not None else "N/A"
            acc_str    = f"{acc:.3f}"    if acc    is not None else "N/A"
            output_lines.append(f"  {name:<{col1}}  {spikes_str:>18}  {acc_str:>16}")

        output_lines.append(divider)

    output_lines.append("")

if not any(grouped.values()):
    output_lines.append("  No matching results found for the specified filters.")

output = "\n".join(output_lines)
print(output)

with open(out_path, 'w') as f:
    f.write(output + "\n")

print(f"\nSaved: {out_path}")