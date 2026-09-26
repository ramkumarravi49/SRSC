# #!/usr/bin/env python3
# """
# Plot threshold progression across epochs from TET-LT training logs.

# Usage:
#     python plot_threshold_progression.py path/to/train.log

# Outputs:
#     threshold_progression_all_layers.pdf
# """

# import re
# import sys
# from collections import defaultdict
# import matplotlib.pyplot as plt


# def parse_thresholds(log_path):
#     """
#     Parses a training log and extracts threshold values per epoch.

#     Returns:
#         thresholds: dict[layer_name] -> list of (epoch, value)
#     """
#     epoch_pat = re.compile(r"Epoch:\[(\d+)/")
#     thresh_pat = re.compile(
#         r"Threshold\[(.*?)\]\s*=\s*([-+]?\d*\.\d+|\d+)"
#     )

#     thresholds = defaultdict(list)
#     current_epoch = None

#     with open(log_path, "r") as f:
#         for line in f:
#             # Detect epoch line
#             epoch_match = epoch_pat.search(line)
#             if epoch_match:
#                 current_epoch = int(epoch_match.group(1))

#             # Detect threshold line
#             thresh_match = thresh_pat.search(line)
#             if thresh_match and current_epoch is not None:
#                 layer = thresh_match.group(1)
#                 value = float(thresh_match.group(2))
#                 thresholds[layer].append((current_epoch, value))

#     return thresholds


# def plot_thresholds(thresholds, save_path):
#     """
#     Plots threshold progression for all layers.
#     """
#     plt.figure(figsize=(10, 6))

#     for layer, values in sorted(thresholds.items()):
#         epochs, vals = zip(*values)
#         plt.plot(epochs, vals, label=layer, alpha=0.85)

#     plt.xlabel("Epoch")
#     plt.ylabel("Learned Threshold Value")
#     plt.title("Threshold Progression Across Layers (TET-LT)")
#     plt.grid(True)

#     # Comment this out if legend becomes too crowded
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"[INFO] Saved threshold progression plot to: {save_path}")


# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python plot_threshold_progression.py path/to/train.log")
#         sys.exit(1)

#     log_path = sys.argv[1]
#     output_path = "threshold_progression_all_layers.pdf"

#     thresholds = parse_thresholds(log_path)

#     if not thresholds:
#         print("[ERROR] No thresholds found in log file.")
#         sys.exit(1)

#     plot_thresholds(thresholds, output_path)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Plot threshold progression across epochs from TET-LT training logs.

Usage:
    python plot_threshold_progression.py path/to/train.log

Outputs:
    threshold_progression_all_layers.png
    threshold_progression_all_layers.svg
"""

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_thresholds(log_path):
    epoch_pat = re.compile(r"Epoch:\[(\d+)/")
    thresh_pat = re.compile(
        r"Threshold\[(.*?)\]\s*=\s*([-+]?\d*\.\d+|\d+)"
    )

    thresholds = defaultdict(list)
    current_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            epoch_match = epoch_pat.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            thresh_match = thresh_pat.search(line)
            if thresh_match and current_epoch is not None:
                layer = thresh_match.group(1)
                value = float(thresh_match.group(2))
                thresholds[layer].append((current_epoch, value))

    return thresholds


def plot_thresholds(thresholds):
    plt.figure(figsize=(10, 6))

    for layer, values in sorted(thresholds.items()):
        epochs, vals = zip(*values)
        plt.plot(epochs, vals, label=layer, linewidth=1.5, alpha=0.85)

    plt.xlabel("Epoch")
    plt.ylabel("Learned Threshold Value")
    plt.title("Threshold Progression Across Layers (TET-LT)")
    plt.grid(True)

    # COMMENT THIS if legend becomes too crowded
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    plt.savefig("threshold_progression_all_layers.png", dpi=300)
    plt.savefig("threshold_progression_all_layers.svg")
    plt.close()

    print("[INFO] Saved:")
    print("  - threshold_progression_all_layers.png")
    print("  - threshold_progression_all_layers.svg")


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_threshold_progression.py path/to/train.log")
        sys.exit(1)

    log_path = sys.argv[1]
    thresholds = parse_thresholds(log_path)

    if not thresholds:
        print("[ERROR] No thresholds found in log file.")
        sys.exit(1)

    plot_thresholds(thresholds)


if __name__ == "__main__":
    main()
