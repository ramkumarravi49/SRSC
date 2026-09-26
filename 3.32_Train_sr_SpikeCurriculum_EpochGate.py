# 3.32_Train_sr_SpikeCurriculum_EpochGate.py
#
# EPOCH-GATE ABLATION of 3.3_Train_sr_SpikeCurriculum.py.
# Cloned verbatim from 3.3; the ONLY change is the gate that starts the ramp:
#   3.3   : ramp begins when test accuracy first crosses --acc_gate.
#   3.32  : ramp begins at a fixed epoch, --epoch_gate (default 75).
# Everything else is identical -- the base->ceil geometric ramp over
# SPIKE_RAMP_EPOCHS, the min() ratchet, the fine-tuning phase, checkpointing.
# Default epoch_gate=75 = the first quarter of the 300-epoch training run
# (spend one quarter at the negligible lambda_base, then start ramping);
# it also matches where 3.3's accuracy gate actually fired in the logged runs
# (epoch 72 for ResNet-18/19, 55 for VGG-16), so this is a like-for-like swap.
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
from tqdm import tqdm

from models.VGG_models import *
from models.resnet_models import *
import data_loaders_qcfs as data_loaders

from functions import TET_loss, seed_all, get_logger
from datetime import datetime
import logging
import glob


# -------------------- SPIKE CURRICULUM CONFIG -------------------- #
SPIKE_RAMP_EPOCHS = 150
# ----------------------------------------------------------------- #


# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser(description='Resume Training from Any Checkpoint')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, dest='lr')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--T', default=2, type=int)
parser.add_argument('--means', default=1.0, type=float)
parser.add_argument('--TET', default=True, type=bool)
parser.add_argument('--lamb', default=1e-3, type=float)
parser.add_argument('--use_cifar10', default=True, type=lambda x: x.lower() != 'false',
                    help='Use CIFAR-10 if True, CIFAR-100 if False (default: True)')
parser.add_argument('--arch', default='resnet18', type=str,
                    choices=['resnet18', 'resnet19', 'vgg16'],
                    help='Model architecture (default: resnet18)')

# --- Spike curriculum arguments ---
parser.add_argument('--lambda_spike_base', default=1e-8, type=float)
parser.add_argument('--lambda_spike_ceil', default=5e-7, type=float)
# EPOCH-GATE ABLATION: the ramp begins at this fixed epoch, replacing 3.3's
# accuracy gate (--acc_gate). Default 75 = first quarter of the 300 training
# epochs.
parser.add_argument('--epoch_gate', default=75, type=int,
                    help='Epoch at which the lambda ramp begins (replaces the accuracy gate).')

# Fine tuning
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--fine_time', default=4, type=int)
parser.add_argument('--fine_lr', default=0.0001, type=float)

parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--auto_resume', action='store_true', default=True)

args = parser.parse_args()
# ---------------------------------------------------- #

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using GPU(s): {args.gpu}")
print(f"Device resolved as: {device}")


# ---------------- CHECKPOINT UTILITIES ---------------- #
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, -1
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None, -1
    epoch_numbers = []
    for ckpt_file in checkpoint_files:
        try:
            epoch_num = int(os.path.basename(ckpt_file).split('_')[-1].replace('.pth', ''))
            epoch_numbers.append((epoch_num, ckpt_file))
        except:
            continue
    if not epoch_numbers:
        return None, -1
    epoch_numbers.sort(reverse=True)
    return epoch_numbers[0][1], epoch_numbers[0][0]


# ---------------- SPIKE CURRICULUM UTILITY ---------------- #
def compute_growth_factor(args):
    return (args.lambda_spike_ceil / args.lambda_spike_base) ** (1.0 / SPIKE_RAMP_EPOCHS)


def update_lambda_spike(current_lambda, gate_triggered, epoch, args, growth_factor):
    # EPOCH-GATE ABLATION: trigger on epoch instead of test accuracy.
    if not gate_triggered and epoch >= args.epoch_gate:
        gate_triggered = True
    if gate_triggered:
        new_lambda = min(current_lambda * growth_factor, args.lambda_spike_ceil)
    else:
        new_lambda = current_lambda
    return new_lambda, gate_triggered


# ---------------- TRAIN FUNCTION ---------------- #
def train(model, device, train_loader, criterion, optimizer, epoch, args, current_lambda_spike):
    running_loss = 0
    total = 0
    correct = 0
    model.train()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (images, labels) in loop:
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        # ---- FIX 1: Clear accumulated spike traces before each forward pass ----
        for module in model.modules():
            if isinstance(module, LIFSpike):
                module.forward_spike_traces = []
        # ------------------------------------------------------------------------

        outputs = model(images)
        mean_out = outputs.mean(1)

        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)

        ##################################################################
        # ================ PHYSICAL SPIKE REGULARIZATION ================
        # FIX 2: Iterate forward_spike_traces list (not singular trace).
        #        This captures BOTH activation events in ResNet18 BasicBlock
        #        (post-conv1 and post-residual share the same LIF module).
        # FIX 3: T = count_per_t.numel() — derived from data, safe for
        #        fine-tuning when model.T changes from 2 to 4.
        ##################################################################
        base_loss_value = loss.item()
        raw_spike_loss = 0.0
        scaled_spike_loss = 0.0

        if current_lambda_spike > 0:
            per_layer_losses = []
            for module in model.modules():
                if isinstance(module, LIFSpike):
                    for spikes in getattr(module, "forward_spike_traces", []):
                        count_per_t = spikes.flatten(2).sum(dim=2).mean(dim=0)
                        T = count_per_t.numel()  # FIX 3: derived from data
                        time_weights = torch.linspace(1.0 / T, 1.0, T, device=outputs.device)
                        weighted = (time_weights * count_per_t).sum()
                        per_layer_losses.append(current_lambda_spike * weighted)

            if per_layer_losses:
                scaled_loss = torch.stack(per_layer_losses).sum()
                raw_spike_loss = sum(
                    spikes.flatten(2).sum(dim=2).mean(dim=0).sum().item()
                    for module in model.modules()
                    if isinstance(module, LIFSpike)
                    for spikes in getattr(module, "forward_spike_traces", [])
                )
                scaled_spike_loss = scaled_loss.item()
                loss = loss + scaled_loss
                base_loss_value = loss.item() - scaled_spike_loss
        ##################################################################

        running_loss += loss.item()
        loss.mean().backward()

        global_step = epoch * len(train_loader) + i
        writer.add_scalar("Loss/Base_TET_or_CE", base_loss_value, global_step)
        writer.add_scalar("Loss/SpikeRaw", raw_spike_loss, global_step)
        writer.add_scalar("Loss/SpikeScaled", scaled_spike_loss, global_step)
        writer.add_scalar("Loss/Total", loss.item(), global_step)

        if i % 60 == 0:
            logger.info(
                f"[MiniBatch {i:04d}] "
                f"BaseLoss={base_loss_value:.4f} "
                f"SpikeRaw={raw_spike_loss:.4f} "
                f"SpikeScaled={scaled_spike_loss:.6f} "
                f"TotalLoss={loss.item():.4f} "
                f"LambdaSpike={current_lambda_spike:.2e}"
            )

        optimizer.step()

        total += labels.size(0)
        _, predicted = mean_out.cpu().max(1)
        correct += predicted.eq(labels.cpu()).sum().item()

        loop.set_description(f"Epoch [{epoch+1}]")
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    return running_loss, 100 * correct / total


# ---------------- TEST FUNCTION ---------------- #
@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    proxy_total_spikes = 0
    lif_total_spikes = 0
    total_samples = 0
    current_batch_size = 0

    model.eval()

    number_of_neurons = []
    spike_sum_over_samples = []
    layer_index = {}
    module_call_counter = {}
    module_to_name = {m: n for n, m in model.named_modules()}

    def spike_hook(module, inputs, out):
        nonlocal lif_total_spikes, current_batch_size
        spk = (out > 0).float()
        lif_total_spikes += spk.sum().item()
        if spk.dim() >= 3 and spk.size(0) == current_batch_size:
            spk = spk.transpose(0, 1)
        feature_dims = tuple(range(2, spk.dim()))
        num_neurons = 1
        for d in feature_dims:
            num_neurons *= spk.size(d)
        spikes_per_sample = spk.sum(dim=(0,) + feature_dims) / num_neurons
        cnt = module_call_counter.get(module, 0) + 1
        module_call_counter[module] = cnt
        key = (module_to_name[module], cnt)
        idx = layer_index.get(key)
        if idx is None:
            idx = len(number_of_neurons)
            layer_index[key] = idx
            number_of_neurons.append(num_neurons)
            spike_sum_over_samples.append(spikes_per_sample.sum().item())
        else:
            spike_sum_over_samples[idx] += spikes_per_sample.sum().item()

    hooks = [m.register_forward_hook(spike_hook)
             for m in model.modules() if m.__class__.__name__ == "LIFSpike"]

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        current_batch_size = inputs.size(0)
        total_samples += current_batch_size
        module_call_counter.clear()

        # ---- FIX 4: Clear spike traces to prevent accumulation across test batches ----
        for m in model.modules():
            if m.__class__.__name__ == 'LIFSpike':
                m.forward_spike_traces = []
        # -------------------------------------------------------------------------------

        outputs = model(inputs)
        mean_out = outputs.mean(1)
        proxy_total_spikes += (outputs > 0).sum().item()
        _, predicted = mean_out.cpu().max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.cpu()).sum().item()

    for h in hooks:
        h.remove()

    acc = 100 * correct / total
    proxy_avg_spikes = proxy_total_spikes / total_samples
    number_of_spikes = [s / total_samples for s in spike_sum_over_samples]
    network_avg_spikes = sum(number_of_spikes) / len(number_of_spikes) if number_of_spikes else 0.0

    return acc, proxy_avg_spikes, network_avg_spikes, lif_total_spikes, total_samples


# ===================== MAIN ===================== #
if __name__ == '__main__':
    seed_all(args.seed)

    run_id = args.run_name
    checkpoint_dir = os.path.join("CHECKPOINTS", run_id)

    # ============ FIND CHECKPOINT ============ #
    if args.checkpoint_path:
        checkpoint_file = args.checkpoint_path
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_file}")
    elif args.auto_resume:
        checkpoint_file, resume_epoch = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_file is None:
            print(f"No checkpoint found in {checkpoint_dir} — starting fresh.")
            start_epoch = 0
            checkpoint = None
        else:
            print(f"Auto-detected checkpoint: {checkpoint_file}")
            start_epoch = resume_epoch + 1
    else:
        raise ValueError("Must specify --checkpoint_path or use --auto_resume")

    # ============ LOAD CHECKPOINT ============ #
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', 0.0)
        current_lambda_spike = checkpoint.get('current_lambda_spike', args.lambda_spike_base)
        spike_gate_triggered = checkpoint.get('spike_gate_triggered', False)
        print(f"Resuming from epoch {start_epoch} | best_acc={best_test_acc:.3f}")
        print(f"lambda_spike={current_lambda_spike:.2e} | gate={spike_gate_triggered}")
    else:
        start_epoch = 0
        best_test_acc = 0.0
        checkpoint = None
        current_lambda_spike = args.lambda_spike_base
        spike_gate_triggered = False
        print("No checkpoint — starting from scratch.")

    growth_factor = compute_growth_factor(args)
    print(f"Growth factor: {growth_factor:.6f}  (base→ceil in {SPIKE_RAMP_EPOCHS} epochs)")

    # ============ LOGGING ============ #
    writer = SummaryWriter(log_dir=os.path.join("Logs", "runs", run_id), purge_step=start_epoch)

    log_path = os.path.join("Logs", f"{run_id}.log")
    logger = logging.getLogger('Train_sr_SpikeCurriculum_EpochGate')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ============ DATA ============ #
    train_dataset, val_dataset = data_loaders.build_cifar_qcfs(
        cutout=True, use_cifar10=args.use_cifar10, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # ============ MODEL ============ #
    # FIX 5: --arch argument replaces hardcoded model (was vgg16 in original 3.3)
    num_classes = 10 if args.use_cifar10 else 100
    if args.arch == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif args.arch == 'resnet19':
        model = resnet19(num_classes=num_classes)
    elif args.arch == 'vgg16':
        model = vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    model.T = args.T
    print(f"Architecture: {args.arch} | Classes: {num_classes} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # ============ OPTIMIZER & SCHEDULER ============ #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    best_train_acc = 0
    best_epoch = start_epoch

    # ============ LOG CONFIG ============ #
    logger.info("=" * 80)
    logger.info("SRSC EPOCH-GATE ABLATION (3.32) — cloned from 3.3, ramp starts at --epoch_gate")
    logger.info("RESUMING" if checkpoint is not None else "FRESH START")
    logger.info("=" * 80)
    for arg, val in vars(args).items():
        logger.info(f"  {arg}: {val}")
    logger.info(f"  SPIKE_RAMP_EPOCHS: {SPIKE_RAMP_EPOCHS}")
    logger.info(f"  growth_factor: {growth_factor:.6f}")
    logger.info(f"  initial lambda_spike: {current_lambda_spike:.2e}")
    logger.info(f"  gate_triggered: {spike_gate_triggered}")
    logger.info(f"  epoch_gate: {args.epoch_gate}")
    logger.info("=" * 80)

    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every = 100

    # ============================================================ #
    #                      TRAINING PHASE                           #
    # ============================================================ #
    if start_epoch < args.epochs:
        logger.info(f"TRAINING phase: epochs {start_epoch} → {args.epochs - 1}")

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_loss, train_acc = train(
                model, device, train_loader, criterion,
                optimizer, epoch, args, current_lambda_spike
            )
            best_train_acc = max(best_train_acc, train_acc)
            scheduler.step()

            test_acc, proxy_avg_spikes, network_avg_spikes, lif_total_spikes, total_samples = test(
                model, test_loader, device
            )

            prev_lambda = current_lambda_spike
            prev_gate = spike_gate_triggered
            current_lambda_spike, spike_gate_triggered = update_lambda_spike(
                current_lambda_spike, spike_gate_triggered, epoch, args, growth_factor
            )

            if spike_gate_triggered and not prev_gate:
                logger.info(f"[SpikeCurriculum] *** EPOCH-GATE TRIGGERED at epoch {epoch} (epoch_gate={args.epoch_gate}) ***")
            if spike_gate_triggered and current_lambda_spike != prev_lambda:
                logger.info(f"[SpikeCurriculum] Gate ACTIVE | lambda: {prev_lambda:.2e} → {current_lambda_spike:.2e}")
            elif not spike_gate_triggered:
                logger.info(f"[SpikeCurriculum] Gate WAITING (epoch={epoch} < {args.epoch_gate}) | lambda held at {current_lambda_spike:.2e}")

            writer.add_scalar("SpikeCurriculum/lambda_spike", current_lambda_spike, epoch)
            writer.add_scalar("SpikeCurriculum/gate_triggered", int(spike_gate_triggered), epoch)

            # Log thresholds
            for name, module in model.named_modules():
                if isinstance(module, LIFSpike):
                    writer.add_scalar(f"Thresholds/{name}", float(module.thresh.item()), epoch)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc,
                    'current_lambda_spike': current_lambda_spike,
                    'spike_gate_triggered': spike_gate_triggered,
                }, os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"Saved BEST model at epoch {epoch + 1} (acc={test_acc:.3f})")

            if (epoch + 1) % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc,
                    'current_lambda_spike': current_lambda_spike,
                    'spike_gate_triggered': spike_gate_triggered,
                }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
                logger.info(f"Saved checkpoint at epoch {epoch + 1}")

            peak_mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch:[{epoch}/{args.epochs}] "
                f"LR={current_lr:.6f} "
                f"TrainLoss={train_loss:.5f} TrainAcc={train_acc:.3f} "
                f"TestAcc={test_acc:.3f} "
                f"LambdaSpike={current_lambda_spike:.2e} "
                f"GateTriggered={spike_gate_triggered} "
                f"ReLU-Spikes={proxy_avg_spikes:.3f} "
                f"NetworkAvgSpikes={network_avg_spikes:.3f} "
                f"LIFTotalSpikesPerImg={lif_total_spikes / total_samples:.3f} "
                f"GPU={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s"
            )

            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            writer.add_scalar("GPU/PeakGB", peak_mem_gb, epoch)
            writer.add_scalar("Time/EpochSec", epoch_time, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            writer.add_scalar("Spikes/Proxy_Avg_per_inf", proxy_avg_spikes, epoch)
            writer.add_scalar("Spikes/LIF_Avg_per_inf", network_avg_spikes, epoch)
            writer.add_scalar("Spikes/LIF_Total_per_inf", lif_total_spikes / total_samples, epoch)

        logger.info("Training phase complete.")
        start_epoch = args.epochs

    # ============================================================ #
    #                     FINE-TUNING PHASE                         #
    # ============================================================ #
    if start_epoch >= args.epochs:
        logger.info("=" * 80)
        logger.info("FINE-TUNING phase...")
        logger.info(f"lambda_spike entering fine-tuning: {current_lambda_spike:.2e}")
        logger.info("=" * 80)

        model.T = args.fine_time
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr)
        fine_start = max(0, start_epoch - args.epochs)

        for fine_epoch in range(fine_start, args.fine_epochs):
            epoch = args.epochs + fine_epoch
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_loss, train_acc = train(
                model, device, train_loader, criterion,
                optimizer, epoch, args, current_lambda_spike
            )
            best_train_acc = max(best_train_acc, train_acc)

            test_acc, proxy_avg_spikes, network_avg_spikes, lif_total_spikes, total_samples = test(
                model, test_loader, device
            )

            prev_lambda = current_lambda_spike
            current_lambda_spike, spike_gate_triggered = update_lambda_spike(
                current_lambda_spike, spike_gate_triggered, epoch, args, growth_factor
            )
            writer.add_scalar("SpikeCurriculum/lambda_spike", current_lambda_spike, epoch)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_acc': best_test_acc,
                    'current_lambda_spike': current_lambda_spike,
                    'spike_gate_triggered': spike_gate_triggered,
                }, os.path.join(checkpoint_dir, "best_model.pth"))
                logger.info(f"[FineTune] Saved BEST model at epoch {epoch + 1} (acc={test_acc:.3f})")

            peak_mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch:[{epoch}/{args.epochs + args.fine_epochs - 1}] "
                f"TrainLoss={train_loss:.5f} TrainAcc={train_acc:.3f} "
                f"TestAcc={test_acc:.3f} "
                f"LambdaSpike={current_lambda_spike:.2e} "
                f"ReLU-Spikes={proxy_avg_spikes:.3f} "
                f"NetworkAvgSpikes={network_avg_spikes:.3f} "
                f"LIFTotalSpikesPerImg={lif_total_spikes / total_samples:.3f} "
                f"GPU={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s"
            )

            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            writer.add_scalar("GPU/PeakGB", peak_mem_gb, epoch)
            writer.add_scalar("Time/EpochSec", epoch_time, epoch)
            writer.add_scalar("Spikes/Proxy_Avg_per_inf", proxy_avg_spikes, epoch)
            writer.add_scalar("Spikes/LIF_Avg_per_inf", network_avg_spikes, epoch)
            writer.add_scalar("Spikes/LIF_Total_per_inf", lif_total_spikes / total_samples, epoch)

    # ============================================================ #
    #                          WRAP-UP                              #
    # ============================================================ #
    logger.info("=" * 80)
    logger.info("ALL TRAINING COMPLETED")
    logger.info(f"Final BestTrain: {best_train_acc:.3f}")
    logger.info(f"Final BestTest:  {best_test_acc:.3f} at epoch {best_epoch}")
    logger.info(f"Final lambda_spike: {current_lambda_spike:.2e}")
    logger.info(f"Final NetworkAvgSpikes: {network_avg_spikes:.4f}")
    logger.info(f"Final LIFTotalSpikesPerImg: {lif_total_spikes / total_samples:.3f}")
    logger.info("=" * 80)

    writer.close()
    torch.save(model, os.path.join(checkpoint_dir, f"final_model_{run_id}.pth"))
    logger.info(f"Saved final model → {checkpoint_dir}/final_model_{run_id}.pth")
