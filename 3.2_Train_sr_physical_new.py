# 3.2_Train_sr_physical_new.py
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


# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser(description='Resume Training from Any Checkpoint')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int, help='Total training epochs')
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, dest='lr')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--T', default=2, type=int)
parser.add_argument('--means', default=1.0, type=float)
parser.add_argument('--TET', default=True, type=bool)
parser.add_argument('--lamb', default=1e-3, type=float)
parser.add_argument('--lambda_spike', default=0.0, type=float, help='Temporal spike regularization strength')
parser.add_argument('--use_cifar10', default=True, type=lambda x: x.lower() != 'false',
                    help='Use CIFAR-10 if True, CIFAR-100 if False (default: True)')

# Fine tuning
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--fine_time', default=4, type=int)
parser.add_argument('--fine_lr', default=0.0001, type=float)

parser.add_argument('--run_name', type=str, required=True,
                    help='Experiment run name (must match original run)')
parser.add_argument('--gpu', type=str, default="0",
                    help='CUDA_VISIBLE_DEVICES value (e.g. "0", "1", "0,1")')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Specific checkpoint path to resume from (optional - will auto-detect latest)')
parser.add_argument('--auto_resume', action='store_true', default=True,
                    help='Automatically find and resume from latest checkpoint')

args = parser.parse_args()
# ----------------------------------------------------- #

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using GPU(s): {args.gpu}")
print(f"Device resolved as: {device}")


# ---------------- CHECKPOINT UTILITIES ---------------- #
def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the directory.
    Returns: (checkpoint_path, epoch_number) or (None, -1) if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None, -1
    
    # Look for checkpoint_epoch_*.pth files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        return None, -1
    
    # Extract epoch numbers and find the maximum
    epoch_numbers = []
    for ckpt_file in checkpoint_files:
        try:
            # Extract number from "checkpoint_epoch_123.pth"
            basename = os.path.basename(ckpt_file)
            epoch_num = int(basename.split('_')[-1].replace('.pth', ''))
            epoch_numbers.append((epoch_num, ckpt_file))
        except:
            continue
    
    if not epoch_numbers:
        return None, -1
    
    # Sort and get the latest
    epoch_numbers.sort(reverse=True)
    latest_epoch, latest_path = epoch_numbers[0]
    
    return latest_path, latest_epoch


# ---------------- TRAIN FUNCTION ---------------- #
def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    total = 0
    correct = 0
    model.train()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (images, labels) in loop:
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        mean_out = outputs.mean(1)

        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)
        
        ######################################################################
        # # ================= PHYSICAL SPIKE REGULARIZATION =================
        ######################################################################
        # Initialize these in case λ_spike = 0
        base_loss_value = loss.item()
        raw_spike_loss = 0.0
        scaled_spike_loss = 0.0

        if args.lambda_spike > 0:
            spike_counts = []
            for module in model.modules():
                if isinstance(module, LIFSpike):
                    spikes = getattr(module, "forward_spike_trace", None)
                    if spikes is not None:
                        count_per_t = spikes.flatten(2).sum(dim=2).mean(dim=0)
                        spike_counts.append(count_per_t)

            if spike_counts:
                total_spikes_per_t = torch.stack(spike_counts).sum(dim=0)
                T = total_spikes_per_t.numel()
                time_weights = torch.linspace(1.0 / T, 1.0, T, device=outputs.device)
                physical_spike_loss = (time_weights * total_spikes_per_t).sum()

                scaled_loss = args.lambda_spike * physical_spike_loss
                loss = loss + scaled_loss
                base_loss_value = loss.item() - scaled_loss.item()
                raw_spike_loss = physical_spike_loss.item()
                scaled_spike_loss = scaled_loss.item()
        ######################################################################

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
                f"TotalLoss={loss.item():.4f}"
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

    hooks = []
    for m in model.modules():
        if m.__class__.__name__ == "LIFSpike":
            hooks.append(m.register_forward_hook(spike_hook))

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        current_batch_size = inputs.size(0)
        total_samples += current_batch_size
        module_call_counter.clear()

        outputs = model(inputs)
        mean_out = outputs.mean(1)

        batch_spikes = (outputs > 0).sum().item()
        proxy_total_spikes += batch_spikes

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

    # ============ FIND CHECKPOINT TO RESUME FROM ============ #
    if args.checkpoint_path:
        checkpoint_file = args.checkpoint_path
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_file}")
    elif args.auto_resume:
        checkpoint_file, resume_epoch = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_file is None:
            print(f"No checkpoint found in {checkpoint_dir}")
            print("Starting fresh training from epoch 0")
            start_epoch = 0
            checkpoint = None
        else:
            print(f"Auto-detected latest checkpoint: {checkpoint_file}")
            print(f"Will resume from epoch {resume_epoch + 1}")
            start_epoch = resume_epoch + 1
    else:
        raise ValueError("Must specify --checkpoint_path or use --auto_resume")

    # ============ LOAD CHECKPOINT IF FOUND ============ #
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best test accuracy so far: {best_test_acc:.3f}")
    else:
        start_epoch = 0
        best_test_acc = 0.0
        checkpoint = None
        print("No checkpoint loaded - starting from scratch")

    # ============ SETUP LOGGING ============ #
    # TensorBoard - use purge_step to continue from checkpoint
    writer = SummaryWriter(log_dir=os.path.join("Logs", "runs", run_id), purge_step=start_epoch)

    # Logger - append mode
    log_path = os.path.join("Logs", f"{run_id}.log")
    logger = logging.getLogger('Train_sr_physical')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ============ DATA LOADING ============ #
    # train_dataset, val_dataset = data_loaders.build_cifar_qcfs(
    #     cutout=True, use_cifar10=True, download=True
    # )

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

    # ============ MODEL SETUP ============ #
    num_classes = 10 if args.use_cifar10 else 100
    model = vgg16(num_classes=num_classes)
    model.T = args.T
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params:,}")

    # Load model weights if checkpoint exists
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded from checkpoint")

    parallel_model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # ============ OPTIMIZER & SCHEDULER SETUP ============ #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    # Load optimizer and scheduler state if checkpoint exists
    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded from checkpoint")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded from checkpoint")

    best_train_acc = 0
    best_epoch = start_epoch

    # ============ LOG RESUME INFO ============ #
    logger.info("=" * 80)
    if checkpoint is not None:
        logger.info("RESUMING TRAINING FROM CHECKPOINT")
        logger.info(f"Checkpoint: {checkpoint_file}")
        logger.info(f"Resuming from epoch: {start_epoch}")
        logger.info(f"Best test acc so far: {best_test_acc:.3f}")
    else:
        logger.info("STARTING FRESH TRAINING")
    logger.info("=" * 80)
    logger.info("========== Training Configuration ==========")
    for arg, val in vars(args).items():
        logger.info(f"{arg}: {val}")
    logger.info("=" * 80)

    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every = 100

    # ============ DETERMINE TRAINING PHASE ============ #
    if start_epoch < args.epochs:
        # ------------------- TRAINING PHASE ------------------- #
        logger.info(f"Starting/Resuming TRAINING phase (epochs {start_epoch} to {args.epochs-1})")
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
            best_train_acc = max(best_train_acc, train_acc)

            scheduler.step()

            test_acc, proxy_avg_spikes, network_avg_spikes, lif_total_spikes, total_samples = test(parallel_model, test_loader, device)

            # Log thresholds per epoch
            for name, module in model.named_modules():
                if isinstance(module, LIFSpike):
                    thresh_val = float(module.thresh.item())
                    writer.add_scalar(f"Thresholds/{name}", thresh_val, epoch)
                    logger.info(f"Threshold[{name}] = {thresh_val:.6f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc
                }, os.path.join(checkpoint_dir, "best_model.pth"))

                logger.info(f"Saved BEST model at epoch {epoch+1} (acc={test_acc:.3f})")

            # Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc
                }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

                logger.info(f"Saved checkpoint at epoch {epoch+1}")

            peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3) \
                if torch.cuda.is_available() else 0.0

            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Epoch:[{epoch}/{args.epochs}] "
                f"LR={current_lr:.6f} "
                f"TrainLoss={train_loss:.5f} TrainAcc={train_acc:.3f} "
                f"TestAcc={test_acc:.3f} "
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

        logger.info("Training phase completed!")
        start_epoch = args.epochs  # Move to fine-tuning

    # ------------------- FINE-TUNING PHASE ------------------- #
    if start_epoch >= args.epochs:
        logger.info("=" * 80)
        logger.info("Starting/Resuming FINE-TUNING phase...")
        logger.info("=" * 80)

        # Update model T for fine-tuning
        model.T = args.fine_time
        
        # Create new optimizer for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr)

        fine_start = max(0, start_epoch - args.epochs)  # How many fine-tune epochs already done
        
        for fine_epoch in range(fine_start, args.fine_epochs):
            epoch = args.epochs + fine_epoch

            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
            best_train_acc = max(best_train_acc, train_acc)

            test_acc, proxy_avg_spikes, network_avg_spikes, lif_total_spikes, total_samples = test(parallel_model, test_loader, device)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_acc': best_test_acc
                }, os.path.join(checkpoint_dir, "best_model.pth"))

                logger.info(f"[FineTune] Saved BEST model at epoch {epoch+1} (acc={test_acc:.3f})")

            peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3) \
                if torch.cuda.is_available() else 0.0

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch:[{epoch}/{args.epochs + args.fine_epochs - 1}] "
                f"TrainLoss={train_loss:.5f} TrainAcc={train_acc:.3f} "
                f"TestAcc={test_acc:.3f} "
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

    # ============ WRAP-UP ============ #
    logger.info("=" * 80)
    logger.info("ALL TRAINING COMPLETED")
    logger.info(f"Final BestTrain {best_train_acc:.3f}")
    logger.info(f"Final BestTest {best_test_acc:.3f} at epoch {best_epoch}")
    logger.info("=" * 80)

    writer.close()

    # Save final model
    torch.save(model, os.path.join(checkpoint_dir, f"final_model_{run_id}.pth"))
    logger.info(f"Saved final model to {checkpoint_dir}/final_model_{run_id}.pth")