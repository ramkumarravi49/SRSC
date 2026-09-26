#Train_sr.py
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
import data_loaders_qcfs as data_loaders     # <<< FULL SWAP TO QCFS LOADER

from functions import TET_loss, seed_all, get_logger
from datetime import datetime


# ---------------- GPU Monitoring ---------------- #
def get_gpu_utilization():
    mem_bytes = torch.cuda.memory_allocated()
    return torch.cuda.max_memory_allocated() / (1024**3), mem_bytes / (1024**3)
# ------------------------------------------------- #

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- ARGUMENTS -------------------- #
parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, dest='lr')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--T', default=2, type=int)
parser.add_argument('--means', default=1.0, type=float)
parser.add_argument('--TET', default=True, type=bool)
parser.add_argument('--lamb', default=1e-3, type=float)
parser.add_argument('--lambda_spike', default=0.0, type=float, help='Temporal spike regularization strength' )

# Fine tuning
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--fine_time', default=4, type=int)
parser.add_argument('--fine_lr', default=0.0001, type=float)

args = parser.parse_args()
# ----------------------------------------------------- #


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

        # if args.TET:
        #     loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        # else:
        #     loss = criterion(mean_out, labels)

        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)

        # ================= TEMPORAL SPIKE REGULARIZATION =================
        if args.lambda_spike > 0:
            # outputs: [B, T, ...]
            # Differentiable proxy for spiking activity
            spike_proxy = torch.relu(outputs)

            # Sum over neurons, average over batch → [T]
            proxy_per_t = spike_proxy.flatten(2).sum(dim=2).mean(dim=0)

            T = proxy_per_t.numel()

            # Linearly increasing penalty (late activity costs more)
            time_weights = torch.linspace(
                1.0 / T, 1.0, T, device=outputs.device
            )

            spike_loss = (time_weights * proxy_per_t).sum()
            loss = loss + args.lambda_spike * spike_loss
        # =================================================================



        running_loss += loss.item()
        loss.mean().backward()
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
    total_spikes = 0
    num_inferences = 0

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)

        batch_spikes = (outputs > 0).sum().item()
        total_spikes += batch_spikes
        num_inferences += inputs.size(0)

        _, predicted = mean_out.cpu().max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100 * correct / total, total_spikes / num_inferences


# ===================== MAIN ===================== #
if __name__ == '__main__':
    seed_all(args.seed)

    # <<< MANUALLY EDIT THIS FOR EVERY RUN
    run_id = "LT_SR_5e3_resnet19_T_8"

    # TensorBoard + Logs
    writer = SummaryWriter(log_dir=os.path.join("Logs", "runs", run_id))

    # DATA LOADING (QCFS ONLY)
    train_dataset, val_dataset = data_loaders.build_cifar_qcfs( 
        cutout=True, use_cifar10=True, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # MODEL
    model = resnet19(num_classes=10)
    model.T = args.T
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params:,}")

    parallel_model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0

    # LOGGER
    log_path = os.path.join("Logs", f"{run_id}.log")
    logger = get_logger(log_path)

    logger.info("========== Training Configuration ==========")
    for arg, val in vars(args).items():
        logger.info(f"{arg}: {val}")
    logger.info("============================================")
    logger.info("Start training!")

    # Save text of args to TB
    writer.add_text("Hyperparameters", "\n".join(f"{a}: {v}" for a, v in vars(args).items()))

    # CHECKPOINT DIRECTORY
    checkpoint_dir = os.path.join("CHECKPOINTS", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_every = 250

    # ------------------- TRAINING LOOP ------------------- #
    for epoch in range(args.epochs):
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        best_train_acc = max(best_train_acc, train_acc)

        scheduler.step()

        test_acc, avg_spikes = test(parallel_model, test_loader, device)
        # ===== Log thresholds per epoch =====
        for name, module in model.named_modules():
            if isinstance(module, LIFSpike):
                thresh_val = float(module.thresh.item())

                # Log to TensorBoard
                writer.add_scalar(f"Thresholds/{name}", thresh_val, epoch)

                # Log to logfile
                logger.info(f"Threshold[{name}] = {thresh_val:.6f}")
        # ====================================


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
            f"TestAcc={test_acc:.3f} AvgSpikes={avg_spikes:.3f} "
            f"GPU={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s"
        )

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        writer.add_scalar("Spikes/Avg_per_inf", avg_spikes, epoch)
        writer.add_scalar("GPU/PeakGB", peak_mem_gb, epoch)
        writer.add_scalar("Time/EpochSec", epoch_time, epoch)
        writer.add_scalar("LR", current_lr, epoch)

    # ------------------- FINE-TUNING ------------------- #
    logger.info("Starting fine-tuning…")

    model.T = args.fine_time
    optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr)

    for fine_epoch in range(args.fine_epochs):
        epoch = args.epochs + fine_epoch

        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        best_train_acc = max(best_train_acc, train_acc)

        test_acc, avg_spikes = test(parallel_model, test_loader, device)
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

            logger.info(f"[FineTune] Saved BEST model at epoch {epoch+1} (acc={test_acc:.3f})")

        peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3) \
            if torch.cuda.is_available() else 0.0

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch:[{epoch}/{args.epochs + args.fine_epochs}] "
            f"TrainLoss={train_loss:.5f} TrainAcc={train_acc:.3f} "
            f"TestAcc={test_acc:.3f} AvgSpikes={avg_spikes:.3f} "
            f"GPU={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s"
        )

        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        writer.add_scalar("Spikes/Avg_per_inf", avg_spikes, epoch)
        writer.add_scalar("GPU/PeakGB", peak_mem_gb, epoch)
        writer.add_scalar("Time/EpochSec", epoch_time, epoch)

    # WRAP-UP
    writer.add_text("Final Results",
                    f"Best Train Acc: {best_train_acc:.3f}\n"
                    f"Best Test Acc: {best_test_acc:.3f} at epoch {best_epoch}")

    writer.add_text("Model Info",
                    f"Total trainable parameters: {total_trainable_params:,}")

    logger.info(f"Final BestTrain {best_train_acc:.3f}")
    logger.info(f"Final BestTest {best_test_acc:.3f} at epoch {best_epoch}")

    writer.close()

    torch.save(model, os.path.join(checkpoint_dir, f"final_model_{run_id}.pth"))
