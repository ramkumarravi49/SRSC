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
#import pynvml
from tqdm import tqdm

from models.VGG_models import *
from models.resnet_models import *
import data_loaders
from functions import TET_loss, seed_all, get_logger
from datetime import datetime


# ---------------- GPU Monitoring ---------------- #
def get_gpu_utilization():
    mem_bytes = torch.cuda.memory_allocated()
    return torch.cuda.max_memory_allocated() / (1024**3), mem_bytes / (1024**3)  # convert to GB

# ------------------------------------------------- #

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, dest='lr')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('-T', default=2, type=int)
parser.add_argument('--means', default=1.0, type=float)
parser.add_argument('--TET', default=True, type=bool)
parser.add_argument('--lamb', default=1e-3, type=float)

# Finetuning
parser.add_argument('--fine_epochs', default=50, type=int)
parser.add_argument('--fine_time', default=4, type=int)
parser.add_argument('--fine_lr', default=0.0001, type=float)
args = parser.parse_args()

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

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        total += labels.size(0)
        _, predicted = mean_out.cpu().max(1)
        correct += predicted.eq(labels.cpu()).sum().item()

        # 👇 Update tqdm bar
        loop.set_description(f"Epoch [{epoch+1}]")
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    total_spikes = 0
    num_inferences = 0

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)  # [B, T, num_classes]
        mean_out = outputs.mean(1)

        # Spike counting
        batch_spikes = (outputs > 0).sum().item()
        total_spikes += batch_spikes
        num_inferences += inputs.size(0)

        _, predicted = mean_out.cpu().max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    final_acc = 100 * correct / total
    avg_spikes_per_inf = total_spikes / num_inferences
    return final_acc, avg_spikes_per_inf


if __name__ == '__main__':
    seed_all(args.seed)

    # Unique run folder for TensorBoard
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"./runs/experiment_{run_id}")

    train_dataset, val_dataset = data_loaders.build_cifar(cutout=True, use_cifar10=True, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = resnet19(num_classes=10)
   # model = resnet18(num_classes=100)
    model.T = args.T
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params:,}")
    
    #parallel_model = torch.nn.DataParallel(model).to(device)
    parallel_model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_train_acc = 0
    best_test_acc = 0
    best_epoch = 0

    logger = get_logger('TET_18_Sep_Tune_50e_4t.log')
    logger.info("========== Training Configuration ==========")
    for arg, val in vars(args).items():
        logger.info(f"{arg}: {val}")
    logger.info("============================================")

    logger.info('start training!')

    # Log all hyperparameters into TensorBoard
    args_text = "\n".join([f"{arg}: {val}" for arg, val in vars(args).items()])
    writer.add_text("Hyperparameters", args_text)

    save_every = 10   # Save every N epochs
    checkpoint_dir = "./checkpoints"
    #os.makedirs(checkpoint_dir, exist_ok=True)


    # -------------------- Load Best Model -------------------- #
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best pretrained model from epoch {checkpoint['epoch']+1} "
                f"(Test acc={checkpoint['best_test_acc']:.3f})")

    # -------------------- Fine-tuning -------------------- #
    logger.info("Starting fine-tuning...")
    model.T = args.fine_time
    optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr)  # fresh optimizer

    for fine_epoch in range(args.fine_epochs):
        epoch = fine_epoch  # start numbering fine-tune epochs from 0

        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats()

        train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        best_train_acc = max(best_train_acc, train_acc)

        test_acc, avg_spikes = test(parallel_model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            # Save fine-tuned best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_acc': best_test_acc
            }, os.path.join(checkpoint_dir, "50e_2t_best_finetuned_model.pth"))
            logger.info(f"Saved new fine-tuned best model at fine-epoch {epoch+1} (Test acc={test_acc:.3f})")

        peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
        epoch_time = time.time() - epoch_start

        logger.info(f"Fine-tune Epoch:[{epoch}/{args.fine_epochs}] "
                    f"Train loss={train_loss:.5f} Train acc={train_acc:.3f} "
                    f"Test acc={test_acc:.3f} Avg Spikes={avg_spikes:.3f} "
                    f"GPU Peak Mem={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s")

        writer.add_scalar("FineTune/Accuracy/Train", train_acc, epoch)
        writer.add_scalar("FineTune/Accuracy/Test", test_acc, epoch)
        writer.add_scalar("FineTune/Spikes/Avg_per_inference", avg_spikes, epoch)
        writer.add_scalar("FineTune/GPU Memory/Peak (GB)", peak_mem_gb, epoch)
        writer.add_scalar("FineTune/Time/Epoch_time_sec", epoch_time, epoch)


    

    # for epoch in range(args.epochs):
    #     epoch_start = time.time()

    #     # Reset GPU memory stats before measuring for this epoch
    #     torch.cuda.reset_peak_memory_stats()

    #     # Training
    #     train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
    #     best_train_acc = max(best_train_acc, train_acc)

    #     scheduler.step()

    #     # Testing
    #     test_acc, avg_spikes = test(parallel_model, test_loader, device)
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         best_epoch = epoch + 1
    #         # Save best model checkpoint
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'best_test_acc': best_test_acc
    #         }, os.path.join(checkpoint_dir, "best_model.pth"))
    #         logger.info(f"Saved new best model at epoch {epoch+1} (Test acc={test_acc:.3f})")

    #     # Save every N epochs
    #     if (epoch + 1) % save_every == 0:
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'best_test_acc': best_test_acc
    #         }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    #         logger.info(f"Saved checkpoint at epoch {epoch+1}")

    #     # Get peak GPU memory usage for this epoch in GB
    #     peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
    #     epoch_time = time.time() - epoch_start
    #     current_lr = optimizer.param_groups[0]['lr']


    #     logger.info(f"Epoch:[{epoch}/{args.epochs}] "
    #                 f"LR={current_lr:.6f} "
    #                 f"Train loss={train_loss:.5f} Train acc={train_acc:.3f} "
    #                 f"Test acc={test_acc:.3f} Avg Spikes={avg_spikes:.3f} "
    #                 f"GPU Peak Mem={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s")

    #     # TensorBoard logging
    #     writer.add_scalar("Accuracy/Train", train_acc, epoch)
    #     writer.add_scalar("Accuracy/Test", test_acc, epoch)
    #     writer.add_scalar("Spikes/Avg_per_inference", avg_spikes, epoch)
    #     writer.add_scalar("GPU Memory/Peak (GB)", peak_mem_gb, epoch)
    #     writer.add_scalar("Time/Epoch_time_sec", epoch_time, epoch)
    #     writer.add_scalar("LR", current_lr, epoch) 


    # # -------------------- Fine-tuning -------------------- #
    # logger.info("Starting fine-tuning...")
    # model.T = args.fine_time
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_lr)  # No scheduler

    # for fine_epoch in range(args.fine_epochs):
    #     epoch = args.epochs + fine_epoch  # continue numbering

    #     epoch_start = time.time()
    #     torch.cuda.reset_peak_memory_stats()

    #     train_loss, train_acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
    #     best_train_acc = max(best_train_acc, train_acc)

    #     test_acc, avg_spikes = test(parallel_model, test_loader, device)
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         best_epoch = epoch + 1

    #     peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
    #     epoch_time = time.time() - epoch_start

    #     logger.info(f"Epoch:[{epoch}/{args.epochs + args.fine_epochs}] Train loss={train_loss:.5f} Train acc={train_acc:.3f} "
    #                 f"Test acc={test_acc:.3f} Avg Spikes={avg_spikes:.3f} "
    #                 f"GPU Peak Mem={peak_mem_gb:.2f}GB Time={epoch_time:.2f}s")

    #     writer.add_scalar("Accuracy/Train", train_acc, epoch)
    #     writer.add_scalar("Accuracy/Test", test_acc, epoch)
    #     writer.add_scalar("Spikes/Avg_per_inference", avg_spikes, epoch)
    #     writer.add_scalar("GPU Memory/Peak (GB)", peak_mem_gb, epoch)
    #     writer.add_scalar("Time/Epoch_time_sec", epoch_time, epoch)

    # writer.add_text("Final Results", f"Best Train Accuracy: {best_train_acc:.3f}\nBest Test Accuracy: {best_test_acc:.3f}")
    # writer.add_text("Model Info", f"Total trainable parameters: {total_trainable_params:,}")

    # logger.info(f"Final Best Train Acc: {best_train_acc:.3f}")
    # logger.info(f"Final Best Test Acc: {best_test_acc:.3f} at epoch {best_epoch}")

    # writer.close()
    # torch.save(model, "cifar10_TET_resnet19_17_Sep.pth")


