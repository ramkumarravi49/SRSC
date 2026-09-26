#!/usr/bin/env python3
"""
run_cifar100_queue.py
---------------------
Orchestrator for 6 CIFAR-100 training runs across 3 GPUs.

Rules:
  - Max 1 run per GPU at a time (professor's constraint)
  - Only launch on a GPU with >= 10 GB free memory
  - Polls every 60 seconds
  - Each run gets its own named tmux window inside session 'snn_c100'
  - All terminal output (stdout + stderr) is tee'd to terminal_logs/<run_name>_terminal.log
  - Orchestrator itself logs to terminal_logs/orchestrator_terminal.log (handled by launch script)
"""

import subprocess
import time
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TMUX_SESSION    = "snn_c100"
POLL_INTERVAL   = 300         # seconds between GPU polls
MIN_FREE_GB     = 10.0        # minimum free GPU memory to launch a run
NUM_GPUS        = 3           # GPU indices 0, 1, 2
TERMINAL_LOGDIR = "terminal_logs"

os.makedirs(TERMINAL_LOGDIR, exist_ok=True)

# ─────────────────────────────────────────────
# RUN DEFINITIONS
# ─────────────────────────────────────────────
# Each entry is a dict:
#   name    : run_name passed to --run_name, also used as tmux window name
#   script  : python training script
#   extra   : list of extra CLI args beyond the shared defaults
#
# Shared defaults applied to ALL runs:
#   --use_cifar10 false
#   --workers 4
#   --gpu <assigned at launch time>

SHARED_ARGS = "--use_cifar10 false --workers 4"

RUNS = [
    # ── STATIC (3.2) ──────────────────────────────────────────────────────────
    {
        "name":   "TEST_resnet19_cifar100_1e-6_sr_phy",
        "script": "3.2_Train_sr_physical_new.py",
        "extra":  "--lambda_spike 1e-6",
    },
    {
        "name":   "TEST_resnet19_cifar100_3e-7_sr_phy",
        "script": "3.2_Train_sr_physical_new.py",
        "extra":  "--lambda_spike 3e-7",
    },
    {
        "name":   "TEST_resnet19_cifar100_5e-7_sr_phy",
        "script": "3.2_Train_sr_physical_new.py",
        "extra":  "--lambda_spike 5e-7",
    },
    # ── DYNAMIC / SRSC (3.3) ─────────────────────────────────────────────────
    {
        "name":   "SRSC_resnet19_cifar100_sr_phy_1e-8__3e-7_acc_gate_65",
        "script": "3.3_Train_sr_SpikeCurriculum.py",
        "extra":  "--lambda_spike_base 1e-8 --lambda_spike_ceil 3e-7 --acc_gate 65",
    },
    {
        "name":   "SRSC_resnet19_cifar100_sr_phy_1e-8__5e-7_acc_gate_65",
        "script": "3.3_Train_sr_SpikeCurriculum.py",
        "extra":  "--lambda_spike_base 1e-8 --lambda_spike_ceil 5e-7 --acc_gate 65",
    },
    {
        "name":   "SRSC_resnet19_cifar100_sr_phy_1e-8__7e-7_acc_gate_65",
        "script": "3.3_Train_sr_SpikeCurriculum.py",
        "extra":  "--lambda_spike_base 1e-8 --lambda_spike_ceil 7e-7 --acc_gate 65",
    },
]

# ─────────────────────────────────────────────
# GPU UTILITIES
# ─────────────────────────────────────────────

def get_free_memory_gb(gpu_id: int) -> float:
    """Query free memory (GiB) for a single GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10
        )
        free_mib = float(result.stdout.strip().split("\n")[0])
        return free_mib / 1024.0  # MiB → GiB
    except Exception as e:
        print(f"[WARN] Could not query GPU {gpu_id}: {e}")
        return 0.0


def gpu_status_string() -> str:
    """Return a one-line summary of free memory on all GPUs."""
    parts = []
    for g in range(NUM_GPUS):
        free = get_free_memory_gb(g)
        parts.append(f"GPU{g}={free:.1f}GB")
    return "  ".join(parts)

# ─────────────────────────────────────────────
# TMUX UTILITIES
# ─────────────────────────────────────────────

def tmux_window_exists(window_name: str) -> bool:
    """Return True if a window with this name exists in TMUX_SESSION."""
    result = subprocess.run(
        ["tmux", "list-windows", "-t", TMUX_SESSION, "-F", "#{window_name}"],
        capture_output=True, text=True
    )
    return window_name in result.stdout.split("\n")


def tmux_window_alive(window_name: str) -> bool:
    """
    Return True if the window still exists (process still running).
    We use this to detect when a run has finished/crashed.
    """
    return tmux_window_exists(window_name)


def launch_run_in_tmux(run: dict, gpu_id: int):
    """
    Open a new tmux window named run['name'] and start the training command.
    stdout+stderr are tee'd to terminal_logs/<run_name>_terminal.log
    """
    name    = run["name"]
    script  = run["script"]
    extra   = run["extra"]

    log_path = os.path.abspath(os.path.join(TERMINAL_LOGDIR, f"{name}_terminal.log"))

    cmd = (
        f"source /data/cs24m037/ann2snn_QCFS/qcfs/bin/activate && "
        f"cd /data/cs24m037/TET_LT && "
        f"python {script} "
        f"--run_name {name} "
        f"--gpu {gpu_id} "
        f"{SHARED_ARGS} "
        f"{extra} "
        f"2>&1 | tee {log_path} ; "
        f"echo '[DONE] {name} finished at $(date)' | tee -a {log_path}"
    )

    # Create a new window in the existing session
    subprocess.run([
        "tmux", "new-window",
        "-t", TMUX_SESSION,
        "-n", name,          # window name = run name
        "--",
        "bash", "-c", cmd
    ])

    print(f"  → Launched '{name}' on GPU {gpu_id}  |  log: {log_path}")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    pending  = list(RUNS)          # runs not yet started
    running  = {}                  # gpu_id → run dict
    done     = []                  # completed run names

    log("=" * 70)
    log(f"CIFAR-100 Orchestrator started.  {len(pending)} runs queued.")
    log(f"Session: {TMUX_SESSION}  |  Poll interval: {POLL_INTERVAL}s  |  Min free: {MIN_FREE_GB}GB")
    log("=" * 70)

    while pending or running:

        # ── 1. Check if any running jobs have finished ──────────────────────
        finished_gpus = []
        for gpu_id, run in running.items():
            if not tmux_window_alive(run["name"]):
                log(f"[DONE] '{run['name']}' on GPU {gpu_id} has finished.")
                done.append(run["name"])
                finished_gpus.append(gpu_id)
        for g in finished_gpus:
            del running[g]

        # ── 2. Try to fill free GPUs from the pending queue ─────────────────
        if pending:
            for gpu_id in range(NUM_GPUS):
                if not pending:
                    break
                if gpu_id in running:
                    continue   # already occupied

                free_gb = get_free_memory_gb(gpu_id)
                if free_gb >= MIN_FREE_GB:
                    run = pending.pop(0)
                    running[gpu_id] = run
                    log(f"[LAUNCH] GPU {gpu_id} has {free_gb:.1f}GB free → starting '{run['name']}'")
                    launch_run_in_tmux(run, gpu_id)
                else:
                    log(f"[SKIP]   GPU {gpu_id} only {free_gb:.1f}GB free (need {MIN_FREE_GB}GB) — waiting")

        # ── 3. Status summary ───────────────────────────────────────────────
        log(
            f"Status | Pending: {len(pending)}  "
            f"Running: {len(running)} {list(running.keys())}  "
            f"Done: {len(done)}  |  {gpu_status_string()}"
        )

        if pending or running:
            log(f"Sleeping {POLL_INTERVAL}s ...\n")
            time.sleep(POLL_INTERVAL)

    # ── All done ─────────────────────────────────────────────────────────────
    log("=" * 70)
    log("ALL 6 RUNS COMPLETED.")
    log("Finished runs:")
    for d in done:
        log(f"  ✓ {d}")
    log("=" * 70)


if __name__ == "__main__":
    main()