#!/usr/bin/env bash
# launch_orchestrator.sh
# ──────────────────────────────────────────────────────────────────────────────
# Creates tmux session 'snn_c100', then starts the orchestrator in window 0.
# All subsequent training runs are launched by the orchestrator as new windows
# inside the same session.
#
# Usage:
#   bash launch_orchestrator.sh
#
# To watch what's happening after detaching:
#   tmux attach -t snn_c100
#
# To jump to a specific run window:
#   tmux select-window -t snn_c100:<window_name>
#   e.g.  tmux select-window -t snn_c100:TEST_resnet19_cifar100_1e-6_sr_phy
#
# Terminal logs for every window (including orchestrator) are written to:
#   terminal_logs/<run_name>_terminal.log
#   terminal_logs/orchestrator_terminal.log
# ──────────────────────────────────────────────────────────────────────────────

SESSION="snn_c100"
LOGDIR="terminal_logs"
ORCH_LOG="$(pwd)/${LOGDIR}/orchestrator_terminal.log"

mkdir -p "$LOGDIR"

# ── Check tmux is available ──────────────────────────────────────────────────
if ! command -v tmux &> /dev/null; then
    echo "[ERROR] tmux is not installed or not in PATH."
    exit 1
fi

# ── Check nvidia-smi is available ───────────────────────────────────────────
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. Cannot query GPU memory."
    exit 1
fi

# ── If session already exists, warn and exit ─────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[WARN] tmux session '$SESSION' already exists."
    echo "       To kill it:  tmux kill-session -t $SESSION"
    echo "       To attach:   tmux attach -t $SESSION"
    exit 1
fi

# ── Create session and launch orchestrator in window 0 ───────────────────────
echo "[INFO] Creating tmux session '$SESSION'..."
echo "[INFO] Poll interval: 300s  |  Min free GPU RAM: 10GB"
echo "[INFO] Orchestrator log: $ORCH_LOG"
echo ""

tmux new-session -d -s "$SESSION" -n "orchestrator" \
    "python run_cifar100_queue.py 2>&1 | tee ${ORCH_LOG} ; \
     echo '[ORCHESTRATOR DONE]' | tee -a ${ORCH_LOG} ; \
     read -p 'Press Enter to close this window...'"

echo "[OK]  Session '$SESSION' created."
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Orchestrator is running in the background."
echo ""
echo "  Attach to watch live:    tmux attach -t $SESSION"
echo "  Detach (keep running):   Ctrl+B then D"
echo ""
echo "  Jump to a run window:"
echo "    tmux select-window -t $SESSION:TEST_resnet19_cifar100_1e-6_sr_phy"
echo "    tmux select-window -t $SESSION:SRSC_resnet19_cifar100_sr_phy_1e-8__5e-7_acc_gate_65"
echo ""
echo "  All terminal logs saved to: ./$LOGDIR/"
echo "    orchestrator_terminal.log"
echo "    <run_name>_terminal.log  (one per run, created when run launches)"
echo ""
echo "  Kill everything:         tmux kill-session -t $SESSION"
echo "══════════════════════════════════════════════════════════════"