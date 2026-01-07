#!/bin/bash

echo "=========================================="
echo "Training Status Debugging"
echo "=========================================="
echo

# 1. Process check
echo "[1] Process Status:"
PROC_COUNT=$(ps aux | grep -c "train_orgin.py")
if [ $PROC_COUNT -gt 1 ]; then
    echo "  ✅ Training process running"
    ps aux | grep train.py | grep -v grep
else
    echo "  ❌ No training process found"
fi
echo

# 2. GPU check
echo "[2] GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo

# 3. Latest output directory
echo "[3] Latest Output Directory:"
LATEST_OUTPUT=$(ls -dt output/conversation-LAVA-* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "  Directory: $LATEST_OUTPUT"
    echo "  Files:"
    ls -lht "$LATEST_OUTPUT" | head -10
    
    # Check for checkpoints
    CKPT_COUNT=$(find "$LATEST_OUTPUT" -name "checkpoint-*" -type d | wc -l)
    echo "  Checkpoints: $CKPT_COUNT"
else
    echo "  ❌ No output directory found"
fi
echo

# 4. WandB status
echo "[4] WandB Status:"
WANDB_DIR=$(ls -dt wandb/run-* 2>/dev/null | head -1)
if [ -n "$WANDB_DIR" ]; then
    echo "  Latest run: $WANDB_DIR"
    if [ -f "$WANDB_DIR/logs/debug.log" ]; then
        echo "  Last 5 lines:"
        tail -5 "$WANDB_DIR/logs/debug.log"
    fi
else
    echo "  ❌ No WandB run found"
fi
echo

# 5. Recent errors
echo "[5] Recent Errors (last 20 lines):"
if [ -f nohup.out ]; then
    tail -20 nohup.out | grep -i "error\|exception\|killed\|oom" || echo "  No errors found in nohup.out"
else
    echo "  No nohup.out found"
fi
echo

echo "=========================================="
echo "Debugging Complete"
echo "=========================================="
