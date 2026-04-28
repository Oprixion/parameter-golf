export NUM_LAYERS=9
export MODEL_DIM=256
export NUM_HEADS=4
export NUM_KV_HEADS=2
export MLP_MULT=2
export VOCAB_SIZE=4096
export TRAIN_SEQ_LEN=512
export ITERATIONS=200
export WARMDOWN_ITERS=50
export WARMUP_STEPS=5
export TRAIN_BATCH_TOKENS=65536
export VAL_LOSS_EVERY=50
export TRAIN_LOG_EVERY=10
export MAX_WALLCLOCK_SECONDS=120
export RECUR_START_STEP=100
export SEED=42

python train_gpt.py

# --- Deploy to training server ---
# Uncomment to push latest scripts before running remotely:
# $SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519"
# $REMOTE  = "root@64.247.201.43:/workspace/PARAMETER-GOLF/"
# scp -P 12945 -i $env:USERPROFILE\.ssh\id_ed25519 train_gpt.py mini_run.ps1 root@64.247.201.43:/workspace/PARAMETER-GOLF/