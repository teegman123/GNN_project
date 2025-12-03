#!/bin/bash

# ---------- ENV SETUP (edit if needed) ----------
source ~/Downloads/anaconda3/etc/profile.d/conda.sh
conda activate pyg112   # <- change to your env if different

DATASET="ogbn-products"
METHOD="R_GAMLP_RLU"
PRED_DIR="./output/${DATASET}"

# ---------- 1. PREPROCESS ----------
#echo "[SCR TEST] Running preprocessing..."
#python -u pre_processing.py --num_hops 5 --dataset "${DATASET}"

# ---------- 2. TRAIN (5 SHORT STAGES) ----------
echo "[SCR TEST] Running multi-stage training (short test)..."
python -u main_tm.py \
  --use-rlu \
  --method "${METHOD}" \
  --stages 20 2 \
  --train-num-epochs 0 0 \
  --threshold 0.6 \
  --input-drop 0.2 \
  --att-drop 0.5 \
  --label-drop 0 \
  --pre-process \
  --residual \
  --dataset "${DATASET}" \
  --num-runs 1 \
  --eval 1 \
  --act leaky_relu \
  --batch_size 50000 \
  --patience 10 \
  --n-layers-1 4 \
  --n-layers-2 4 \
  --gama 0.1 \
  --consis \
  --tem 0.5 \
  --lam 0.1 \
  --hidden 512 \
  --gpu 1 \
  --ema

# At this point, if your model-saving code is in run(), you should have something like:
#   ./output/ogbn-products/model_ogbn-products_R_GAMLP_RLU_...pth

# ---------- 3. FIND LAST-STAGE PREDICTION FILE ----------
# 5 stages => stage indices 0..4, so LAST_STAGE=4
LAST_STAGE=1

echo "[SCR TEST] Searching for last-stage prediction file (stage ${LAST_STAGE})..."
FILE=$(ls -t ${PRED_DIR}/*_${LAST_STAGE}_${METHOD}.pt | head -n 1)

if [ -z "$FILE" ]; then
    echo "[SCR TEST][ERROR] No SCR prediction file found for stage ${LAST_STAGE} in ${PRED_DIR}"
    exit 1
fi

BASENAME=$(basename "$FILE" .pt)
echo "[SCR TEST] Using SCR prediction file: $FILE"
echo "[SCR TEST] Basename for post-processing: $BASENAME"

# ---------- 4. RUN POST-PROCESSING (C&S) ----------
echo "[SCR TEST] Running post_processing (C&S)..."
python -u post_processing_tm.py \
  --dataset "${DATASET}" \
  --file_name "${BASENAME}" \
  --correction-alpha 0.4780826957236622 \
  --smoothing-alpha 0.40049734940262954

# ---------- 5. (OPTIONAL) SAVE POST-PROCESSED PREDICTIONS ----------
# This part assumes you've added a small change inside post_processing.py to save the
# final corrected+smoothed logits, e.g. near the end of that script:
#
#   torch.save(final_preds, f"./output/{args.dataset}/{args.file_name}_cs.pt")
#
# After that change, you should see a new file like:
#   ./output/ogbn-products/<uuid>_4_R_GAMLP_RLU_cs.pt
#
echo "[SCR TEST] If you patched post_processing.py to save C&S outputs, check:"
echo "  ls ${PRED_DIR}/${BASENAME}_cs.pt"
echo "[SCR TEST] Also check your model checkpoint(s) in ${PRED_DIR}."