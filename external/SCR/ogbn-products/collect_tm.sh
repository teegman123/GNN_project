# Collect and run CS
#!/usr/bin/env bash

# Activate env
source /home/temccrac/Downloads/anaconda3/bin/activate pyg112

# 1) Collect final-stage logits into a clean folder
python collect_and_cs_tm.py \
  --src-dir ./output/ogbn-products \
  --dst-dir ./output/ogbn-products_final \
  --method R_GAMLP_RLU \
  --final-stage 5

# 2) run your C&S / post-processing on each
for f in ./output/ogbn-products_final/*_R_GAMLP_RLU.pt; do
    base=$(basename "$f")               # e.g. "uuid_5_R_GAMLP_RLU.pt"
    stem="${base%.pt}"                  # e.g. "uuid_5_R_GAMLP_RLU"

    echo "Running C&S on $f"
    python post_processing_tm.py \
        --root . \
        --file_name "$stem" \
        --correction-alpha 0.4780826957236622 \
        --smoothing-alpha 0.40049734940262954
done
