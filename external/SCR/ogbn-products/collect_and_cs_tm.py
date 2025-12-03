#!/usr/bin/env python3

import argparse
import glob
import os
import re
import shutil

import torch


def find_final_stage_files(src_dir: str, method: str, final_stage: int):
    """
    Look for files of the form:
        <uuid>_<stage>_<method>.pt
    and keep ONLY those whose stage == final_stage.
    This will return one file per run (UUID) *if* every run has reached that stage.
    """
    pattern = re.compile(
        rf'(?P<uuid>[0-9a-fA-F]+)_(?P<stage>\d+)_({re.escape(method)})\.pt$'
    )

    matched_paths = []

    for path in glob.glob(os.path.join(src_dir, f"*_{method}.pt")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if m is None:
            continue
        stage = int(m.group("stage"))
        if stage == final_stage:
            matched_paths.append(path)

    return matched_paths


def run_cs_on_file(logit_path: str, out_dir: str, data_root: str = "./data"):
    """
    OPTIONAL: Load a logits tensor from <logit_path>, run Correct & Smooth on ogbn-products
    using OGB + PyG, and save final node labels as <stem>_cs.pt in out_dir.
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.nn import CorrectAndSmooth

    print(f"[C&S] Loading logits from {logit_path}")
    logits = torch.load(logit_path)  # [N, C]

    dataset = PygNodePropPredDataset("ogbn-products", root=data_root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    y = data.y.view(-1)
    edge_index = data.edge_index

    cs = CorrectAndSmooth(
        num_correction_layers=50,
        correction_alpha=0.5,
        num_smoothing_layers=50,
        smoothing_alpha=0.8,
        autoscale=True,
    )

    y_soft = logits.softmax(dim=-1)

    train_mask = torch.zeros(y_soft.size(0), dtype=torch.bool)
    train_mask[split_idx["train"]] = True

    print(f"[C&S] Running correction...")
    y_cs = cs.correct(y_soft, y, train_mask, edge_index)
    print(f"[C&S] Running smoothing...")
    y_cs = cs.smooth(y_cs, y, train_mask, edge_index)

    pred = y_cs.argmax(dim=-1)  # final label per node

    base = os.path.basename(logit_path)
    stem, _ = os.path.splitext(base)
    out_path = os.path.join(out_dir, stem + "_cs.pt")
    torch.save(pred, out_path)
    print(f"[C&S] Saved smoothed predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        default="./output/ogbn-products",
        help="Folder where <uuid>_<stage>_<method>.pt files live",
    )
    parser.add_argument(
        "--dst-dir",
        default="./output/ogbn-products_final",
        help="Folder to copy final-stage logits (and optional C&S outputs) into",
    )
    parser.add_argument(
        "--method",
        default="R_GAMLP_RLU",
        help="Method tag used in filenames, e.g. R_GAMLP_RLU",
    )
    parser.add_argument(
        "--final-stage",
        type=int,
        default=5,  # your stages are 0..5, so final is 5
        help="Stage index to treat as 'final' (e.g. 5 if stages are 0..5)",
    )
    parser.add_argument(
        "--run-cs",
        action="store_true",
        help="If set, run Correct&Smooth on each copied logits file.",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root dir for ogbn-products when using PyG + OGB",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # 1) find all final-stage files
    final_paths = find_final_stage_files(args.src_dir, args.method, args.final_stage)
    print(f"Found {len(final_paths)} final-stage (stage={args.final_stage}) files")

    # 2) copy them and optionally run C&S
    for src in final_paths:
        dst = os.path.join(args.dst_dir, os.path.basename(src))
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)

        if args.run_cs:
            run_cs_on_file(dst, args.dst_dir, data_root=args.data_root)


if __name__ == "__main__":
    main()