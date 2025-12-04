# Graph Neural Network Ensemble for OGBn-Products

This repository contains an ensemble approach for node classification on the **ogbn-products** dataset, combining predictions from 6 different graph-based models using 9 different voting strategies.

## Project Overview

This project implements and evaluates an ensemble of graph neural network models for node classification on the ogbn-products dataset from the Open Graph Benchmark (OGB). The ensemble combines predictions from 6 diverse models and tests 9 different voting/aggregation strategies to achieve improved performance over individual models.

### Dataset
- **Dataset**: ogbn-products
- **Task**: Node classification
- **Test Set Size**: 2,213,091 nodes
- **Total Nodes**: 2,449,029 nodes
- **Number of Classes**: 47

## Ensemble Models

The ensemble consists of 6 models, each from different architectures and training approaches:

### 1. Plain Model (CorrectAndSmooth)
- **Location**: `external/CorrectAndSmooth/`
- **Architecture**: Plain linear model with Correct and Smooth (C&S) post-processing
- **Individual Accuracy**: 82.54%
- **Model Name**: `plain_run1`
- **Description**: A simple linear model trained on node features, enhanced with the C&S technique for label propagation and smoothing.

### 2. Linear Model (CorrectAndSmooth)
- **Location**: `external/CorrectAndSmooth/`
- **Architecture**: Linear model with embeddings + C&S
- **Individual Accuracy**: 82.97%
- **Model Name**: `linear_run3`
- **Description**: Linear classifier using spectral and diffusion embeddings, with C&S post-processing.

### 3. MLP Model (CorrectAndSmooth)
- **Location**: `external/CorrectAndSmooth/`
- **Architecture**: Multi-layer perceptron with embeddings + C&S
- **Individual Accuracy**: 84.22%
- **Model Name**: `mlp_run2`
- **Description**: MLP with hidden channels (200) using spectral and diffusion embeddings, enhanced with C&S.

### 4. GCN Model (tunedGNN)
- **Location**: `external/GCN/tunedGNN/`
- **Architecture**: Graph Convolutional Network (GCN)
- **Individual Accuracy**: 80.41%
- **Model Name**: `gcn_run0`
- **Description**: Classic GCN implementation from the "Classic GNNs are Strong Baselines" paper, optimized for large graphs.

### 5. GAMLP+CS Model (SCR)
- **Location**: `external/SCR/ogbn-products/`
- **Architecture**: Graph Attention Multi-Layer Perceptron with Correct and Smooth
- **Individual Accuracy**: 85.19%
- **Model Name**: `gamlp_cs_09883da0`
- **Description**: GAMLP model trained with Consistency Regularization (SCR) and C&S post-processing.

### 6. GAMLP Model (SCR)
- **Location**: `external/SCR/ogbn-products/`
- **Architecture**: Graph Attention Multi-Layer Perceptron
- **Individual Accuracy**: 85.08%
- **Model Name**: `gamlp_09883da0`
- **Description**: GAMLP model trained with Consistency Regularization (SCR) without C&S.

### Model Performance Summary
- **Average Individual Accuracy**: 83.40%
- **Best Individual Accuracy**: 85.19% (GAMLP+CS)
- **Worst Individual Accuracy**: 80.41% (GCN)

## Voting Strategies

The project evaluates 9 different voting/aggregation strategies to combine predictions from the 6 models:

### 1. Basic Voting
- **Method**: Each model gets 1 vote for their top prediction
- **Ensemble Accuracy**: 84.42%
- **Description**: Simple majority voting where each model's top prediction receives equal weight.

### 2. Ranked Choice Voting
- **Method**: Each model gets 3 votes: 3 points for 1st choice, 2 for 2nd, 1 for 3rd
- **Ensemble Accuracy**: 84.89%
- **Description**: Weighted voting system that considers top-3 predictions with decreasing weights.

### 3. Probability Sum Voting
- **Method**: Sum all output probabilities across models, highest sum wins
- **Ensemble Accuracy**: 85.24%
- **Description**: Aggregates probability distributions from all models by summing them.

### 4. Proportional Top-5 Voting
- **Method**: Each model gets 5 points, split proportionally among top 5 probabilities
- **Ensemble Accuracy**: 85.38%
- **Description**: Allocates 5 points per model proportionally based on top-5 class probabilities.

### 5. Performance-Weighted Probability Sum
- **Method**: Sum probabilities weighted by each model's individual accuracy
- **Ensemble Accuracy**: 85.24%
- **Description**: Weights each model's probabilities by its validation/test accuracy.

### 6. Confidence-Weighted Voting
- **Method**: Weight each model's probabilities by its confidence (max probability) for that node
- **Ensemble Accuracy**: 85.12%
- **Description**: Dynamically weights models based on their confidence for each specific prediction.

### 7. Geometric Mean Voting
- **Method**: Product of probabilities (more robust to outliers)
- **Ensemble Accuracy**: 85.39%
- **Description**: Uses geometric mean instead of arithmetic mean, reducing the influence of outlier predictions.

### 8. Exponential Weighted Voting
- **Method**: Raise probabilities to a power before summing
- **Ensemble Accuracy**: 85.08%
- **Description**: Applies exponential transformation to probabilities to emphasize high-confidence predictions.

### 9. Hybrid Weighted Voting
- **Method**: Combine performance weight + confidence weight
- **Ensemble Accuracy**: 85.12%
- **Description**: Combines both model-level performance weights and node-level confidence weights.

### Best Voting Strategy
- **Best Method**: **Geometric Mean Voting** (85.39%)
- **Improvement over average individual**: +1.99% (2.38% relative improvement)
- **Improvement over best individual**: +0.20% (0.23% relative improvement)

## Repository Structure

```
GNN_project/
├── README.md                          # This file
├── external/
│   ├── CorrectAndSmooth/              # Plain, Linear, and MLP models with C&S
│   │   ├── gen_models.py              # Model generation script
│   │   ├── run_experiments.py         # Experiment runner
│   │   ├── product_job.sh             # Job script for products dataset
│   │   ├── plain_product_job.sh       # Plain model job script
│   │   ├── linear_product_job.sh     # Linear model job script
│   │   ├── mlp_product_job.sh         # MLP model job script
│   │   └── README.md                  # C&S specific documentation
│   │
│   ├── GCN/
│   │   └── tunedGNN/                  # GCN model implementation
│   │       ├── large_graph/           # Large graph experiments
│   │       │   └── products.sh        # Products dataset script
│   │       ├── product_job.sh         # GCN job script
│   │       └── README.md              # GCN documentation
│   │
│   ├── SCR/
│   │   └── ogbn-products/             # GAMLP models with SCR
│   │       ├── main.py                # Main training script
│   │       ├── pre_processing.py      # Preprocessing
│   │       ├── post_processing.py     # C&S post-processing
│   │       ├── model_runs_tm_*.sh     # Job scripts
│   │       └── README.md              # SCR documentation
│   │
│   └── Voting/                        # Voting strategy experiments
│       ├── voting_experiments.ipynb   # Main voting experiments (9 strategies)
│       ├── model_aggregation.ipynb    # Model analysis and selection
│       ├── data_fix.ipynb             # Data preprocessing utilities
│       └── requirements_notebook.txt   # Notebook dependencies
│
└── [prediction files and model outputs]
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.12.1+
- PyTorch Geometric 2.3.1+
- OGB (Open Graph Benchmark)
- CUDA-capable GPU (recommended for training)

### Installation

Each subdirectory has its own requirements. See individual README files for specific dependencies:

- **CorrectAndSmooth**: See `external/CorrectAndSmooth/README.md`
- **GCN**: See `external/GCN/tunedGNN/README.md`
- **SCR**: See `external/SCR/ogbn-products/README.md`
- **Voting**: See `external/Voting/requirements_notebook.txt`

### Running Individual Models

#### CorrectAndSmooth Models (Plain, Linear, MLP)

```bash
cd external/CorrectAndSmooth

# Generate models
python gen_models.py --dataset products --model plain --epochs 1000 --lr 0.1
python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python gen_models.py --dataset products --model mlp --hidden_channels 200 --use_embeddings

# Run experiments with C&S
python run_experiments.py --dataset products --method plain
python run_experiments.py --dataset products --method linear
python run_experiments.py --dataset products --method mlp
```

#### GCN Model

```bash
cd external/GCN/tunedGNN/large_graph
python product.py --device 0 --ln --gnn gcn --save_outputs gcn_results.pt
```

#### SCR Models (GAMLP)

```bash
cd external/SCR/ogbn-products

# Preprocessing
python pre_processing.py --num_hops 5 --dataset ogbn-products

# Training GAMLP+SCR
python main.py --use-rlu --method R_GAMLP_RLU --stages 800 --dataset ogbn-products --num-runs 10

# Optional: Apply C&S post-processing
python post_processing.py --file_name <model_output> --correction_alpha 0.478 --smoothing_alpha 0.400
```

### Running Voting Experiments

The voting experiments are conducted in Jupyter notebooks:

```bash
cd external/Voting
jupyter notebook voting_experiments.ipynb
```

The notebook:
1. Loads predictions from all 6 models
2. Implements 9 different voting strategies
3. Evaluates ensemble performance on the test set
4. Compares results and identifies the best strategy

**Key Configuration** (in the notebook):
- `dataset_name = 'products'`
- `predictions_dir = 'predictions'` (where model predictions are stored)
- `models_dir = 'models'` (where model outputs are stored)
- `labels_csv = 'gcn_predictions_2.csv'` (true labels file)

## Results Summary

### Individual Model Performance
| Model | Accuracy | Improvement over Average |
|-------|----------|-------------------------|
| GAMLP+CS | 85.19% | +1.79% |
| GAMLP | 85.08% | +1.68% |
| MLP | 84.22% | +0.82% |
| Linear | 82.97% | -0.43% |
| Plain | 82.54% | -0.86% |
| GCN | 80.41% | -2.99% |
| **Average** | **83.40%** | - |

### Ensemble Performance (Top 3 Methods)
| Voting Method | Accuracy | Improvement over Best Individual |
|---------------|----------|--------------------------------|
| **Geometric Mean** | **85.39%** | **+0.20%** |
| Proportional Top-5 | 85.38% | +0.19% |
| Probability Sum | 85.24% | +0.05% |

### Key Findings
1. **Ensemble Benefit**: All voting strategies improve over the average individual model accuracy (83.40%)
2. **Best Strategy**: Geometric Mean Voting achieves the highest accuracy (85.39%)
3. **Robustness**: Multiple strategies achieve similar performance (85.24-85.39%), suggesting ensemble robustness
4. **Model Diversity**: The ensemble benefits from diverse architectures (linear, MLP, GCN, GAMLP) and training approaches (C&S, SCR)

## Job Scripts

The repository includes job scripts for running experiments on cluster systems (LSF format):

- `external/CorrectAndSmooth/product_job.sh` - Main products job
- `external/CorrectAndSmooth/plain_product_job.sh` - Plain model
- `external/CorrectAndSmooth/mlp_product_job.sh` - MLP model
- `external/GCN/tunedGNN/product_job.sh` - GCN model
- `external/SCR/ogbn-products/model_runs_tm_*.sh` - SCR/GAMLP models

## References

### Papers and Repositories

1. **Correct and Smooth**: [Combining Label Propagation and Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993)
   - Repository: `external/CorrectAndSmooth/`

2. **Classic GNNs**: [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://openreview.net/forum?id=xkljKdGe4E)
   - Repository: `external/GCN/tunedGNN/`

3. **SCR**: [Training Graph Neural Networks with Consistency Regularization](https://arxiv.org/abs/2112.04319)
   - Repository: `external/SCR/`

4. **GAMLP**: [Graph Attention Multi-Layer Perceptron](https://arxiv.org/abs/2108.10097)

5. **OGB Dataset**: [Open Graph Benchmark](https://ogb.stanford.edu/)

## License

Each external repository maintains its own license. Please refer to individual LICENSE files in each subdirectory.

## Contributors

This project was developed as part of CSC591 - Machine Learning on Graphs coursework.

- Christopher Elchik
- Teague McCracken
- Harsh Manoj Moore

## Notes

- All experiments were run on GPU clusters (A100, H100, H200, L40)
- Model predictions are stored in `.pt` (PyTorch) and `.csv` formats
- The voting experiments require all model predictions to be available
- Some models use Correct and Smooth (C&S) post-processing, which requires additional hyperparameter tuning
