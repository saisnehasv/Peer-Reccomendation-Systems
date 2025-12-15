# Peer Recommendation System: Complementarity-Based Student Matching

A machine learning system that identifies pedagogically beneficial student pairings using graph neural networks and the Open University Learning Analytics Dataset (OULAD).

## Authors
- Sai Sneha Siddapura Venkataramappa (saisneha@umich.edu)
- Yuganshi Agrawal (yuganshi@umich.edu)

Department of Statistics, University of Michigan

## Project Overview

Traditional peer matching approaches rely on similarity-based algorithms that pair students with comparable skills and backgrounds. This project implements a complementarity-based approach, where students with different but mutually beneficial strengths are matched together to maximize learning opportunities.


## Repository Structure

```
.
├── notebook-1-data-processing-feature-eng.ipynb   # Data loading and feature engineering
├── notebook-2-baselines.ipynb                      # Baseline model training and evaluation
├── notebook-3-gnn-model.ipynb                      # Graph neural network implementation
├── notebook-4-gnn-recommendation.ipynb             # Advanced GNN with interpretability
├── OULAD/                                          # Raw OULAD dataset
│   ├── studentInfo.csv
│   ├── studentVle.csv
│   ├── vle.csv
│   ├── assessments.csv
│   ├── studentAssessment.csv
│   ├── courses.csv
│   └── studentRegistration.csv
├── data/
│   ├── processed/                                  # Processed data files
│   │   ├── student_features.pkl
│   │   ├── training_pairs.pkl
│   │   └── data_summary.json
│   └── features/                                   # Additional feature files and checkpoints
├── models/
│   └── checkpoints/                                # Saved model files
│   └── embeddings/                                 # Saved embeddings from gnn
├── results/                                        # Visualizations and analysis
│   └── analysis/                                   # Saved graphs
│       ├── roc_curves.png
│       ├── learning_curve_logistic_regression.png
│       ├── learning_curve_xgboost.png
│       ├── feature_importance.png
│       ├── complementarity_distribution.png
│       ├── model_comparison.png
│       └── reccomender-explination.png
│   └── metrics/                                    # Saved model metrics
│       ├── baseline_metrics.json
│       ├── feature_info.json
│       ├── gnn_metrics.json
│       ├── gnn vs baseline.josn
│       ├── interpreteability.json
│       └── reccomendation_metrics_complete.json
├── requirements.txt                                # Python dependencies
├── README.md                                       # This file
```

## Dataset Statistics

### OULAD Dataset
- **Download Link**: https://analyse.kmi.open.ac.uk/open-dataset
- **Total Enrollments**: 32,593
- **Unique Students**: 28,785 (after filtering)
- **VLE Interactions**: 10,655,280 click events
- **Assessment Records**: 173,912 across TMA, CMA, and Exam types
- **Modules**: 7 courses (AAA through GGG)
- **Presentations**: 4 time periods

### Processed Features
- **Feature Matrix Shape**: (28,785, 15)
- **Features per Student**: 14 (plus student ID)
- **Feature Categories**:
  - Temporal engagement (early, late, consistency, improvement)
  - Skills (avg_score, TMA, CMA, Exam, variance)
  - Learning patterns (total_clicks, active_days, click_variability)
  - Demographics (encoded)

### Training Data
- **Total Pairs Generated**: 49,999
- **Positive Pairs (label=1)**: 12,500 (25.0%)
- **Negative Pairs (label=0)**: 37,499 (75.0%)
- **Class Imbalance Ratio**: 1:3.0
- **Complementarity Threshold**: 0.557 (75th percentile)

### Complementarity Score Distribution
- **Mean**: 0.3910
- **Std**: 0.2056
- **Min**: 0.0000
- **25th percentile**: 0.2039
- **50th percentile (median)**: 0.4257
- **75th percentile**: 0.5572
- **Max**: 0.9892

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- At least 8GB RAM recommended
- CPU or GPU (GPU recommended for faster GNN training)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd peer-recommendation-system

# Start Jupyter
jupyter notebook
```
# Install dependencies
pip install -r requirements.txt

## Usage
### Set up Dataset 

Download the dataset from (https://analyse.kmi.open.ac.uk/open-dataset) and place it in OULAD/ folder. 

### Running the Complete Pipeline

Execute the notebooks in order:

#### 1. Data Processing and Feature Engineering
**File**: `notebook-1-data-processing-feature-eng.ipynb`

**What it does:**
- Loads raw OULAD CSV files
- Engineers 14 student features across 4 categories
- Computes complementarity scores for 49,999 student pairs
- Applies module-specific normalization
- Generates binary labels using 75th percentile threshold (0.557)

#### 2. Baseline Model Training
**File**: `notebook-2-baselines.ipynb`

**What it does:**
- Trains Logistic Regression (ROC-AUC: 0.7882)
- Trains XGBoost with extensive regularization (ROC-AUC: 0.8315)
- Implements KNN similarity-based recommendations
- Generates learning curves and feature importance analysis
- Creates comprehensive comparison visualizations

#### 3. Graph Neural Network Training
**File**: `notebook-3-gnn-model.ipynb`

**What it does:**
- Constructs student similarity graphs (k=20 neighbors per student)
- Implements GraphSAGE with 2-layer architecture
- Trains with focal loss (α=0.75, γ=2.0) for class imbalance
- Achieves 97.64% ROC-AUC on test set
- Generates student embeddings (64-dimensional)

**Architecture:**
- Input features: 14
- Embedding dimension: 64
- Hidden dimension: 64
- Number of layers: 2
- Dropout: 0.3
- Graph edges: k=20 nearest neighbors

**Training:**
- Optimizer: Adam (lr=0.001)
- Batch size: 4096 edges
- Early stopping: patience 3
- Loss function: Focal loss (custom implementation)

#### 4. Advanced GNN with Interpretability
**File**: `notebook-4-gnn-recommendation.ipynb`

**What it does:**
- Implements hierarchical GraphSAGE with custom attention
- Applies hard negative mining (2 from 5 candidates per positive)
- Implements temperature scaling for calibration (T=2.0)
- Creates ensemble scoring (0.4 raw + 0.6 calibrated)
- Generates interpretable recommendations with factor decomposition
- Achieves 98.38% ROC-AUC (98.4% rounded)

**Hierarchical Architecture:**
- Separate processing for early/late engagement patterns
- Attention-based neighbor aggregation (custom implementation)
- Learned fusion layer for temporal streams

**Interpretability Framework:**
Each recommendation includes:
- Engagement difference score (0-100%)
- Performance gap score (0-100%)
- Gender diversity score (0-100%)
- Age diversity score (0-100%)
- Overall complementarity percentage

**Example Explanation** (Student 35355 → Student 537811):
- Engagement difference: 87% (25,159 vs 3,358 clicks)
- Performance gap: 22% (97.1 vs 75.2 avg score)
- Gender diversity: 50% (M vs F)
- Age diversity: 60% (35-55 vs 0-35)
- **Overall complementarity: 55% = "Good match"**

## Model Performance

### Complete Results Table

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score | Improvement |
|-------|---------|--------|----------|-----------|--------|----------|-------------|
| **Baselines** |
| Logistic Regression | 0.7882 | 0.4797 | 0.6896 | 0.4346 | 0.8032 | 0.5640 | baseline |
| XGBoost | 0.8315 | 0.5845 | 0.7028 | 0.4501 | 0.8520 | 0.5890 | +5.5% |
| **Graph Neural Networks** |
| GraphSAGE | 0.9764 | 0.8930 | N/A | N/A | N/A | N/A | +17.4% |
| **Hierarchical GraphSAGE** | **0.9838** | **0.93+** | **N/A** | **N/A** | **N/A** | **N/A** | **+18.3%** |

### Cross-Validation Results

| Model | CV ROC-AUC | Std Dev |
|-------|------------|---------|
| Logistic Regression | 0.6983 | ±0.0091 |
| XGBoost | 0.8315 | ±0.0045 |

### Confusion Matrix (XGBoost - Test Set)
- **True Negatives (TN)**: 4,898
- **False Positives (FP)**: 2,602
- **False Negatives (FN)**: 370
- **True Positives (TP)**: 2,130

### KNN Baseline
- **Approach**: Similarity-based (cosine similarity)
- **K**: 10 neighbors
- **Average Complementarity**: 0.5829
- **Std Complementarity**: 0.1986
- **Total Recommendations**: 287,850
- **Students Covered**: 28,785

**Analysis**: KNN recommendations cluster around 0.58 complementarity, with ~60% falling below the 0.557 threshold for positive pairs. This demonstrates that similarity-based matching is inadequate for identifying complementary pairings.

### Key Results

- **Best Model**: Hierarchical GraphSAGE achieves **98.4% ROC-AUC**
- **Improvement**: 17.4% relative improvement over best baseline (XGBoost at 83.2%)
- **Dataset**: 28,785 students, 10.6M VLE interactions, 174K assessment records
- **Training Pairs**: 49,999 pairs with 3:1 class imbalance (12,500 positive, 37,499 negative)

## Key Components

### 1. Complementarity Score

Ground truth combines three educational factors:

```
Complementarity(i,j) = 0.5 × SkillDiversity + 0.3 × ActivityAlignment - 0.2 × TemporalMismatch
```

**SkillDiversity**: Absolute differences in TMA, CMA, and Exam performance
**ActivityAlignment**: Cosine similarity of weekly engagement patterns
**TemporalMismatch**: Penalty for incompatible schedules

### 2. Custom GNN Architecture

**Hierarchical GraphSAGE Features:**
- Two-layer graph convolutions
- Attention-based neighbor aggregation (not in PyTorch Geometric)
- Separate processing of early (weeks 1-10) and late (weeks 20+) engagement
- Hierarchical fusion layer
- Dropout (0.3) for regularization

**Link Prediction:**
```python
prediction = sigmoid(W × [h_i || h_j || |h_i - h_j| || h_i ⊙ h_j])
```

### 3. Training Enhancements

**Focal Loss** (custom PyTorch implementation):
```python
L_focal = -α_t (1-p_t)^γ log(p_t)
```
- α_t = 0.75 for minority class (positive pairs)
- γ = 2.0 (focusing parameter)
- Addresses 3:1 class imbalance

**Hard Negative Mining:**
- For each positive pair: sample 5 random negatives
- Select 2 with highest predicted scores as hard negatives
- Dynamic buffer management during training

**Temperature Scaling:**
```python
p_calibrated = exp(z/T) / (exp(z/T) + exp(-z/T))
```
- T = 2.0 (learned on validation set)
- Reduces overconfidence in predictions

**Ensemble Scoring:**
```python
Score_final = 0.4 × p_raw + 0.6 × p_calibrated
```
- Weights optimized via grid search
- Balances discrimination and calibration

## Configuration

### Hyperparameters

**Feature Engineering:**
- Complementarity weights: α=0.5, β=0.3, γ=0.2
- Positive label threshold: 75th percentile (0.557)
- Training pairs: 49,999 sampled
- Class imbalance: 1:3 (positive:negative)

**GNN Architecture:**
- Embedding dimension: 64
- Hidden dimension: 64
- Number of layers: 2
- Dropout rate: 0.3
- k-neighbors for graph: 20
- Similarity metric: Cosine

**Training:**
- Optimizer: Adam (lr=0.001)
- Batch size: 4096 edges
- Epochs: 5-15 (with early stopping)
- Focal loss: α_t=0.75, γ=2.0
- Hard negatives per positive: 2 (from 5 candidates)
- Temperature: 2.0
- Ensemble weights: 0.4 raw, 0.6 calibrated

**Baselines:**
- XGBoost: max_depth=3, lr=0.05, min_child_weight=5, scale_pos_weight=3.0
- Logistic Regression: L2 penalty, class_weight='balanced'
- KNN: k=10, cosine similarity

## Feature Importance Analysis

### Top 10 XGBoost Features (by gain)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | Diff_9 | 0.2987 | Temporal engagement difference |
| 2 | Diff_10 | 0.2063 | Resource diversity difference |
| 3 | Diff_0 | 0.0815 | Overall engagement difference |
| 4 | Diff_8 | 0.0787 | Activity pattern difference |
| 5 | Diff_4 | 0.0750 | Early engagement difference |
| 6 | Diff_5 | 0.0688 | Late engagement difference |
| 7 | Diff_13 | 0.0542 | Demographic difference |
| 8 | Prod_9 | 0.0382 | Engagement interaction |
| 9 | Prod_10 | 0.0298 | Diversity interaction |
| 10 | Diff_3 | 0.0272 | Skill variance difference |

**Key Finding**: Difference features (capturing dissimilarity) dominate importance rankings, validating that complementarity is driven by diversity rather than similarity.

## Computational Requirements

### Hardware Used
- **CPU**: 32 cores (parallel processing enabled)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: ~10GB for data and models
- **GPU**: Optional but recommended for GNN training

### Runtime Benchmarks

| Notebook | CPU (32-core) | GPU (T4) | Output Size |
|----------|---------------|----------|-------------|
| Notebook 1 | ~18 minutes | N/A | ~5 MB |
| Notebook 2 | ~12 minutes | N/A | ~10 MB |
| Notebook 3 | ~35 minutes | ~8 minutes | ~50 MB |
| Notebook 4 | ~50 minutes | ~12 minutes | ~50 MB |
| **Total** | **~2 hours** | **~30 minutes** | **~115 MB** |

## Acknowledgments

- Open University UK for providing the OULAD dataset
- SI 670: Applied Machine Learning (Fall 2024) course staff
- PyTorch and scikit-learn communities

## License

This project is for educational purposes as part of SI 670 coursework at the University of Michigan.

## Contact

For questions or issues:
- Sai Sneha Siddapura Venkataramappa: saisneha@umich.edu
- Yuganshi Agrawal: yuganshi@umich.edu

## References

1. Webb, N. M. (1982). Student interaction and learning in small groups. *Review of Educational Research*, 52(3), 421-445.
2. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
3. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NIPS*.
4. Lin, T. Y., et al. (2017). Focal loss for dense object detection. *ICCV*.
5. Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.