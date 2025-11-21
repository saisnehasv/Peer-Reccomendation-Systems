"""
Evaluation metrics for peer recommendation system.

Implements ranking metrics: Precision@K, Recall@K, NDCG, MAP, MRR.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score


def precision_at_k(y_true, y_scores, k=5):
    """Calculate Precision@K.
    
    Args:
        y_true: Binary relevance labels
        y_scores: Predicted scores
        k: Number of top predictions to consider
        
    Returns:
        float: Precision@K score
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Get top k
    top_k_indices = sorted_indices[:k]
    top_k_true = y_true[top_k_indices]
    
    # Calculate precision
    return np.sum(top_k_true) / k


def recall_at_k(y_true, y_scores, k=5):
    """Calculate Recall@K.
    
    Args:
        y_true: Binary relevance labels
        y_scores: Predicted scores
        k: Number of top predictions to consider
        
    Returns:
        float: Recall@K score
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Get top k
    top_k_indices = sorted_indices[:k]
    top_k_true = y_true[top_k_indices]
    
    # Calculate recall
    n_relevant = np.sum(y_true)
    if n_relevant == 0:
        return 0.0
    
    return np.sum(top_k_true) / n_relevant


def dcg_at_k(y_true, y_scores, k=5):
    """Calculate Discounted Cumulative Gain at K.
    
    Args:
        y_true: Relevance labels (can be binary or graded)
        y_scores: Predicted scores
        k: Number of top predictions to consider
        
    Returns:
        float: DCG@K score
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Get top k
    top_k_indices = sorted_indices[:k]
    top_k_true = y_true[top_k_indices]
    
    # Calculate DCG
    gains = 2 ** top_k_true - 1
    discounts = np.log2(np.arange(2, k + 2))
    
    return np.sum(gains / discounts)


def ndcg_at_k(y_true, y_scores, k=5):
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        y_true: Relevance labels (can be binary or graded)
        y_scores: Predicted scores
        k: Number of top predictions to consider
        
    Returns:
        float: NDCG@K score
    """
    dcg = dcg_at_k(y_true, y_scores, k)
    
    # Calculate ideal DCG (sort by true relevance)
    ideal_scores = y_true.copy()
    idcg = dcg_at_k(y_true, ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_average_precision(y_true, y_scores):
    """Calculate Mean Average Precision.
    
    Args:
        y_true: Binary relevance labels
        y_scores: Predicted scores
        
    Returns:
        float: MAP score
    """
    return average_precision_score(y_true, y_scores)


def mean_reciprocal_rank(y_true, y_scores):
    """Calculate Mean Reciprocal Rank.
    
    Args:
        y_true: Binary relevance labels
        y_scores: Predicted scores
        
    Returns:
        float: MRR score
    """
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_true = y_true[sorted_indices]
    
    # Find first relevant item
    relevant_positions = np.where(sorted_true == 1)[0]
    
    if len(relevant_positions) == 0:
        return 0.0
    
    # Rank is 1-indexed
    first_relevant_rank = relevant_positions[0] + 1
    
    return 1.0 / first_relevant_rank


def evaluate_ranker(y_true, y_scores, k_values=[1, 3, 5, 10]):
    """Comprehensive evaluation of a ranking model.
    
    Args:
        y_true: Binary relevance labels
        y_scores: Predicted scores
        k_values: List of K values for Precision@K and NDCG@K
        
    Returns:
        dict: Dictionary of metric scores
    """
    results = {}
    
    # Classification metrics (using median as threshold)
    threshold = np.median(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    results['accuracy'] = np.mean(y_pred == y_true)
    results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Try AUC-ROC (if both classes present)
    if len(np.unique(y_true)) > 1:
        results['auc_roc'] = roc_auc_score(y_true, y_scores)
    else:
        results['auc_roc'] = np.nan
    
    # Ranking metrics
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(y_true, y_scores, k)
        results[f'recall@{k}'] = recall_at_k(y_true, y_scores, k)
        results[f'ndcg@{k}'] = ndcg_at_k(y_true, y_scores, k)
    
    # MAP and MRR
    results['map'] = mean_average_precision(y_true, y_scores)
    results['mrr'] = mean_reciprocal_rank(y_true, y_scores)
    
    return results


def print_evaluation_results(results, model_name="Model"):
    """Pretty print evaluation results.
    
    Args:
        results: Dictionary from evaluate_ranker
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}\n")
    
    print("Classification Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    
    print("\nRanking Metrics:")
    # Extract k values
    k_values = sorted([int(k.split('@')[1]) for k in results.keys() if 'precision@' in k])
    
    print(f"{'K':<5} {'P@K':<10} {'R@K':<10} {'NDCG@K':<10}")
    print("-" * 40)
    for k in k_values:
        print(f"{k:<5} {results[f'precision@{k}']:<10.4f} "
              f"{results[f'recall@{k}']:<10.4f} "
              f"{results[f'ndcg@{k}']:<10.4f}")
    
    print(f"\n  MAP: {results['map']:.4f}")
    print(f"  MRR: {results['mrr']:.4f}")
    print()


def compute_coverage(recommended_students, all_students):
    """Calculate recommendation coverage.
    
    Args:
        recommended_students: Set of students who received recommendations
        all_students: Set of all students
        
    Returns:
        float: Coverage percentage
    """
    return len(recommended_students) / len(all_students)


def compute_diversity(recommendations, feature_matrix):
    """Calculate intra-list diversity of recommendations.
    
    Args:
        recommendations: List of recommended student IDs for each student
        feature_matrix: Feature matrix for computing similarity
        
    Returns:
        float: Average pairwise distance in recommendations
    """
    from scipy.spatial.distance import pdist
    
    diversities = []
    
    for rec_list in recommendations:
        if len(rec_list) < 2:
            continue
        
        # Get features for recommended students
        rec_features = feature_matrix[rec_list]
        
        # Compute pairwise distances
        distances = pdist(rec_features, metric='euclidean')
        diversities.append(np.mean(distances))
    
    return np.mean(diversities) if diversities else 0.0


class MetricsTracker:
    """Track and save evaluation metrics across experiments."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, model_name, metrics, metadata=None):
        """Add evaluation result.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric scores
            metadata: Additional metadata (hyperparameters, etc.)
        """
        result = {
            'model': model_name,
            **metrics
        }
        
        if metadata:
            result.update(metadata)
        
        self.results.append(result)
    
    def get_dataframe(self):
        """Get results as DataFrame."""
        return pd.DataFrame(self.results)
    
    def save(self, filepath):
        """Save results to CSV."""
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        print(f"✓ Saved results to {filepath}")
    
    def print_comparison(self, metrics=['precision@3', 'ndcg@5', 'map']):
        """Print comparison table."""
        df = self.get_dataframe()
        
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}\n")
        
        cols = ['model'] + metrics
        print(df[cols].to_string(index=False))
        print()


def main():
    """Test evaluation functions."""
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.binomial(1, 0.25, n_samples)
    y_scores = np.random.random(n_samples)
    
    # Add some correlation
    y_scores = y_scores * 0.7 + y_true * 0.3
    
    # Evaluate
    results = evaluate_ranker(y_true, y_scores, k_values=[1, 3, 5, 10])
    print_evaluation_results(results, "Test Model")


if __name__ == "__main__":
    main()