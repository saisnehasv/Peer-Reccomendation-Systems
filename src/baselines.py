"""
Baseline models for peer recommendation.

Implements:
1. Heuristic ranker
2. Logistic Regression
3. Random Forest
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from evaluation import evaluate_ranker, print_evaluation_results


class HeuristicRanker:
    """Simple heuristic-based ranker using complementarity scores."""
    
    def __init__(self):
        self.name = "Heuristic Ranker"
    
    def fit(self, X, y):
        """No training needed for heuristic."""
        pass
    
    def predict_proba(self, X):
        """Use complementarity score directly as probability.
        
        Assumes last column is complementarity_score.
        """
        if isinstance(X, pd.DataFrame):
            if 'complementarity_score' in X.columns:
                scores = X['complementarity_score'].values
            else:
                # Use last column
                scores = X.iloc[:, -1].values
        else:
            scores = X[:, -1]
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # Return as probability for class 1
        return np.column_stack([1 - scores, scores])
    
    def predict(self, X):
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def create_pairwise_features(df_pairs, include_raw=True):
    """Create features for pairwise classification.
    
    Args:
        df_pairs: DataFrame with pair features
        include_raw: Whether to include raw features
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    features = []
    feature_names = []
    
    # Raw features for both students
    if include_raw:
        for suffix in ['_i', '_j']:
            for col in ['score', 'engagement', 'clicks', 'slope']:
                col_name = col + suffix
                if col_name in df_pairs.columns:
                    features.append(df_pairs[col_name].values)
                    feature_names.append(col_name)
    
    # Difference features
    for col in ['score_diff', 'engagement_diff']:
        if col in df_pairs.columns:
            features.append(df_pairs[col].values)
            feature_names.append(col)
    
    # Product features
    if 'score_i' in df_pairs.columns and 'score_j' in df_pairs.columns:
        features.append(df_pairs['score_i'].values * df_pairs['score_j'].values)
        feature_names.append('score_product')
    
    if 'engagement_i' in df_pairs.columns and 'engagement_j' in df_pairs.columns:
        features.append(df_pairs['engagement_i'].values * df_pairs['engagement_j'].values)
        feature_names.append('engagement_product')
    
    # Complementarity score (ground truth heuristic)
    if 'complementarity_score' in df_pairs.columns:
        features.append(df_pairs['complementarity_score'].values)
        feature_names.append('complementarity_score')
    
    # Stack features
    X = np.column_stack(features)
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Handle missing values - fill with median
    for col in X_df.columns:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            if np.isnan(median_val):  # If all values are NaN
                median_val = 0.0
            X_df[col] = X_df[col].fillna(median_val)
    
    return X_df


def split_data(df_pairs, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train/val/test sets.
    
    Stratified by label to maintain class balance.
    
    Args:
        df_pairs: DataFrame with pairs and labels
        test_size: Proportion for test set
        val_size: Proportion of remaining for validation
        random_state: Random seed
        
    Returns:
        tuple: (train, val, test) DataFrames
    """
    print("Splitting data...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df_pairs,
        test_size=test_size,
        random_state=random_state,
        stratify=df_pairs['is_good_pair']
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['is_good_pair']
    )
    
    print(f"  Train: {len(train):,} pairs ({train['is_good_pair'].mean():.2%} positive)")
    print(f"  Val:   {len(val):,} pairs ({val['is_good_pair'].mean():.2%} positive)")
    print(f"  Test:  {len(test):,} pairs ({test['is_good_pair'].mean():.2%} positive)")
    
    return train, val, test


class BaselineModel:
    """Wrapper for baseline ML models."""
    
    def __init__(self, model_type='logistic', **kwargs):
        """Initialize model.
        
        Args:
            model_type: 'logistic' or 'random_forest'
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **kwargs
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Train model.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X):
        """Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Probability predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        """Predict labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Binary predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"✓ Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath, model_type='logistic'):
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=model_type)
        instance.model = data['model']
        instance.scaler = data['scaler']
        return instance


def train_and_evaluate_baseline(train_df, val_df, test_df, model_type='logistic', **kwargs):
    """Train and evaluate a baseline model.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        model_type: 'heuristic', 'logistic', or 'random_forest'
        **kwargs: Model hyperparameters
        
    Returns:
        tuple: (model, train_results, val_results, test_results)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}\n")
    
    # Create features
    X_train = create_pairwise_features(train_df)
    y_train = train_df['is_good_pair'].values
    
    X_val = create_pairwise_features(val_df)
    y_val = val_df['is_good_pair'].values
    
    X_test = create_pairwise_features(test_df)
    y_test = test_df['is_good_pair'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {X_train.columns.tolist()}\n")
    
    # Initialize model
    if model_type == 'heuristic':
        model = HeuristicRanker()
    else:
        model = BaselineModel(model_type=model_type, **kwargs)
    
    # Train
    print("Training...")
    model.fit(X_train, y_train)
    print("✓ Training complete\n")
    
    # Evaluate on all splits
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        print(f"\nEvaluating on {split_name} set...")
        y_scores = model.predict_proba(X)[:, 1]
        split_results = evaluate_ranker(y, y_scores, k_values=[1, 3, 5, 10])
        print_evaluation_results(split_results, f"{model_type.upper()} - {split_name.upper()}")
        results[split_name] = split_results
    
    return model, results


def main():
    """Main baseline training pipeline."""
    # Load data
    print("Loading data...")
    df_pairs = pd.read_csv("../data/processed/pairs_labeled.csv")
    print(f"✓ Loaded {len(df_pairs):,} pairs\n")
    
    # Split data
    train_df, val_df, test_df = split_data(df_pairs, test_size=0.2, val_size=0.1)
    
    # Train heuristic baseline
    heuristic_model, heuristic_results = train_and_evaluate_baseline(
        train_df, val_df, test_df, 
        model_type='heuristic'
    )
    
    # Train logistic regression
    lr_model, lr_results = train_and_evaluate_baseline(
        train_df, val_df, test_df,
        model_type='logistic',
        C=1.0,
        class_weight='balanced'
    )
    
    # Save logistic regression model
    lr_model.save("../models/logistic_baseline.pkl")
    
    # Train random forest
    rf_model, rf_results = train_and_evaluate_baseline(
        train_df, val_df, test_df,
        model_type='random_forest',
        n_estimators=100,
        max_depth=10,
        class_weight='balanced'
    )
    
    # Save random forest model
    rf_model.save("../models/random_forest_baseline.pkl")
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON (Test Set)")
    print("="*60 + "\n")
    
    comparison_df = pd.DataFrame({
        'Model': ['Heuristic', 'Logistic Regression', 'Random Forest'],
        'Precision@3': [
            heuristic_results['test']['precision@3'],
            lr_results['test']['precision@3'],
            rf_results['test']['precision@3']
        ],
        'NDCG@5': [
            heuristic_results['test']['ndcg@5'],
            lr_results['test']['ndcg@5'],
            rf_results['test']['ndcg@5']
        ],
        'MAP': [
            heuristic_results['test']['map'],
            lr_results['test']['map'],
            rf_results['test']['map']
        ],
        'AUC': [
            heuristic_results['test']['auc_roc'],
            lr_results['test']['auc_roc'],
            rf_results['test']['auc_roc']
        ]
    })
    
    print(comparison_df.to_string(index=False))
    print()
    
    # Save results
    comparison_df.to_csv("../results/baseline_comparison.csv", index=False)
    print("✓ Results saved to results/baseline_comparison.csv")


if __name__ == "__main__":
    main()