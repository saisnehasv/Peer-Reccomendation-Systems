"""
Pair construction and complementarity labeling.

This module creates student pairs and assigns complementarity scores
based on multiple heuristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def sample_pairs(student_ids, max_pairs=50000, seed=42):
    """Sample student pairs for analysis.
    
    Args:
        student_ids: Array of student IDs
        max_pairs: Maximum number of pairs to sample
        seed: Random seed for reproducibility
        
    Returns:
        list: List of (student_i, student_j) tuples
    """
    print(f"Sampling pairs from {len(student_ids)} students...")
    
    # Calculate total possible pairs
    n = len(student_ids)
    total_pairs = n * (n - 1) // 2
    
    print(f"  Total possible pairs: {total_pairs:,}")
    
    if total_pairs <= max_pairs:
        # Use all pairs
        print(f"  Using all pairs")
        pairs = list(combinations(student_ids, 2))
    else:
        # Random sample
        print(f"  Sampling {max_pairs:,} random pairs")
        np.random.seed(seed)
        pairs = []
        
        while len(pairs) < max_pairs:
            i, j = np.random.choice(student_ids, size=2, replace=False)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((min(i, j), max(i, j)))
    
    print(f"✓ Created {len(pairs):,} pairs")
    return pairs


def compute_topical_complementarity(row_i, row_j, topic_vec_i, topic_vec_j):
    """Compute topical complementarity score.
    
    Complementarity = one student is strong where the other is weak.
    
    Args:
        row_i: Features for student i
        row_j: Features for student j
        topic_vec_i: Topic vector for student i
        topic_vec_j: Topic vector for student j
        
    Returns:
        float: Complementarity score [0, 1]
    """
    # Method 1: Topic complementarity
    # Where one is strong (>0.5) and other is weak (<0.5)
    complementary_topics = 0
    
    for t_i, t_j in zip(topic_vec_i, topic_vec_j):
        # Check if one strong, one weak
        if (t_i > 0.5 and t_j < 0.5) or (t_i < 0.5 and t_j > 0.5):
            complementary_topics += 1
    
    # Normalize by total topics
    topic_comp = complementary_topics / len(topic_vec_i)
    
    # Method 2: Overall performance complementarity
    # One has higher assessment score than the other
    score_i = row_i['avg_assess_score']
    score_j = row_j['avg_assess_score']
    
    # Avoid pairing very weak students together or very strong together
    # Optimal gap is 10-30 points
    score_gap = abs(score_i - score_j)
    
    if 10 <= score_gap <= 30:
        score_comp = 1.0
    elif score_gap < 10:
        score_comp = score_gap / 10.0  # Too similar
    else:
        score_comp = max(0, 1.0 - (score_gap - 30) / 40.0)  # Too different
    
    # Combine both signals
    complementarity = 0.6 * topic_comp + 0.4 * score_comp
    
    return complementarity


def compute_engagement_complementarity(row_i, row_j):
    """Compute engagement-based complementarity.
    
    Pairs high-engagement tutor with motivated weaker partner.
    
    Args:
        row_i: Features for student i
        row_j: Features for student j
        
    Returns:
        float: Complementarity score [0, 1]
    """
    # Get engagement metrics
    engagement_i = row_i['engagement_rate']
    engagement_j = row_j['engagement_rate']
    
    # Get performance
    score_i = row_i['avg_assess_score']
    score_j = row_j['avg_assess_score']
    
    # Identify high performer and low performer
    high_idx = 0 if score_i > score_j else 1
    low_idx = 1 - high_idx
    
    high_engagement = engagement_i if high_idx == 0 else engagement_j
    low_engagement = engagement_i if low_idx == 0 else engagement_j
    high_score = score_i if high_idx == 0 else score_j
    low_score = score_i if low_idx == 0 else score_j
    
    # Good pairing: high performer is highly engaged (good tutor)
    # AND lower performer is also engaged (motivated learner)
    tutor_quality = (high_engagement + 3) / 6  # Normalize engagement z-score
    learner_motivation = (low_engagement + 3) / 6
    
    # Performance gap should be moderate
    score_gap = abs(high_score - low_score)
    gap_score = 1.0 if 10 <= score_gap <= 30 else 0.5
    
    # Combine signals
    complementarity = 0.4 * tutor_quality + 0.4 * learner_motivation + 0.2 * gap_score
    
    return np.clip(complementarity, 0, 1)


def compute_learning_trajectory_complementarity(row_i, row_j):
    """Compute complementarity based on learning trajectories.
    
    Pairs improving student with steady high performer.
    
    Args:
        row_i: Features for student i
        row_j: Features for student j
        
    Returns:
        float: Complementarity score [0, 1]
    """
    slope_i = row_i['improvement_slope']
    slope_j = row_j['improvement_slope']
    
    score_i = row_i['avg_assess_score']
    score_j = row_j['avg_assess_score']
    
    # Good pairing: one improving (positive slope) with one stable/high performer
    if slope_i > 0 and score_j > 70 and abs(slope_j) < 0.5:
        # i is improving, j is stable high performer
        complementarity = min(1.0, (slope_i / 2.0 + 0.5))
    elif slope_j > 0 and score_i > 70 and abs(slope_i) < 0.5:
        # j is improving, i is stable high performer
        complementarity = min(1.0, (slope_j / 2.0 + 0.5))
    else:
        # Not a good trajectory match
        complementarity = 0.3
    
    return complementarity


def label_pairs(pairs, df_features, topic_vectors, label_method='combined'):
    """Assign complementarity labels to pairs.
    
    Args:
        pairs: List of (student_i, student_j) tuples
        df_features: Student features dataframe
        topic_vectors: Topic vectors array
        label_method: 'topical', 'engagement', 'trajectory', or 'combined'
        
    Returns:
        pd.DataFrame: Pairs with labels and features
    """
    print(f"Labeling {len(pairs):,} pairs using '{label_method}' method...")
    
    # Create student ID to index mapping
    id_to_idx = {sid: idx for idx, sid in enumerate(df_features['id_student'].values)}
    
    pair_data = []
    
    for idx, (sid_i, sid_j) in enumerate(pairs):
        if (idx + 1) % 10000 == 0:
            print(f"  Labeled {idx + 1:,}/{len(pairs):,} pairs...")
        
        # Get features
        try:
            i_idx = id_to_idx[sid_i]
            j_idx = id_to_idx[sid_j]
        except KeyError:
            continue
        
        row_i = df_features.iloc[i_idx]
        row_j = df_features.iloc[j_idx]
        topic_i = topic_vectors[i_idx]
        topic_j = topic_vectors[j_idx]
        
        # Skip if same course/presentation (for diversity)
        if row_i['code_module'] != row_j['code_module']:
            continue
        
        # Compute complementarity scores
        if label_method == 'topical':
            score = compute_topical_complementarity(row_i, row_j, topic_i, topic_j)
        elif label_method == 'engagement':
            score = compute_engagement_complementarity(row_i, row_j)
        elif label_method == 'trajectory':
            score = compute_learning_trajectory_complementarity(row_i, row_j)
        else:  # combined
            score_topic = compute_topical_complementarity(row_i, row_j, topic_i, topic_j)
            score_engage = compute_engagement_complementarity(row_i, row_j)
            score_traj = compute_learning_trajectory_complementarity(row_i, row_j)
            score = 0.4 * score_topic + 0.4 * score_engage + 0.2 * score_traj
        
        # Store pair data
        pair_data.append({
            'student_i': sid_i,
            'student_j': sid_j,
            'complementarity_score': score,
            # Student i features
            'score_i': row_i['avg_assess_score'],
            'engagement_i': row_i['engagement_rate'],
            'clicks_i': row_i['total_clicks'],
            'slope_i': row_i['improvement_slope'],
            # Student j features
            'score_j': row_j['avg_assess_score'],
            'engagement_j': row_j['engagement_rate'],
            'clicks_j': row_j['total_clicks'],
            'slope_j': row_j['improvement_slope'],
            # Difference features
            'score_diff': abs(row_i['avg_assess_score'] - row_j['avg_assess_score']),
            'engagement_diff': abs(row_i['engagement_rate'] - row_j['engagement_rate']),
        })
    
    df_pairs = pd.DataFrame(pair_data)
    
    print(f"✓ Labeled {len(df_pairs):,} valid pairs")
    return df_pairs


def create_binary_labels(df_pairs, threshold_percentile=75):
    """Create binary labels from complementarity scores.
    
    Args:
        df_pairs: Pairs dataframe with complementarity scores
        threshold_percentile: Percentile for positive class
        
    Returns:
        pd.DataFrame: Pairs with binary labels
    """
    threshold = np.percentile(df_pairs['complementarity_score'], threshold_percentile)
    
    df_pairs['is_good_pair'] = (df_pairs['complementarity_score'] >= threshold).astype(int)
    
    print(f"\nBinary label statistics:")
    print(f"  Threshold (P{threshold_percentile}): {threshold:.3f}")
    print(f"  Positive pairs: {df_pairs['is_good_pair'].sum():,} ({df_pairs['is_good_pair'].mean()*100:.1f}%)")
    print(f"  Negative pairs: {(1-df_pairs['is_good_pair']).sum():,} ({(1-df_pairs['is_good_pair']).mean()*100:.1f}%)")
    
    return df_pairs


def main():
    """Main pair construction and labeling pipeline."""
    # Load processed data
    print("Loading processed data...")
    df_features = pd.read_csv("../data/processed/student_features.csv")
    topic_vectors = np.load("../data/processed/student_topic_vectors.npy")
    
    # Sample pairs
    student_ids = df_features['id_student'].values
    pairs = sample_pairs(student_ids, max_pairs=50000, seed=42)
    
    # Label pairs with combined method
    df_pairs = label_pairs(pairs, df_features, topic_vectors, label_method='combined')
    
    # Create binary labels
    df_pairs = create_binary_labels(df_pairs, threshold_percentile=75)
    
    # Save
    output_file = Path("../data/processed/pairs_labeled.csv")
    df_pairs.to_csv(output_file, index=False)
    print(f"\n✓ Saved labeled pairs to {output_file}")
    
    # Summary statistics
    print("\n=== PAIR LABELING SUMMARY ===")
    print(f"Total pairs: {len(df_pairs):,}")
    print(f"\nComplementarity score distribution:")
    print(df_pairs['complementarity_score'].describe())
    print(f"\nPositive pairs: {df_pairs['is_good_pair'].sum():,}")
    print(f"Negative pairs: {(1-df_pairs['is_good_pair']).sum():,}")


if __name__ == "__main__":
    main()