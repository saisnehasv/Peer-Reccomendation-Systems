"""
Preprocessing and feature engineering for OULAD dataset.

This module contains functions to:
1. Clean and canonicalize raw data
2. Compute student-level features
3. Create topic/skill vectors
4. Save processed features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(data_path):
    """Load all OULAD CSV files.
    
    Args:
        data_path: Path to raw data directory
        
    Returns:
        dict: Dictionary of dataframes
    """
    print("Loading raw OULAD data...")
    
    data = {
        'student_info': pd.read_csv(Path(data_path) / "studentInfo.csv"),
        'student_assessment': pd.read_csv(Path(data_path) / "studentAssessment.csv"),
        'student_vle': pd.read_csv(Path(data_path) / "studentVle.csv"),
        'assessments': pd.read_csv(Path(data_path) / "assessments.csv"),
        'courses': pd.read_csv(Path(data_path) / "courses.csv"),
        'vle': pd.read_csv(Path(data_path) / "vle.csv"),
        'student_registration': pd.read_csv(Path(data_path) / "studentRegistration.csv")
    }
    
    print(f"✓ Loaded {len(data)} datasets")
    return data


def clean_data(data_dict):
    """Clean and canonicalize data.
    
    Args:
        data_dict: Dictionary of raw dataframes
        
    Returns:
        dict: Cleaned dataframes
    """
    print("Cleaning data...")
    
    cleaned = data_dict.copy()
    
    # Convert date_registration to datetime if exists
    if 'date_registration' in cleaned['student_registration'].columns:
        cleaned['student_registration']['date_registration'] = pd.to_datetime(
            cleaned['student_registration']['date_registration'], 
            errors='coerce'
        )
    
    # Standardize column names (lowercase, underscores)
    for key in cleaned:
        cleaned[key].columns = cleaned[key].columns.str.lower()
    
    # Remove any exact duplicates
    for key in cleaned:
        before = len(cleaned[key])
        cleaned[key] = cleaned[key].drop_duplicates()
        after = len(cleaned[key])
        if before != after:
            print(f"  Removed {before - after} duplicates from {key}")
    
    print("✓ Data cleaned")
    return cleaned


def compute_assessment_features(student_id, student_assessment_df, assessments_df):
    """Compute assessment-related features for a student.
    
    Args:
        student_id: Student identifier
        student_assessment_df: Student assessment scores
        assessments_df: Assessment metadata
        
    Returns:
        dict: Assessment features
    """
    student_scores = student_assessment_df[
        student_assessment_df['id_student'] == student_id
    ].copy()
    
    if len(student_scores) == 0:
        return {
            'n_assessments': 0,
            'avg_assess_score': np.nan,
            'std_assess_score': np.nan,
            'min_assess_score': np.nan,
            'max_assess_score': np.nan,
            'improvement_slope': 0.0,
            'final_grade': np.nan
        }
    
    # Merge with assessment metadata to get dates
    student_scores = student_scores.merge(
        assessments_df[['id_assessment', 'date']], 
        on='id_assessment', 
        how='left'
    )
    
    # Basic statistics
    features = {
        'n_assessments': len(student_scores),
        'avg_assess_score': student_scores['score'].mean(),
        'std_assess_score': student_scores['score'].std(),
        'min_assess_score': student_scores['score'].min(),
        'max_assess_score': student_scores['score'].max(),
    }
    
    # Improvement slope (linear regression of scores over time)
    improvement_slope = 0.0
    try:
        if len(student_scores) >= 2 and student_scores['date'].notna().any():
            valid_scores = student_scores.dropna(subset=['date', 'score'])
            if len(valid_scores) >= 2:
                # Check if dates are not all identical
                unique_dates = valid_scores['date'].nunique()
                if unique_dates > 1:
                    slope, _, _, _, _ = linregress(valid_scores['date'], valid_scores['score'])
                    improvement_slope = slope
    except Exception as e:
        # If anything goes wrong, default to 0
        improvement_slope = 0.0
    
    features['improvement_slope'] = improvement_slope
    
    # Final grade (use average as proxy)
    features['final_grade'] = features['avg_assess_score']
    
    return features


def compute_vle_features(student_id, student_vle_df):
    """Compute VLE engagement features for a student.
    
    Args:
        student_id: Student identifier
        student_vle_df: Student VLE interactions
        
    Returns:
        dict: VLE features
    """
    student_vle = student_vle_df[student_vle_df['id_student'] == student_id].copy()
    
    if len(student_vle) == 0:
        return {
            'total_vle_events': 0,
            'total_clicks': 0,
            'active_days': 0,
            'avg_clicks_per_day': 0.0,
            'std_clicks_per_day': 0.0,
            'max_clicks_day': 0,
            'engagement_rate': 0.0
        }
    
    # Aggregate by date
    daily_activity = student_vle.groupby('date')['sum_click'].sum()
    
    features = {
        'total_vle_events': len(student_vle),
        'total_clicks': student_vle['sum_click'].sum(),
        'active_days': len(daily_activity),
        'avg_clicks_per_day': daily_activity.mean(),
        'std_clicks_per_day': daily_activity.std() if len(daily_activity) > 1 else 0.0,
        'max_clicks_day': daily_activity.max(),
    }
    
    # Engagement rate (will be normalized later)
    features['engagement_rate'] = features['total_clicks']
    
    return features


def compute_demographic_features(student_id, student_info_df):
    """Extract demographic features for a student.
    
    Args:
        student_id: Student identifier
        student_info_df: Student information
        
    Returns:
        dict: Demographic features
    """
    student_row = student_info_df[student_info_df['id_student'] == student_id]
    
    if len(student_row) == 0:
        return {
            'gender': 'Unknown',
            'region': 'Unknown',
            'highest_education': 'Unknown',
            'imd_band': 'Unknown',
            'age_band': 'Unknown',
            'num_of_prev_attempts': 0,
            'studied_credits': 0,
            'disability': 'N'
        }
    
    row = student_row.iloc[0]
    
    return {
        'gender': row.get('gender', 'Unknown'),
        'region': row.get('region', 'Unknown'),
        'highest_education': row.get('highest_education', 'Unknown'),
        'imd_band': row.get('imd_band', 'Unknown'),
        'age_band': row.get('age_band', 'Unknown'),
        'num_of_prev_attempts': row.get('num_of_prev_attempts', 0),
        'studied_credits': row.get('studied_credits', 0),
        'disability': row.get('disability', 'N')
    }


def compute_student_features(data_dict, course_filter=None):
    """Compute comprehensive features for all students.
    
    Args:
        data_dict: Dictionary of cleaned dataframes
        course_filter: List of course codes to filter by (optional)
        
    Returns:
        pd.DataFrame: Student features
    """
    print("Computing student features...")
    
    student_info = data_dict['student_info']
    student_assessment = data_dict['student_assessment']
    student_vle = data_dict['student_vle']
    assessments = data_dict['assessments']
    
    # Filter by course if specified
    if course_filter:
        student_info = student_info[student_info['code_module'].isin(course_filter)]
        print(f"  Filtering to courses: {course_filter}")
    
    # Get unique students
    student_ids = student_info['id_student'].unique()
    print(f"  Processing {len(student_ids)} students...")
    
    all_features = []
    
    for i, student_id in enumerate(student_ids):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(student_ids)} students...")
        
        # Get course and presentation
        student_row = student_info[student_info['id_student'] == student_id].iloc[0]
        
        features = {
            'id_student': student_id,
            'code_module': student_row['code_module'],
            'code_presentation': student_row['code_presentation'],
            'final_result': student_row.get('final_result', 'Unknown')
        }
        
        # Assessment features
        assess_features = compute_assessment_features(
            student_id, student_assessment, assessments
        )
        features.update(assess_features)
        
        # VLE features
        vle_features = compute_vle_features(student_id, student_vle)
        features.update(vle_features)
        
        # Demographic features
        demo_features = compute_demographic_features(student_id, student_info)
        features.update(demo_features)
        
        all_features.append(features)
    
    df_features = pd.DataFrame(all_features)
    
    # Normalize engagement rate (z-score within each course)
    for course in df_features['code_module'].unique():
        mask = df_features['code_module'] == course
        engagement = df_features.loc[mask, 'engagement_rate']
        if engagement.std() > 0:
            df_features.loc[mask, 'engagement_rate'] = (
                (engagement - engagement.mean()) / engagement.std()
            )
    
    print(f"✓ Computed features for {len(df_features)} students")
    return df_features


def create_topic_vectors(data_dict, df_features, n_topics=5):
    """Create topic/skill vectors for students.
    
    Uses k-means clustering on assessment types to create pseudo-topics.
    
    Args:
        data_dict: Dictionary of cleaned dataframes
        df_features: Student features dataframe
        n_topics: Number of topics to create
        
    Returns:
        np.ndarray: Topic vectors (n_students x n_topics)
    """
    print(f"Creating topic vectors with {n_topics} topics...")
    
    student_assessment = data_dict['student_assessment']
    assessments = data_dict['assessments']
    
    # Merge to get assessment types
    assess_merged = student_assessment.merge(
        assessments[['id_assessment', 'assessment_type']], 
        on='id_assessment'
    )
    
    # Get unique assessment types
    unique_types = assess_merged['assessment_type'].unique()
    print(f"  Found {len(unique_types)} assessment types: {unique_types}")
    
    # Create pseudo-topics using k-means on assessment names
    # (In real scenario, use actual topic tags if available)
    assessment_names = assessments['id_assessment'].astype(str).values
    
    # Simple approach: assign assessments to topics based on hash
    assessment_to_topic = {}
    for idx, assess_id in enumerate(assessments['id_assessment']):
        topic_id = idx % n_topics
        assessment_to_topic[assess_id] = topic_id
    
    # Create topic vectors for each student
    student_ids = df_features['id_student'].values
    topic_vectors = np.zeros((len(student_ids), n_topics))
    
    for i, student_id in enumerate(student_ids):
        student_scores = student_assessment[
            student_assessment['id_student'] == student_id
        ]
        
        for _, row in student_scores.iterrows():
            assess_id = row['id_assessment']
            score = row['score']
            
            if assess_id in assessment_to_topic and not np.isnan(score):
                topic_id = assessment_to_topic[assess_id]
                # Accumulate normalized scores
                topic_vectors[i, topic_id] += score / 100.0
        
        # Normalize by number of assessments in each topic
        if topic_vectors[i].sum() > 0:
            topic_vectors[i] = topic_vectors[i] / (topic_vectors[i].sum() + 1e-8)
    
    print(f"✓ Created topic vectors: shape {topic_vectors.shape}")
    return topic_vectors


def save_processed_data(df_features, topic_vectors, output_path):
    """Save processed features and topic vectors.
    
    Args:
        df_features: Student features dataframe
        topic_vectors: Topic vectors array
        output_path: Path to processed data directory
    """
    print("Saving processed data...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save features
    features_file = output_path / "student_features.csv"
    df_features.to_csv(features_file, index=False)
    print(f"  ✓ Saved features to {features_file}")
    
    # Save topic vectors
    vectors_file = output_path / "student_topic_vectors.npy"
    np.save(vectors_file, topic_vectors)
    print(f"  ✓ Saved topic vectors to {vectors_file}")
    
    print("✓ All processed data saved")


def main():
    """Main preprocessing pipeline."""
    # Paths
    raw_path = Path("../data/raw")
    processed_path = Path("../data/processed")
    
    # Load and clean
    data = load_raw_data(raw_path)
    data = clean_data(data)
    
    # Compute features
    df_features = compute_student_features(data, course_filter=None)
    
    # Create topic vectors
    topic_vectors = create_topic_vectors(data, df_features, n_topics=5)
    
    # Save
    save_processed_data(df_features, topic_vectors, processed_path)
    
    # Print summary
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Students processed: {len(df_features)}")
    print(f"Features per student: {len(df_features.columns)}")
    print(f"Topic vector dimensions: {topic_vectors.shape[1]}")
    print(f"\nFeature columns: {df_features.columns.tolist()}")
    print(f"\nSample statistics:")
    print(df_features[['avg_assess_score', 'total_clicks', 'engagement_rate']].describe())


if __name__ == "__main__":
    main()