#!/usr/bin/env python3
"""
Create standardized key-pair scoring tables from individual score files

Takes individual key-pair score CSV files and prepares unified tables with normalized scores.
Creates both detailed component tables and composite score tables.

Input files expected:
- keypair_time_scores.csv (key_pair, time_setup, time_interval, time_return, time_total)
- keypair_comfort_scores.csv (key_pair, comfort_score)  
- keypair_distance_scores.csv (key_pair, distance_setup, distance_interval, distance_return, distance_total)
- keypair_dvorak7_scores.csv (key_pair, dvorak7_score)
- individual Dvorak-7 component files
- engram_2key_scores_avg4.csv (key_pair, engram_avg4_score)
- individual Engram component files

Output:
- tables/scores_2key_detailed.csv: All individual components with normalized versions
- tables/scores_2key_composite.csv: Composite scores (engram_avg4, dvorak7, time, distance, comfort)

Usage:
    python prep_scoring_tables.py --input-dir ../tables/
    python prep_scoring_tables.py --input-dir ../tables/ --verbose
"""

import sys
import argparse
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def get_universal_normalization_ranges():
    """
    Define universal normalization ranges for cross-dataset comparability.
    These ranges should be consistent across all datasets and scoring runs.
    Refer to the min/max values in the tables/keypair_*_scores.csv files.
    """
    return {
        # Distance components (in mm) - based on physical keyboard dimensions
        'distance_setup': (0, 35),       # 0 to max reasonable setup distance
        'distance_interval': (0, 55),    # 0 to max reasonable interval distance  
        'distance_return': (0, 70),      # 0 to max reasonable return distance
        'distance_total': (0, 135),      # 0 to max reasonable total distance
        'distance_score': (0, 135),      # Legacy single distance
        
        # Time components (in ms) - based on typical typing ranges
        'time_setup': (200, 400),        # 0 to max reasonable setup time
        'time_interval': (100, 450),     # 0 to max reasonable interval time
        'time_return': (200, 400),       # 0 to max reasonable return time
        'time_total': (570, 1250),       # 0 to max reasonable total time
        'time_score': (570, 1250),       # Legacy single time
        
        # Comfort scores - adjusted for negative values (penalty-based scoring)
        'comfort_score': (-1.5, 0.0),    # Accommodate negative penalty scores

        # Comfort-combo scores - combination of key and key-pair comfort scores
        'comfort_combo_score': (0, 100), # Assuming 0-100 scale

        # Engram scores and components
        'engram_avg4_score': (0, 1),    # Engram avg4 score range (average of 4 components)
        'engram_key_preference': (0, 1),       
        'engram_row_separation': (0, 1),      
        'engram_same_row': (0, 1),          
        'engram_outside': (0, 1),          
        'engram_same_finger': (0, 1),          

        # Dvorak-7 scores and components
        'dvorak7_score': (0, 7),          # Overall Dvorak-7 score range (sum of 7 components)
        'dvorak7_distribution': (0, 1),   # Individual component ranges
        'dvorak7_strength': (0, 1),      
        'dvorak7_home': (0, 1),      
        'dvorak7_vspan': (0, 1),    
        'dvorak7_columns': (0, 1),      
        'dvorak7_remote': (0, 1),          
        'dvorak7_inward': (0, 1),       
    }


def normalize_with_universal_range(scores: np.ndarray, score_name: str, verbose: bool = True) -> np.ndarray:
    """
    Normalize scores using universal ranges for cross-dataset comparability.
    """
    universal_ranges = get_universal_normalization_ranges()
    
    if score_name not in universal_ranges:
        if verbose:
            print(f"  {score_name}: No universal range defined, using adaptive normalization")
        return detect_and_normalize_distribution(scores, score_name, verbose)
    
    min_val, max_val = universal_ranges[score_name]
    
    # Clip scores to the universal range
    clipped_scores = np.clip(scores, min_val, max_val)
    
    # Normalize to [0, 1] using the universal range
    if max_val == min_val:
        normalized = np.zeros_like(clipped_scores)
    else:
        normalized = (clipped_scores - min_val) / (max_val - min_val)
    
    if verbose:
        clipped_count = np.sum((scores < min_val) | (scores > max_val))
        print(f"  {score_name}: Universal range [{min_val}, {max_val}], {clipped_count} values clipped")
        print(f"    Original range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"    Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return normalized


def detect_and_normalize_distribution(scores: np.ndarray, name: str = '', verbose: bool = True) -> np.ndarray:
    """
    Automatically detect distribution type and apply appropriate normalization.
    Returns scores normalized to [0,1] range.
    """
    # Handle empty or constant arrays
    if len(scores) == 0 or np.all(scores == scores[0]):
        return np.zeros_like(scores)

    # Get basic statistics
    non_zeros = scores[scores != 0]
    if len(non_zeros) == 0:
        return np.zeros_like(scores)
        
    min_nonzero = np.min(non_zeros)
    max_val = np.max(scores)
    mean = np.mean(non_zeros)
    median = np.median(non_zeros)
    skew = np.mean(((non_zeros - mean) / np.std(non_zeros)) ** 3) if np.std(non_zeros) > 0 else 0
    
    # Calculate ratio between consecutive sorted values
    sorted_nonzero = np.sort(non_zeros)
    ratios = sorted_nonzero[1:] / sorted_nonzero[:-1] if len(sorted_nonzero) > 1 else np.array([1.0])
    
    # Detect distribution type and apply appropriate normalization
    if len(scores[scores == 0]) / len(scores) > 0.3:
        # Sparse distribution with many zeros
        if verbose:
            print(f"  {name}: Sparse distribution detected")
        norm_scores = np.sqrt(scores)
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))

    elif skew > 2 or np.median(ratios) > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        if verbose:
            print(f"  {name}: Heavy-tailed distribution detected")
        norm_scores = np.sqrt(np.abs(scores))
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))
        
    elif abs(mean - median) / mean < 0.1:
        # Roughly symmetric distribution
        if verbose:
            print(f"  {name}: Symmetric distribution detected")
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
    else:
        # Default to robust scaling
        if verbose:
            print(f"  {name}: Using robust scaling")
        q1, q99 = np.percentile(scores, [1, 99])
        scaled = (scores - q1) / (q99 - q1)
        return np.clip(scaled, 0, 1)


def load_score_file(filepath: Path, score_column: str, verbose: bool = False) -> pd.DataFrame:
    """Load a score CSV file and return DataFrame with key_pair and score columns."""
    if not filepath.exists():
        raise FileNotFoundError(f"Score file not found: {filepath}")
    
    try:
        # CRITICAL: Prevent pandas from converting 'NA' to NaN by specifying keep_default_na=False
        # and setting na_values to only handle truly empty values
        df = pd.read_csv(filepath, 
                        dtype={'key_pair': str}, 
                        keep_default_na=False,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan'])
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    if 'key_pair' not in df.columns:
        raise ValueError(f"Missing 'key_pair' column in {filepath}")
    
    if score_column not in df.columns:
        if verbose:
            print(f"  ⚠ Missing '{score_column}' column in {filepath.name}")
            print(f"    Available columns: {list(df.columns)}")
        return None
    
    # Keep only key_pair and score columns
    result_df = df[['key_pair', score_column]].copy()
    
    # Clean the data: remove rows with empty, null, or whitespace-only key_pair values
    # BUT preserve literal 'NA' strings
    initial_rows = len(result_df)
    
    # Remove rows where key_pair is actually null/empty (but keep 'NA' strings)
    result_df = result_df[result_df['key_pair'].notna()]
    result_df = result_df[result_df['key_pair'].astype(str).str.strip() != '']
    
    # Remove rows where key_pair length is not exactly 2 (assuming all key pairs should be 2 characters)
    result_df = result_df[result_df['key_pair'].astype(str).str.len() == 2]
    
    cleaned_rows = len(result_df)
    
    if verbose and initial_rows != cleaned_rows:
        print(f"  Cleaned {initial_rows - cleaned_rows} invalid rows from {filepath.name}")
    
    if result_df.empty:
        raise ValueError(f"No valid data remaining after cleaning: {filepath}")
    
    # Debug: Check if 'NA' survived
    if verbose and 'NA' in result_df['key_pair'].values:
        print(f"  ✓ 'NA' key pair preserved in {filepath.name}")
    
    if verbose:
        print(f"Loaded {len(result_df)} rows from {filepath.name}")
        print(f"  Score range: {result_df[score_column].min():.6f} - {result_df[score_column].max():.6f}")
    
    return result_df


def load_detailed_file(filepath: Path, required_columns: List[str], verbose: bool = False) -> pd.DataFrame:
    """Load detailed CSV file with multiple components."""
    if not filepath.exists():
        raise FileNotFoundError(f"Detailed file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, 
                        dtype={'key_pair': str}, 
                        keep_default_na=False,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan'])
    except Exception as e:
        raise ValueError(f"Error reading detailed CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"Detailed CSV file is empty: {filepath}")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in {filepath}: {missing_columns}")
    
    # Keep only required columns
    result_df = df[required_columns].copy()
    
    # Clean the data
    initial_rows = len(result_df)
    
    # Remove rows where key_pair is actually null/empty (but keep 'NA' strings)
    result_df = result_df[result_df['key_pair'].notna()]
    result_df = result_df[result_df['key_pair'].astype(str).str.strip() != '']
    
    # Remove rows where key_pair length is not exactly 2
    result_df = result_df[result_df['key_pair'].astype(str).str.len() == 2]
    
    cleaned_rows = len(result_df)
    
    if verbose and initial_rows != cleaned_rows:
        print(f"  Cleaned {initial_rows - cleaned_rows} invalid rows from {filepath.name}")
    
    if result_df.empty:
        raise ValueError(f"No valid data remaining after cleaning: {filepath}")
    
    if verbose:
        print(f"Loaded {len(result_df)} rows from {filepath.name}")
        component_cols = [col for col in required_columns if col != 'key_pair']
        for col in component_cols:
            print(f"  {col} range: {result_df[col].min():.6f} - {result_df[col].max():.6f}")
    
    return result_df


def create_composite_scores(unified_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Create composite scores from detailed components."""
    
    if verbose:
        print("\n🎯 Creating composite scores from detailed components...")
    
    composite_df = unified_df[['key_pair']].copy()
    
    # Comfort-combo composite (if available)
    if 'comfort_combo_score' in unified_df.columns:
        composite_df['comfort_combo_score'] = unified_df['comfort_combo_score']
        if verbose:
            print("  ✓ Comfort-combo: Using direct score")
    
    # Engram composite
    engram_components = [col for col in unified_df.columns if col.startswith('engram_') and col != 'engram_avg4_score']
    if len(engram_components) > 1:
        # Create weighted composite from components
        composite_df['engram_avg4_score'] = unified_df[engram_components].mean(axis=1)
        if verbose:
            print(f"  ✅ Engram: Composite from {len(engram_components)} components")
    elif 'engram_avg4_score' in unified_df.columns:
        composite_df['engram_avg4_score'] = unified_df['engram_avg4_score']
        if verbose:
            print("  ✅ Engram: Using direct avg4 score")
                
    # Dvorak-7 composite
    dvorak7_components = [col for col in unified_df.columns if col.startswith('dvorak7_') and col != 'dvorak7_score']
    if len(dvorak7_components) > 1:
        # Create weighted composite from components
        composite_df['dvorak7_score'] = unified_df[dvorak7_components].mean(axis=1)
        if verbose:
            print(f"  ✓ Dvorak-7: Composite from {len(dvorak7_components)} components")
    elif 'dvorak7_score' in unified_df.columns:
        composite_df['dvorak7_score'] = unified_df['dvorak7_score']
        if verbose:
            print("  ✓ Dvorak-7: Using direct score")
    
    # Time composite
    time_components = [col for col in unified_df.columns if col.startswith('time_') and col != 'time_score']
    if 'time_total' in unified_df.columns:
        composite_df['time_score'] = unified_df['time_total']
        if verbose:
            print("  ✓ Time: Using time_total")
    elif len(time_components) >= 3:
        # Sum the components
        composite_df['time_score'] = unified_df[time_components].sum(axis=1)
        if verbose:
            print(f"  ✓ Time: Sum of {len(time_components)} components")
    elif 'time_score' in unified_df.columns:
        composite_df['time_score'] = unified_df['time_score']
        if verbose:
            print("  ✓ Time: Using legacy time_score")
    
    # Distance composite
    distance_components = [col for col in unified_df.columns if col.startswith('distance_') and col != 'distance_score']
    if 'distance_total' in unified_df.columns:
        composite_df['distance_score'] = unified_df['distance_total']
        if verbose:
            print("  ✓ Distance: Using distance_total")
    elif len(distance_components) >= 3:
        # Sum the components
        composite_df['distance_score'] = unified_df[distance_components].sum(axis=1)
        if verbose:
            print(f"  ✓ Distance: Sum of {len(distance_components)} components")
    elif 'distance_score' in unified_df.columns:
        composite_df['distance_score'] = unified_df['distance_score']
        if verbose:
            print("  ✓ Distance: Using legacy distance_score")
    
    # Comfort composite
    if 'comfort_score' in unified_df.columns:
        composite_df['comfort_score'] = unified_df['comfort_score']
        if verbose:
            print("  ✓ Comfort: Using comfort_score")
    
    return composite_df


def create_key_comfort_scores(input_dir: str, verbose: bool = False) -> None:
    """Create individual key comfort scores table from same-key bigrams."""
    
    input_path = Path(input_dir)
    comfort_file = input_path / 'keypair_comfort_scores.csv'
    
    if not comfort_file.exists():
        if verbose:
            print(f"Warning: {comfort_file} not found, skipping key comfort scores extraction")
        return
    
    try:
        # CRITICAL: Prevent pandas from converting 'NA' to NaN
        df = pd.read_csv(comfort_file, 
                        dtype={'key_pair': str},
                        keep_default_na=False,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan'])
    except Exception as e:
        if verbose:
            print(f"Error reading comfort scores file: {e}")
        return
    
    if 'key_pair' not in df.columns or 'comfort_score' not in df.columns:
        if verbose:
            print("Warning: comfort scores file missing required columns")
        return
    
    # Clean the data first: remove rows with empty, null, or invalid key_pair values
    # BUT preserve literal 'NA' strings
    initial_rows = len(df)
    
    # Remove rows where key_pair is actually null/empty (but keep 'NA' strings)
    df = df[df['key_pair'].notna()]
    df = df[df['key_pair'].astype(str).str.strip() != '']
    
    # Remove rows where key_pair length is not exactly 2
    df = df[df['key_pair'].astype(str).str.len() == 2]
    
    cleaned_rows = len(df)
    
    if verbose and initial_rows != cleaned_rows:
        print(f"Cleaned {initial_rows - cleaned_rows} invalid rows from comfort scores")
    
    if df.empty:
        if verbose:
            print("Warning: No valid data remaining after cleaning comfort scores")
        return
    
    # Debug: Check if 'NA' survived
    if verbose and 'NA' in df['key_pair'].values:
        print(f"  ✓ 'NA' key pair preserved in comfort scores")
    
    # Filter for same-key bigrams (e.g., "AA", ";;", "//")
    same_key_rows = df[df['key_pair'].str[0] == df['key_pair'].str[1]].copy()
    
    if len(same_key_rows) == 0:
        if verbose:
            print("Warning: No same-key bigrams found in comfort scores")
        return
    
    # Extract single key from key_pair
    same_key_rows['key'] = same_key_rows['key_pair'].str[0]
    
    # Apply normalization to comfort scores
    comfort_scores = same_key_rows['comfort_score'].values
    normalized_scores = normalize_with_universal_range(
        comfort_scores, 
        'comfort_score', 
        verbose
    )
    
    # Create output DataFrame
    key_comfort_df = pd.DataFrame({
        'key': same_key_rows['key'],
        'comfort_score': normalized_scores
    })
    
    # Sort by key for consistent output
    key_comfort_df = key_comfort_df.sort_values('key').reset_index(drop=True)
    
    # Round to 6 decimal places
    key_comfort_df['comfort_score'] = key_comfort_df['comfort_score'].round(6)
    
    # Save to output directory
    output_file = input_path / 'key_comfort_scores.csv'
    key_comfort_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\nCreated key comfort scores table: {output_file}")
        print(f"  Keys: {len(key_comfort_df)} individual keys")
        print(f"  Score range: {key_comfort_df['comfort_score'].min():.6f} - {key_comfort_df['comfort_score'].max():.6f}")
        print(f"  Mean score: {key_comfort_df['comfort_score'].mean():.6f}")

        
def create_unified_score_tables(input_dir: str, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create both detailed and composite unified score tables."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Define expected input files and their score columns
    simple_score_files = {
        'comfort_score': 'keypair_comfort_scores.csv',
        'dvorak7_score': 'keypair_dvorak7_scores.csv',
        'engram_avg4_score': 'engram_2key_scores_avg4.csv',
        'comfort_combo_score': 'keypair_comfort_combo_scores.csv',
        # Individual Engram criteria (fixed filenames)
        'engram_key_preference': 'engram_2key_scores_key_preference.csv',
        'engram_row_separation': 'engram_2key_scores_row_separation.csv',
        'engram_same_row': 'engram_2key_scores_same_row.csv',
        'engram_outside': 'engram_2key_scores_outside.csv',
        'engram_same_finger': 'engram_2key_scores_same_finger.csv',
        # Individual Dvorak-7 criteria
        'dvorak7_distribution': 'keypair_dvorak7_distribution_scores.csv',
        'dvorak7_strength': 'keypair_dvorak7_strength_scores.csv',
        'dvorak7_home': 'keypair_dvorak7_home_scores.csv',
        'dvorak7_vspan': 'keypair_dvorak7_vspan_scores.csv',
        'dvorak7_columns': 'keypair_dvorak7_columns_scores.csv',
        'dvorak7_remote': 'keypair_dvorak7_remote_scores.csv',
        'dvorak7_inward': 'keypair_dvorak7_inward_scores.csv',
        # Legacy scores (for backward compatibility)
        'distance_score': 'keypair_distance_scores.csv',
        'time_score': 'keypair_time_scores.csv'
    }

    if verbose:
        print(f"Loading score files from: {input_dir}")
    
    # Load all simple score files
    loaded_scores = {}
    
    for score_name, filename in simple_score_files.items():
        filepath = input_path / filename
        
        if filepath.exists():
            df = load_score_file(filepath, score_name, verbose)
            if df is not None:
                loaded_scores[score_name] = df
                if verbose:
                    print(f"  ✓ Loaded {score_name} from {filename}")
            else:
                if verbose:
                    print(f"  ⚠ Skipping {score_name} from {filename} (column not found)")
        else:
            if verbose:
                print(f"  ⚠ Missing {filename}, skipping {score_name}")
    
    # Try to load detailed distance scores (preferred over legacy single distance)
    detailed_distance_file = input_path / 'keypair_distance_scores.csv'
    detailed_distance_df = None
    
    if detailed_distance_file.exists():
        try:
            required_cols = ['key_pair', 'distance_setup', 'distance_interval', 'distance_return', 'distance_total']
            detailed_distance_df = load_detailed_file(detailed_distance_file, required_cols, verbose)
            if verbose:
                print(f"  ✓ Loaded detailed distance scores from keypair_distance_scores.csv")
                print(f"    Components: distance_setup, distance_interval, distance_return, distance_total")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading detailed distance file: {e}")
    else:
        print(f"  ⚠ No distance score files found")
    
    # Try to load detailed time scores (preferred over legacy single time)
    detailed_time_file = input_path / 'keypair_time_scores.csv'
    detailed_time_df = None
    
    if detailed_time_file.exists():
        try:
            required_cols = ['key_pair', 'time_setup', 'time_interval', 'time_return', 'time_total']
            detailed_time_df = load_detailed_file(detailed_time_file, required_cols, verbose)
            if verbose:
                print(f"  ✓ Loaded detailed time scores from keypair_time_scores.csv")
                print(f"    Components: time_setup, time_interval, time_return, time_total")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading detailed time file: {e}")
    else:
        print(f"  ⚠ No detailed time scores found")
    
    # Check if we have any scores to work with
    if not loaded_scores and detailed_distance_df is None and detailed_time_df is None:
        raise ValueError("No valid score files found")
    
    # Start with the first available dataframe
    unified_df = None
    
    if detailed_distance_df is not None:
        unified_df = detailed_distance_df.copy()
        # Remove legacy distance_score to avoid confusion
        if 'distance_score' in loaded_scores:
            del loaded_scores['distance_score']
            if verbose:
                print("  → Using detailed distance components instead of legacy distance_score")
    
    if detailed_time_df is not None:
        if unified_df is not None:
            unified_df = unified_df.merge(detailed_time_df, on='key_pair', how='outer')
        else:
            unified_df = detailed_time_df.copy()
        # Remove legacy time_score to avoid confusion
        if 'time_score' in loaded_scores:
            del loaded_scores['time_score']
            if verbose:
                print("  → Using detailed time components instead of legacy time_score")
    
    if unified_df is None and loaded_scores:
        first_score_name = list(loaded_scores.keys())[0]
        unified_df = loaded_scores[first_score_name].copy()
        del loaded_scores[first_score_name]
    elif unified_df is None:
        raise ValueError("No valid score data found")
    
    # Merge remaining dataframes
    for score_name, df in loaded_scores.items():
        unified_df = unified_df.merge(df, on='key_pair', how='outer')
    
    if verbose:
        print(f"\nUnified dataframe has {len(unified_df)} rows")
        score_columns = [col for col in unified_df.columns if col != 'key_pair']
        print(f"Score columns: {score_columns}")
    
    # Apply normalization to each score column
    if verbose:
        print("\nApplying universal normalization to score columns:")
    
    # Get all score columns (excluding key_pair)
    score_columns = [col for col in unified_df.columns if col != 'key_pair']
    
    for score_name in score_columns:
        if score_name in unified_df.columns:
            # Fill any missing values with column mean
            col_values = unified_df[score_name].fillna(unified_df[score_name].mean())
            
            # Apply universal normalization for cross-dataset comparability
            normalized_values = normalize_with_universal_range(
                col_values.values, 
                score_name, 
                verbose
            )
            
            # Add normalized column
            normalized_col_name = f"{score_name}_normalized"
            unified_df[normalized_col_name] = normalized_values
            
            if verbose:
                print(f"    Added {normalized_col_name} (range: {normalized_values.min():.6f} - {normalized_values.max():.6f})")
    
    # Create composite scores
    composite_df = create_composite_scores(unified_df, verbose)
    
    # Apply normalization to composite scores
    if verbose:
        print("\nApplying universal normalization to composite scores:")
    
    composite_score_columns = [col for col in composite_df.columns if col != 'key_pair']
    for score_name in composite_score_columns:
        if score_name in composite_df.columns:
            # Fill any missing values with column mean
            col_values = composite_df[score_name].fillna(composite_df[score_name].mean())
            
            # Apply universal normalization for cross-dataset comparability
            normalized_values = normalize_with_universal_range(
                col_values.values, 
                score_name, 
                verbose
            )
            
            # Add normalized column
            normalized_col_name = f"{score_name}_normalized"
            composite_df[normalized_col_name] = normalized_values
            
            if verbose:
                print(f"    Added {normalized_col_name} (range: {normalized_values.min():.6f} - {normalized_values.max():.6f})")
    
    # Sort both dataframes by key_pair for consistent output
    unified_df = unified_df.sort_values('key_pair').reset_index(drop=True)
    composite_df = composite_df.sort_values('key_pair').reset_index(drop=True)
    
    return unified_df, composite_df


def save_score_tables(detailed_df: pd.DataFrame, composite_df: pd.DataFrame, input_dir: str, verbose: bool = False):
    """Save both detailed and composite score tables."""
    
    input_path = Path(input_dir)
    
    # Save detailed scores
    detailed_output_columns = ['key_pair']
    
    # Add original score columns
    original_columns = [col for col in detailed_df.columns if col != 'key_pair' and not col.endswith('_normalized')]
    detailed_output_columns.extend(sorted(original_columns))
    
    # Add normalized score columns
    normalized_columns = [col for col in detailed_df.columns if col.endswith('_normalized')]
    detailed_output_columns.extend(sorted(normalized_columns))
    
    detailed_output_df = detailed_df[detailed_output_columns]
    
    # Format floating point numbers to 6 decimal places
    float_columns = [col for col in detailed_output_df.columns if col != 'key_pair']
    for col in float_columns:
        detailed_output_df[col] = detailed_output_df[col].round(6)
    
    detailed_file = input_path / 'scores_2key_detailed.csv'
    detailed_output_df.to_csv(detailed_file, index=False)
    
    # Save composite scores
    composite_output_columns = ['key_pair']
    
    # Add original composite columns
    original_composite_columns = [col for col in composite_df.columns if col != 'key_pair' and not col.endswith('_normalized')]
    composite_output_columns.extend(sorted(original_composite_columns))
    
    # Add normalized composite columns
    normalized_composite_columns = [col for col in composite_df.columns if col.endswith('_normalized')]
    composite_output_columns.extend(sorted(normalized_composite_columns))
    
    composite_output_df = composite_df[composite_output_columns]
    
    # Format floating point numbers to 6 decimal places
    float_columns = [col for col in composite_output_df.columns if col != 'key_pair']
    for col in float_columns:
        composite_output_df[col] = composite_output_df[col].round(6)
    
    composite_file = input_path / 'scores_2key_composite.csv'
    composite_output_df.to_csv(composite_file, index=False)
    
    if verbose:
        print(f"\nSaved detailed scoring table to: {detailed_file}")
        print(f"  Table contains {len(detailed_output_df)} rows and {len(detailed_output_columns)} columns")
        print(f"  Detailed columns: {detailed_output_columns}")
        
        print(f"\nSaved composite scoring table to: {composite_file}")
        print(f"  Table contains {len(composite_output_df)} rows and {len(composite_output_columns)} columns")
        print(f"  Composite columns: {composite_output_columns}")
        
        # Show sample statistics for composite scores
        print(f"\nComposite score statistics:")
        original_composite_columns = [col for col in composite_df.columns if col != 'key_pair' and not col.endswith('_normalized')]
        for score_name in original_composite_columns:
            if score_name in composite_output_df.columns:
                values = composite_output_df[score_name]
                print(f"  {score_name}: {values.min():.6f} - {values.max():.6f} (mean: {values.mean():.6f})")


def validate_input_directory(input_dir: str) -> None:
    """Validate that input directory exists and contains expected files."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Check for at least one expected file
    expected_files = [
        'keypair_time_scores.csv',
        'keypair_comfort_scores.csv', 
        'keypair_distance_scores.csv',
        'keypair_dvorak7_scores.csv',
        'engram_2key_scores.csv'
    ]
    
    found_files = [f for f in expected_files if (input_path / f).exists()]
    
    if not found_files:
        available_files = [f.name for f in input_path.glob('*.csv')]
        raise ValueError(
            f"No expected score files found in {input_dir}. "
            f"Expected: {expected_files}. "
            f"Available CSV files: {available_files}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create standardized scoring tables from individual score files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prep_scoring_tables.py --input-dir ../tables/
    python prep_scoring_tables.py --input-dir ../tables/ --verbose

Input files expected in input directory:
    - keypair_time_scores.csv (key_pair, time_setup, time_interval, time_return, time_total)
    - keypair_comfort_scores.csv (key_pair, comfort_score)
    - keypair_distance_scores.csv (key_pair, distance_setup, distance_interval, distance_return, distance_total)
    - keypair_dvorak7_scores.csv (key_pair, dvorak7_score)
    - engram_2key_scores.csv (key_pair, engram_score)
    - Individual engram component files (optional)
    - Individual dvorak7 component files (optional)

Creates three standardized output files:
    - tables/scores_2key_detailed.csv: All individual components
        - key_pair: Two-character key pair (e.g., "QW", "AS")
        - Original detailed score columns (time_setup, time_interval, time_return, time_total, distance_setup, etc.)
        - Normalized score columns (*_normalized) with universal ranges for cross-dataset comparability (0-1 range)
    
    - tables/scores_2key_composite.csv: Composite scores for cleaner visualization
        - key_pair: Two-character key pair
        - Composite scores (comfort_combo_score, engram_score, dvorak7_score, time_score, distance_score, comfort_score)
        - Normalized composite columns (*_normalized) with universal ranges
    
    - tables/key_comfort_scores.csv: Individual key comfort scores
        - key: Individual key character
        - comfort_score: Normalized comfort score for that key (extracted from same-key bigrams like "AA", ";;")

Normalization:
    Uses universal ranges for cross-dataset comparability:
    - Distance components: 0-35mm (setup), 0-55mm (interval), 0-70mm (return), 0-135mm (total)
    - Time components: 200-400ms (setup), 100-450ms (interval), 200-400ms (return), 570-1250ms (total)
    - Comfort scores: -1.5 to 0.0 range
    - Comfort-combo scores: 0-100 range
    - Engram avg4 scores: 0-1 (average of 4 core transition criteria)
    - Dvorak-7 scores: 0-7 (overall), 0-1 (components)
    
    Values outside universal ranges are clipped. This ensures normalized scores
    are comparable across different datasets and scoring runs.

Score Combinations:
    Time Scores:
        If keypair_time_scores.csv is available, it will use the detailed breakdown:
            - time_setup: Distance to position finger(s) for first key
            - time_interval: Time to move from first to second key
            - time_return: Time to return finger(s) to home positions
            - time_total: Sum of all components
        For composite, uses time_total (or falls back to legacy time_score)
    
    Distance Scores:
        If keypair_distance_scores.csv is available, it will use the detailed breakdown:
            - distance_setup, distance_interval, distance_return, distance_total
        For composite, uses distance_total (or falls back to legacy distance_score)
    
    Engram and Dvorak-7 Scores:
        If individual component files available, creates composite from components
        Otherwise uses direct engram_score and dvorak7_score

All floating point values are formatted to 6 decimal places.
Missing input files will be skipped with a warning.
The script uses universal normalization ranges for cross-dataset comparability.
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help="Directory containing the individual score CSV files"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed information during processing"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input directory first
        validate_input_directory(args.input_dir)
        
        # Create the unified tables (both detailed and composite)
        detailed_df, composite_df = create_unified_score_tables(
            args.input_dir,
            args.verbose
        )
        
        # Save both tables
        save_score_tables(detailed_df, composite_df, args.input_dir, args.verbose)
        
        # Create individual key comfort scores table
        create_key_comfort_scores(
            args.input_dir,
            args.verbose
        )
        
        print(f"\nSuccessfully created scoring tables:")
        print(f"  - Detailed scores: {Path(args.input_dir) / 'scores_2key_detailed.csv'}")
        print(f"  - Composite scores: {Path(args.input_dir) / 'scores_2key_composite.csv'}")
        print(f"  - Key comfort scores: {Path(args.input_dir) / 'key_comfort_scores.csv'}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())