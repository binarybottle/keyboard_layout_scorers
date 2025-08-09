#!/usr/bin/env python3
"""
prep_scoring_tables.py - Create standardized scoring tables from individual score files

Takes individual score CSV files and prepares unified tables with normalized scores.
Creates both key-pair scores and individual key scores.

Input files expected:
- keypair_time_scores.csv (key_pair, time_score)
- keypair_comfort_scores.csv (key_pair, comfort_score)  
- keypair_distance_scores.csv (key_pair, distance_score)
- keypair_dvorak9_scores.csv (key_pair, dvorak9_score)

Output:
- output/keypair_scores.csv: Unified key-pair scores with normalized versions (0-1 range)
- output/key_scores.csv: Individual key comfort scores extracted from same-key bigrams

Usage:
    python prep_scoring_tables.py --input-dir output/
    python prep_scoring_tables.py --input-dir output/ --verbose
"""

import sys
import argparse
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


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
        df = pd.read_csv(filepath, dtype={'key_pair': str})
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    if 'key_pair' not in df.columns:
        raise ValueError(f"Missing 'key_pair' column in {filepath}")
    
    if score_column not in df.columns:
        raise ValueError(f"Missing '{score_column}' column in {filepath}")
    
    # Keep only key_pair and score columns
    result_df = df[['key_pair', score_column]].copy()
    
    if verbose:
        print(f"Loaded {len(result_df)} rows from {filepath.name}")
        print(f"  Score range: {result_df[score_column].min():.6f} - {result_df[score_column].max():.6f}")
    
    return result_df


def create_key_comfort_scores(input_dir: str, verbose: bool = False) -> None:
    """Create individual key comfort scores table from same-key bigrams."""
    
    input_path = Path(input_dir)
    comfort_file = input_path / 'keypair_comfort_scores.csv'
    
    if not comfort_file.exists():
        if verbose:
            print(f"Warning: {comfort_file} not found, skipping key comfort scores extraction")
        return
    
    try:
        df = pd.read_csv(comfort_file, dtype={'key_pair': str})
    except Exception as e:
        if verbose:
            print(f"Error reading comfort scores file: {e}")
        return
    
    if 'key_pair' not in df.columns or 'comfort_score' not in df.columns:
        if verbose:
            print("Warning: comfort scores file missing required columns")
        return
    
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
    normalized_scores = detect_and_normalize_distribution(
        comfort_scores, 
        'key_comfort', 
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
    output_file = input_path / 'key_scores.csv'
    key_comfort_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\nCreated key comfort scores table: {output_file}")
        print(f"  Keys: {len(key_comfort_df)} individual keys")
        print(f"  Score range: {key_comfort_df['comfort_score'].min():.6f} - {key_comfort_df['comfort_score'].max():.6f}")
        print(f"  Mean score: {key_comfort_df['comfort_score'].mean():.6f}")


def create_unified_score_table(input_dir: str, verbose: bool = False) -> None:
    """Create unified key pair scoring table with normalized scores."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Define expected input files and their score columns
    score_files = {
        'comfort_score': 'keypair_comfort_scores.csv',
        'distance_score': 'keypair_distance_scores.csv',
        'time_score': 'keypair_time_scores.csv',
        'dvorak9_score': 'keypair_dvorak9_scores.csv',
        # Individual dvorak9 criteria
        'dvorak9_hands': 'keypair_dvorak9_hands_scores.csv',
        'dvorak9_fingers': 'keypair_dvorak9_fingers_scores.csv', 
        'dvorak9_skip_fingers': 'keypair_dvorak9_skip_fingers_scores.csv',
        'dvorak9_dont_cross_home': 'keypair_dvorak9_dont_cross_home_scores.csv',
        'dvorak9_same_row': 'keypair_dvorak9_same_row_scores.csv',
        'dvorak9_home_row': 'keypair_dvorak9_home_row_scores.csv',
        'dvorak9_columns': 'keypair_dvorak9_columns_scores.csv',
        'dvorak9_strum': 'keypair_dvorak9_strum_scores.csv',
        'dvorak9_strong_fingers': 'keypair_dvorak9_strong_fingers_scores.csv'
    }

    if verbose:
        print(f"Loading score files from: {input_dir}")
    
    # Load all score files
    dataframes = []
    loaded_scores = {}
    
    for score_name, filename in score_files.items():
        filepath = input_path / filename
        
        if filepath.exists():
            df = load_score_file(filepath, score_name, verbose)
            loaded_scores[score_name] = df
            if verbose:
                print(f"  ✓ Loaded {score_name} from {filename}")
        else:
            if verbose:
                print(f"  ⚠ Missing {filename}, skipping {score_name}")
    
    if not loaded_scores:
        raise ValueError("No valid score files found")
    
    # Start with the first loaded dataframe
    first_score_name = list(loaded_scores.keys())[0]
    unified_df = loaded_scores[first_score_name].copy()
    
    # Merge remaining dataframes
    for score_name, df in list(loaded_scores.items())[1:]:
        unified_df = unified_df.merge(df, on='key_pair', how='outer')
    
    if verbose:
        print(f"\nUnified dataframe has {len(unified_df)} rows")
        print(f"Score columns: {[col for col in unified_df.columns if col != 'key_pair']}")
    
    # Apply normalization to each score column
    if verbose:
        print("\nApplying smart normalization to score columns:")
    
    for score_name in loaded_scores.keys():
        if score_name in unified_df.columns:
            # Fill any missing values with column mean
            col_values = unified_df[score_name].fillna(unified_df[score_name].mean())
            
            # Apply smart normalization
            normalized_values = detect_and_normalize_distribution(
                col_values.values, 
                score_name, 
                verbose
            )
            
            # Add normalized column
            normalized_col_name = f"{score_name}_normalized"
            unified_df[normalized_col_name] = normalized_values
            
            if verbose:
                print(f"    Added {normalized_col_name} (range: {normalized_values.min():.6f} - {normalized_values.max():.6f})")
    
    # Sort by key_pair for consistent output
    unified_df = unified_df.sort_values('key_pair').reset_index(drop=True)
    
    # Prepare columns for output (key_pair first, then original scores, then normalized scores)
    output_columns = ['key_pair']
    
    # Add original score columns
    for score_name in loaded_scores.keys():
        if score_name in unified_df.columns:
            output_columns.append(score_name)
    
    # Add normalized score columns
    for score_name in loaded_scores.keys():
        normalized_col_name = f"{score_name}_normalized"
        if normalized_col_name in unified_df.columns:
            output_columns.append(normalized_col_name)
    
    # Write to CSV
    output_df = unified_df[output_columns]
    
    # Format floating point numbers to 6 decimal places
    float_columns = [col for col in output_df.columns if col != 'key_pair']
    for col in float_columns:
        output_df[col] = output_df[col].round(6)
    
    # Fixed output path
    output_file = input_path / 'keypair_scores.csv'
    output_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\nSaved unified scoring table to: {output_file}")
        print(f"Table contains {len(output_df)} rows and {len(output_columns)} columns")
        print(f"Columns: {output_columns}")
        
        # Show sample statistics
        print(f"\nSample statistics:")
        for score_name in loaded_scores.keys():
            if score_name in output_df.columns:
                values = output_df[score_name]
                print(f"  {score_name}: {values.min():.6f} - {values.max():.6f} (mean: {values.mean():.6f})")
            
            normalized_col = f"{score_name}_normalized"
            if normalized_col in output_df.columns:
                norm_values = output_df[normalized_col]
                print(f"  {normalized_col}: {norm_values.min():.6f} - {norm_values.max():.6f} (mean: {norm_values.mean():.6f})")


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
        'keypair_dvorak9_scores.csv'
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
    python prep_scoring_tables.py --input-dir output/
    python prep_scoring_tables.py --input-dir output/ --verbose

Input files expected in input directory:
    - keypair_time_scores.csv (key_pair, time_score)
    - keypair_comfort_scores.csv (key_pair, comfort_score)
    - keypair_distance_scores.csv (key_pair, distance_score)  
    - keypair_dvorak9_scores.csv (key_pair, dvorak9_score)

Creates two standardized output files:
    - output/keypair_scores.csv: Unified key-pair scores
        - key_pair: Two-character key pair (e.g., "QW", "AS")
        - Original score columns (time_score, comfort_score, distance_score, dvorak9_score)
        - Normalized score columns (*_score_normalized) with smart distribution-aware normalization (0-1 range)
    
    - output/key_scores.csv: Individual key comfort scores
        - key: Individual key character
        - comfort_score: Normalized comfort score for that key (extracted from same-key bigrams like "AA", ";;")

All floating point values are formatted to 6 decimal places.
Missing input files will be skipped with a warning.
The script uses smart normalization with automatic distribution detection.
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
        
        # Create the unified table
        create_unified_score_table(
            args.input_dir,
            args.verbose
        )
        
        # Create individual key comfort scores table
        create_key_comfort_scores(
            args.input_dir,
            args.verbose
        )
        
        print(f"Successfully created scoring tables:")
        print(f"  - Keypair scores: {Path(args.input_dir) / 'keypair_scores.csv'}")
        print(f"  - Key scores: {Path(args.input_dir) / 'key_scores.csv'}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())