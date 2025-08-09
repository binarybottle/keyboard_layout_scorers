#!/usr/bin/env python3
"""
create_keypair_score_table.py - Create comprehensive key pair scoring table from individual score files

Takes individual score CSV files and prepares a unified table with normalized scores.

Input files expected:
- keypair_time_scores.csv (key_pair, time_score)
- keypair_comfort_scores.csv (key_pair, comfort_score)  
- keypair_distance_scores.csv (key_pair, distance_score)
- keypair_dvorak9_scores.csv (key_pair, dvorak9_score)

Output:
- Unified CSV with key_pair and normalized versions of all scores (0-1 range)

Usage:
    python prep_score_table.py --input-dir output/ --output score_table.csv
    python prep_score_table.py --input-dir output/ --output score_table.csv --verbose
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


def create_unified_score_table(input_dir: str, output_file: str, verbose: bool = False) -> None:
    """Create unified key pair scoring table with normalized scores."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Define expected input files and their score columns
    score_files = {
        'time_score': 'keypair_time_scores.csv',
        'comfort_score': 'keypair_comfort_scores.csv',
        'distance_score': 'keypair_distance_scores.csv',
        'dvorak9_score': 'keypair_dvorak9_scores.csv'
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
        description="Create unified key pair scoring table from individual score files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_keypair_score_table.py --input-dir output/ --output score_table.csv
    python create_keypair_score_table.py --input-dir output/ --output score_table.csv --verbose

Input files expected in input directory:
    - keypair_time_scores.csv (key_pair, time_score)
    - keypair_comfort_scores.csv (key_pair, comfort_score)
    - keypair_distance_scores.csv (key_pair, distance_score)  
    - keypair_dvorak9_scores.csv (key_pair, dvorak9_score)

The output CSV will contain:
    - key_pair: Two-character key pair (e.g., "QW", "AS")
    - Original score columns (time_score, comfort_score, distance_score, dvorak9_score)
    - Normalized score columns (*_score_normalized) with smart distribution-aware normalization (0-1 range)

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
        '--output',
        required=True,
        help="Output CSV file path for the unified scoring table"
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
            args.output,
            args.verbose
        )
        
        print(f"Successfully created unified scoring table: {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())