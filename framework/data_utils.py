#!/usr/bin/env python3
"""
Data utilities for keyboard layout scoring.

Common functions for loading, validating, and normalizing data files.
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd


def load_csv_with_validation(filepath: str, 
                           required_columns: List[str],
                           optional_columns: Optional[List[str]] = None,
                           dtype_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Load CSV file with column validation and optional data type specification.
    
    Args:
        filepath: Path to CSV file
        required_columns: List of column names that must be present
        optional_columns: List of optional column names
        dtype_map: Dict mapping column names to pandas dtypes
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if not file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Determine delimiter
    delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
    
    try:
        # Load with basic error handling
        df = pd.read_csv(filepath, delimiter=delimiter, dtype=dtype_map, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    # Validate required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"Missing required columns in {filepath}: {missing_columns}. "
            f"Available columns: {available_columns}"
        )
    
    # Add info about optional columns
    if optional_columns:
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            print(f"Note: Optional columns not found in {filepath}: {missing_optional}")
    
    return df


def load_bigram_frequencies(filepath: str, 
                          bigram_col: Optional[str] = None,
                          frequency_col: Optional[str] = None,
                          verbose: bool = False) -> Tuple[List[str], List[float]]:
    """
    Load bigram frequencies from CSV file with automatic column detection.
    
    Args:
        filepath: Path to CSV file
        bigram_col: Name of bigram column (auto-detected if None)
        frequency_col: Name of frequency column (auto-detected if None)
        verbose: If True, print loading information
        
    Returns:
        Tuple of (bigrams, frequencies) lists
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns can't be found or data is invalid
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Bigram frequencies file not found: {filepath}")
    
    # Load the CSV
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    
    if df.empty:
        raise ValueError(f"Empty CSV file: {filepath}")
    
    columns = list(df.columns)
    
    # Auto-detect bigram column
    if bigram_col is None:
        bigram_candidates = ['bigram', 'letter_pair', 'pair', 'sequence', 'letters']
        for candidate in bigram_candidates:
            if candidate in columns:
                bigram_col = candidate
                break
        
        if bigram_col is None:
            raise ValueError(
                f"Could not find bigram column in {filepath}. "
                f"Available columns: {columns}. "
                f"Expected one of: {bigram_candidates}"
            )
    
    # Auto-detect frequency column
    if frequency_col is None:
        freq_candidates = ['frequency', 'freq', 'probability', 'prob', 'weight', 'count']
        for candidate in freq_candidates:
            if candidate in columns:
                frequency_col = candidate
                break
        
        if frequency_col is None:
            raise ValueError(
                f"Could not find frequency column in {filepath}. "
                f"Available columns: {columns}. "
                f"Expected one of: {freq_candidates}"
            )
    
    if verbose:
        print(f"Loading bigram frequencies from {filepath}")
        #print(f"  Using columns: bigram='{bigram_col}', frequency='{frequency_col}'")
    
    # Extract data
    bigrams = []
    frequencies = []
    
    for _, row in df.iterrows():
        bigram_str = str(row[bigram_col]).strip().lower()
        freq_str = str(row[frequency_col]).strip()
        
        # Skip empty rows
        if not bigram_str or not freq_str:
            continue
        
        # Validate bigram format
        if len(bigram_str) != 2:
            if verbose:
                print(f"Warning: Skipping invalid bigram '{bigram_str}' (not 2 characters)")
            continue
            
        try:
            frequency = float(freq_str)
            bigrams.append(bigram_str)
            frequencies.append(frequency)
        except ValueError:
            if verbose:
                print(f"Warning: Could not parse frequency '{freq_str}' for bigram '{bigram_str}', skipping")
            continue
    
    if not bigrams:
        raise ValueError(f"No valid bigram-frequency pairs found in {filepath}")
    
    if verbose:
        print(f"  Loaded {len(bigrams)} bigram frequencies")
    
    return bigrams, frequencies


def load_key_value_csv(filepath: str,
                      key_col: Optional[str] = None,
                      value_col: Optional[str] = None,
                      key_transform: Optional[callable] = None,
                      value_transform: Optional[callable] = None,
                      verbose: bool = False) -> Dict[str, float]:
    """
    Load a simple key-value CSV file into a dictionary.
    
    Args:
        filepath: Path to CSV file
        key_col: Name of key column (auto-detected if None)
        value_col: Name of value column (auto-detected if None)
        key_transform: Function to transform keys (e.g., str.lower)
        value_transform: Function to transform values (e.g., float)
        verbose: If True, print loading information
        
    Returns:
        Dictionary mapping keys to values
    """
    df = load_csv_with_validation(filepath, [])  # Let auto-detection handle columns
    
    columns = list(df.columns)
    
    if len(columns) < 2:
        raise ValueError(f"CSV file must have at least 2 columns, found {len(columns)}")
    
    # Auto-detect key column (first column if not specified)
    if key_col is None:
        key_col = columns[0]
    elif key_col not in columns:
        raise ValueError(f"Key column '{key_col}' not found. Available: {columns}")
    
    # Auto-detect value column (second column if not specified)
    if value_col is None:
        if len(columns) >= 2:
            value_col = columns[1]
        else:
            raise ValueError("Could not auto-detect value column")
    elif value_col not in columns:
        raise ValueError(f"Value column '{value_col}' not found. Available: {columns}")
    
    if verbose:
        print(f"Loading key-value data from {filepath}")
        #print(f"  Using columns: key='{key_col}', value='{value_col}'")
    
    # Extract data with optional transformations
    result = {}
    
    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        value_str = str(row[value_col]).strip()
        
        if not key or not value_str:
            continue
        
        # Apply transformations
        if key_transform:
            key = key_transform(key)
        
        try:
            if value_transform:
                value = value_transform(value_str)
            else:
                value = float(value_str)
            
            result[key] = value
            
        except (ValueError, TypeError) as e:
            if verbose:
                print(f"Warning: Could not process row {key}='{value_str}': {e}")
            continue
    
    if verbose:
        print(f"  Loaded {len(result)} key-value pairs")
    
    return result


def validate_data_consistency(data_dict: Dict[str, Any], 
                            name: str = "data",
                            expected_count: Optional[int] = None) -> List[str]:
    """
    Validate data dictionary for consistency and completeness.
    
    Args:
        data_dict: Dictionary of data to validate
        name: Name of the data for error messages
        expected_count: Expected number of entries (optional)
        
    Returns:
        List of validation issues (empty if all valid)
    """
    issues = []
    
    if not data_dict:
        issues.append(f"{name} is empty")
        return issues
    
    # Check expected count
    if expected_count is not None and len(data_dict) != expected_count:
        issues.append(f"{name} has {len(data_dict)} entries, expected {expected_count}")
    
    # Check for None or invalid values
    none_keys = [k for k, v in data_dict.items() if v is None]
    if none_keys:
        issues.append(f"{name} has None values for keys: {none_keys[:5]}{'...' if len(none_keys) > 5 else ''}")
    
    # Check for non-numeric values (if expected to be numeric)
    if data_dict and isinstance(next(iter(data_dict.values())), (int, float)):
        non_numeric = [k for k, v in data_dict.items() if v is not None and not isinstance(v, (int, float))]
        if non_numeric:
            issues.append(f"{name} has non-numeric values for keys: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
    
    # Check for extreme values that might indicate errors
    if data_dict and all(isinstance(v, (int, float)) for v in data_dict.values() if v is not None):
        values = [v for v in data_dict.values() if v is not None]
        if values:
            min_val, max_val = min(values), max(values)
            if max_val / min_val > 1000 and min_val > 0:  # Very large range
                issues.append(f"{name} has very large value range: {min_val:.6f} to {max_val:.6f}")
    
    return issues


def create_lookup_table(data_dict: Dict[str, float],
                       key_pairs: List[Tuple[str, str]],
                       default_value: float = 0.0,
                       verbose: bool = False) -> Dict[Tuple[str, str], float]:
    """
    Create a lookup table for key pairs from a dictionary of individual key values.
    
    Args:
        data_dict: Dictionary mapping individual keys to values
        key_pairs: List of key pairs to create lookup for
        default_value: Default value for missing keys
        verbose: If True, print creation information
        
    Returns:
        Dictionary mapping key pairs to combined values
    """
    lookup = {}
    missing_keys = set()
    
    for key1, key2 in key_pairs:
        value1 = data_dict.get(key1, default_value)
        value2 = data_dict.get(key2, default_value)
        
        if key1 not in data_dict:
            missing_keys.add(key1)
        if key2 not in data_dict:
            missing_keys.add(key2)
        
        # Combine values (could be multiplication, addition, etc.)
        combined_value = value1 * value2
        lookup[(key1, key2)] = combined_value
    
    if verbose and missing_keys:
        print(f"Warning: {len(missing_keys)} keys not found in data_dict, using default value {default_value}")
        if len(missing_keys) <= 10:
            print(f"  Missing keys: {sorted(missing_keys)}")
    
    if verbose:
        print(f"Created lookup table with {len(lookup)} key pairs")
    
    return lookup