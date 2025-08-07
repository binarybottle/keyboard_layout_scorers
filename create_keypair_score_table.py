#!/usr/bin/env python3
"""
create_keypair_score_table.py - Standalone script to create comprehensive key pair scoring table

Creates a CSV table with scores for all possible key pairs (1024 ordered pairs)
from 32 QWERTY keys for use by compare_layouts.py visualization.

This is a completely standalone script with no external dependencies beyond standard library.

Columns:
1. key_pair - all 1024 ordered pairs of 32 QWERTY keys  
2. speed - zeros (placeholder for future calculation)
3. speed_0_1 - normalized speed scores (zeros for now)
4. distance - raw distance in mm using euclidean distance
5. distance_0_1 - normalized distance scores (0-1) 
6. comfort - raw comfort scores from input file
7. comfort_0_1 - normalized comfort scores (0-1)
8. dvorak9_score - average of all 9 Dvorak criteria
9-17. Each of the 9 Dvorak criteria individually (with dvorak9_ prefix)

Usage:
    python create_keypair_score_table.py --raw-comfort-scores comfort_raw.csv --output keypair_scores.csv
    python create_keypair_score_table.py --raw-comfort-scores comfort_raw.csv --output keypair_scores.csv --verbose
    python create_keypair_score_table.py --raw-comfort-scores comfort_raw.csv --output keypair_scores.csv --default_comfort_score 0.5 --verbose
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from math import sqrt
import pandas as pd

# ============================================================================
# CONSTANTS AND DATA STRUCTURES (extracted from original codebase)
# ============================================================================

# Standard 32-key QWERTY layout in position order
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

# Physical keyboard layout positions in mm (from distance_scorer.py)
STAGGERED_POSITION_MAP = {
    # Top row (no stagger reference point)
    'q': (0, 0),    'w': (19, 0),   'e': (38, 0),   'r': (57, 0),   't': (76, 0),
    # Home row (staggered 5mm right from top row)
    'a': (5, 19),   's': (24, 19),  'd': (43, 19),  'f': (62, 19),  'g': (81, 19),
    # Bottom row (staggered 10mm right from home row)
    'z': (15, 38),  'x': (34, 38),  'c': (53, 38),  'v': (72, 38),  'b': (91, 38),
    # Top row continued
    'y': (95, 0),   'u': (114, 0),  'i': (133, 0),  'o': (152, 0),  'p': (171, 0),  '[': (190, 0),
    # Home row continued
    'h': (100, 19), 'j': (119, 19), 'k': (138, 19), 'l': (157, 19), ';': (176, 19), "'": (195, 19),
    # Bottom row continued
    'n': (110, 38), 'm': (129, 38), ',': (148, 38), '.': (167, 38), '/': (186, 38)
}

# QWERTY keyboard layout with (row, finger, hand) mapping (from dvorak9_scorer.py)
QWERTY_LAYOUT = {
    # Number row (row 0)
    '1': (0, 4, 'L'), '2': (0, 3, 'L'), '3': (0, 2, 'L'), '4': (0, 1, 'L'), '5': (0, 1, 'L'),
    '6': (0, 1, 'R'), '7': (0, 1, 'R'), '8': (0, 2, 'R'), '9': (0, 3, 'R'), '0': (0, 4, 'R'),
    
    # Top row (row 1)
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    
    # Home row (row 2) 
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    
    # Bottom row (row 3)
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    
    # Additional common keys
    "'": (2, 4, 'R'), '[': (1, 4, 'R'),
}

# Dvorak finger strength and home row definitions
STRONG_FINGERS = {1, 2}
WEAK_FINGERS   = {3, 4}
HOME_ROW = 2

# Define finger column assignments
FINGER_COLUMNS = {
    'L': {
        4: ['Q', 'A', 'Z'],
        3: ['W', 'S', 'X'],
        2: ['E', 'D', 'C'],
        1: ['R', 'F', 'V']
    },
    'R': {
        1: ['U', 'J', 'M'],
        2: ['I', 'K', ','],
        3: ['O', 'L', '.'],
        4: ['P', ';', '/']
    }
}

# ============================================================================
# EMBEDDED FUNCTIONS (extracted from original codebase)
# ============================================================================

def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions in mm."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return sqrt(dx * dx + dy * dy)


def get_key_info(key: str):
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    return QWERTY_LAYOUT.get(key)


def is_finger_in_column(key: str, finger: int, hand: str) -> bool:
    """Check if a key is in the designated column for a finger."""
    key = key.upper()
    if hand in FINGER_COLUMNS and finger in FINGER_COLUMNS[hand]:
        return key in FINGER_COLUMNS[hand][finger]
    return False


def score_bigram_dvorak9(bigram: str) -> Dict[str, float]:
    """Calculate all 9 Dvorak criteria scores for a bigram."""
    if len(bigram) != 2:
        raise ValueError("Bigram must be exactly 2 characters long")    
    
    char1, char2 = bigram[0].upper(), bigram[1].upper()
    
    # Get key information
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    
    if key_info1 is None or key_info2 is None:
        return {criterion: 0.5 for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                                                'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']}
    
    row1, finger1, hand1 = key_info1
    row2, finger2, hand2 = key_info2
    
    scores = {}
    
    # 1. Hands - favor alternating hands
    scores['hands'] = 1.0 if hand1 != hand2 else 0.0
    
    # 2. Fingers - avoid same finger repetition
    if hand1 != hand2:
        scores['fingers'] = 1.0  # Different hands = different fingers
    else:
        scores['fingers'] = 0.0 if finger1 == finger2 else 1.0
    
    # 3. Skip fingers - favor skipping more fingers (same hand only)
    if hand1 != hand2:
        scores['skip_fingers'] = 1.0      # Different hands is good
    elif finger1 == finger2:
        scores['skip_fingers'] = 0.0      # Same finger is bad
    else:
        finger_gap = abs(finger1 - finger2)
        if finger_gap == 1:
            scores['skip_fingers'] = 0    # Adjacent fingers is bad
        elif finger_gap == 2:
            scores['skip_fingers'] = 0.5  # Skipping 1 finger is good
        elif finger_gap == 3:
            scores['skip_fingers'] = 1.0  # Skipping 2 fingers is better
    
    # 4. Don't cross home - avoid hurdling over home row
    if hand1 != hand2:
        scores['dont_cross_home'] = 1.0  # Different hands always score well
    else:
        # Check for hurdling (top to bottom or bottom to top, skipping home)
        if (row1 == 1 and row2 == 3) or (row1 == 3 and row2 == 1):
            scores['dont_cross_home'] = 0.0  # Hurdling over home row
        else:
            scores['dont_cross_home'] = 1.0  # No hurdling
    
    # 5. Same row - favor staying in same row
    scores['same_row'] = 1.0 if row1 == row2 else 0.0
    
    # 6. Home row - favor using home row
    home_count = sum(1 for row in [row1, row2] if row == HOME_ROW)
    if home_count == 2:
        scores['home_row'] = 1.0      # Both in home row
    elif home_count == 1:
        scores['home_row'] = 0.5      # One in home row
    else:
        scores['home_row'] = 0.0      # Neither in home row
    
    # 7. Columns - favor fingers staying in their designated columns
    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    
    if in_column1 and in_column2:
        scores['columns'] = 1.0       # Both in correct columns
    elif in_column1 or in_column2:
        scores['columns'] = 0.5       # One in correct column
    else:
        scores['columns'] = 0.0       # Neither in correct column
    
    # 8. Strum - favor inward rolls (outer to inner fingers)
    if hand1 != hand2:
        scores['strum'] = 1.0         # Different hands get full score
    elif finger1 == finger2:
        scores['strum'] = 0.0         # Same finger gets zero
    else:
        # Inward roll: from higher finger number to lower (4→3→2→1)
        if finger1 > finger2:
            scores['strum'] = 1.0     # Inward roll
        else:
            scores['strum'] = 0.0     # Outward roll
    
    # 9. Strong fingers - favor index and middle fingers
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['strong_fingers'] = 1.0    # Both strong fingers
    elif strong_count == 1:
        scores['strong_fingers'] = 0.5    # One strong finger
    else:
        scores['strong_fingers'] = 0.0    # Both weak fingers
    
    return scores


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


def load_csv_simple(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """Simple CSV loader with basic validation."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {filepath}")
    
    if verbose:
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    return df

# ============================================================================
# MAIN SCORING FUNCTIONS
# ============================================================================

def load_comfort_scores(filepath: str, verbose: bool = False) -> Dict[str, float]:
    """Load comfort scores from CSV file with key_pair,comfort_score format."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Comfort scores file not found: {filepath}")
    
    if verbose:
        print(f"Loading comfort scores from: {filepath}")
    
    # Load with simple validation
    df = load_csv_simple(filepath, verbose)
    
    # Check for required columns
    if 'position_pair' not in df.columns or 'score' not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Required columns 'position_pair' and 'score' not found. Available: {available_cols}")
    
    comfort_scores = {}
    for _, row in df.iterrows():
        key_pair = str(row['position_pair']).strip().upper()
        comfort_score = float(row['score'])
        
        if len(key_pair) == 2:
            comfort_scores[key_pair] = comfort_score
    
    if verbose:
        print(f"  Loaded {len(comfort_scores)} comfort scores")
    
    return comfort_scores


def calculate_all_distances(key_pairs: List[str], verbose: bool = False) -> Tuple[List[float], List[float]]:
    """Calculate raw and normalized distance scores for all key pairs."""
    if verbose:
        print("Calculating distance scores...")
    
    raw_distances = []
    
    for pair in key_pairs:
        key1, key2 = pair[0], pair[1]
        
        # Get physical positions
        pos1 = STAGGERED_POSITION_MAP.get(key1.lower())
        pos2 = STAGGERED_POSITION_MAP.get(key2.lower())
        
        if pos1 and pos2:
            distance = calculate_euclidean_distance(pos1, pos2)
            raw_distances.append(distance)
        else:
            raw_distances.append(0.0)  # Default for missing positions
    
    # Normalize distances using smart normalization
    distances_array = np.array(raw_distances)
    normalized_distances = detect_and_normalize_distribution(distances_array, 'distance', verbose)
    
    if verbose:
        print(f"  Distance range: {np.min(distances_array):.2f} - {np.max(distances_array):.2f} mm")
        print(f"  Calculated {len(raw_distances)} distance scores")
    
    return raw_distances, normalized_distances.tolist()


def calculate_all_dvorak9_scores(key_pairs: List[str], verbose: bool = False) -> Tuple[List[float], Dict[str, List[float]]]:
    """Calculate Dvorak-9 scores for all key pairs."""
    if verbose:
        print("Calculating Dvorak-9 scores...")
    
    average_scores = []
    criteria_scores = {
        'hands': [],
        'fingers': [],
        'skip_fingers': [],
        'dont_cross_home': [],
        'same_row': [],
        'home_row': [],
        'columns': [],
        'strum': [],
        'strong_fingers': []
    }
    
    for pair in key_pairs:
        scores = score_bigram_dvorak9(pair)
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        average_scores.append(avg_score)
        
        # Store individual criteria scores
        for criterion, score in scores.items():
            criteria_scores[criterion].append(score)
    
    if verbose:
        print(f"  Calculated {len(average_scores)} Dvorak-9 scores")
        avg_range = f"{np.min(average_scores):.3f} - {np.max(average_scores):.3f}"
        print(f"  Dvorak-9 average score range: {avg_range}")
    
    return average_scores, criteria_scores


def create_keypair_score_table(raw_comfort_file: str, 
                              output_file: str,
                              default_comfort_score: float = 1.0,
                              verbose: bool = False) -> None:
    """Create the comprehensive key pair scoring table."""
    
    # Generate all 1024 ordered key pairs
    qwerty_keys = list(QWERTY_POSITIONS)
    key_pairs = []
    
    for key1 in qwerty_keys:
        for key2 in qwerty_keys:
            key_pairs.append(key1 + key2)
    
    if verbose:
        print(f"Generated {len(key_pairs)} key pairs from {len(qwerty_keys)} QWERTY keys")
        print(f"QWERTY keys: {QWERTY_POSITIONS}")
    
    # Load raw comfort scores only (we'll normalize using smart detection)
    raw_comfort_scores = load_comfort_scores(raw_comfort_file, verbose)
    
    if verbose:
        missing_pairs = [pair for pair in key_pairs if pair not in raw_comfort_scores]
        print(f"  Found {len(raw_comfort_scores)} comfort scores in input file")
        print(f"  Missing {len(missing_pairs)} key pairs (will use default {default_comfort_score})")
        if len(missing_pairs) <= 10:
            print(f"  Missing pairs: {missing_pairs}")
        elif missing_pairs:
            print(f"  Sample missing pairs: {missing_pairs[:5]}... (and {len(missing_pairs)-5} more)")
    
    # Calculate distance scores (includes smart normalization)
    raw_distances, norm_distances = calculate_all_distances(key_pairs, verbose)
    
    # Calculate Dvorak-9 scores
    dvorak9_averages, dvorak9_criteria = calculate_all_dvorak9_scores(key_pairs, verbose)
    
    # Apply smart normalization to comfort scores
    if verbose:
        print("Applying smart normalization to comfort scores...")
        print(f"  Using default comfort score {default_comfort_score} for missing key pairs")
    
    # Extract raw comfort values in the same order as key_pairs
    raw_comfort_values = [raw_comfort_scores.get(pair, default_comfort_score) for pair in key_pairs]
    norm_comfort_values = detect_and_normalize_distribution(np.array(raw_comfort_values), 'comfort', verbose)
    
    # Prepare data for CSV output
    if verbose:
        print("Preparing CSV output...")
    
    # Create CSV data
    csv_data = []
    
    for i, pair in enumerate(key_pairs):
        row = {
            'key_pair': pair,
            'speed': 0.0,  # Placeholder
            'speed_0_1': 0.0,  # Placeholder  
            'distance': raw_distances[i],
            'distance_0_1': norm_distances[i],
            'comfort': raw_comfort_values[i],
            'comfort_0_1': norm_comfort_values[i],
            'dvorak9_score': dvorak9_averages[i],
            'dvorak9_hands': dvorak9_criteria['hands'][i],
            'dvorak9_fingers': dvorak9_criteria['fingers'][i],
            'dvorak9_skip_fingers': dvorak9_criteria['skip_fingers'][i],
            'dvorak9_dont_cross_home': dvorak9_criteria['dont_cross_home'][i],
            'dvorak9_same_row': dvorak9_criteria['same_row'][i],
            'dvorak9_home_row': dvorak9_criteria['home_row'][i],
            'dvorak9_columns': dvorak9_criteria['columns'][i],
            'dvorak9_strum': dvorak9_criteria['strum'][i],
            'dvorak9_strong_fingers': dvorak9_criteria['strong_fingers'][i]
        }
        csv_data.append(row)
    
    # Write to CSV
    fieldnames = [
        'key_pair', 'speed', 'speed_0_1', 'distance', 'distance_0_1', 
        'comfort', 'comfort_0_1', 'dvorak9_score',
        'dvorak9_hands', 'dvorak9_fingers', 'dvorak9_skip_fingers', 'dvorak9_dont_cross_home',
        'dvorak9_same_row', 'dvorak9_home_row', 'dvorak9_columns', 'dvorak9_strum', 'dvorak9_strong_fingers'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in csv_data:
            # Format floating point numbers to 6 decimal places
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted_row[key] = f"{value:.6f}"
                else:
                    formatted_row[key] = value
            writer.writerow(formatted_row)
    
    if verbose:
        print(f"Saved key pair scoring table to: {output_file}")
        print(f"Table contains {len(csv_data)} rows and {len(fieldnames)} columns")
        
        # Show some statistics
        print("\nSample statistics:")
        print(f"  Distance range: {min(raw_distances):.2f} - {max(raw_distances):.2f} mm")
        print(f"  Dvorak-9 range: {min(dvorak9_averages):.3f} - {max(dvorak9_averages):.3f}")
        if raw_comfort_values:
            print(f"  Raw comfort range: {min(raw_comfort_values):.3f} - {max(raw_comfort_values):.3f}")
            print(f"  Normalized comfort range: {min(norm_comfort_values):.3f} - {max(norm_comfort_values):.3f}")


def validate_input_files(raw_comfort_file: str) -> None:
    """Validate that input files exist and are accessible."""
    if not Path(raw_comfort_file).exists():
        raise FileNotFoundError(f"Raw comfort scores file not found: {raw_comfort_file}")
    
    # Check if file is readable
    try:
        with open(raw_comfort_file, 'r') as f:
            pass
    except PermissionError:
        raise PermissionError(f"Cannot read raw comfort scores file: {raw_comfort_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive key pair scoring table for layout comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_keypair_score_table.py --raw_comfort_scores comfort_raw.csv --output keypair_scores.csv
    python create_keypair_score_table.py --raw_comfort_scores comfort_raw.csv --output keypair_scores.csv --verbose
    python create_keypair_score_table.py --raw_comfort_scores comfort_raw.csv --output keypair_scores.csv --default_comfort_score 0.5

Input CSV file format:
    key_pair,comfort_score
    QW,1.234567
    AS,2.345678
    ...

The output CSV will contain 1024 rows (one for each ordered key pair) with the following columns:
- key_pair: Two-character key pair (e.g., "QW", "AS")
- speed, speed_0_1: Placeholder columns (zeros) for future speed scores
- distance, distance_0_1: Raw distance in mm and normalized (0-1) distance scores  
- comfort, comfort_0_1: Raw comfort scores and smart-normalized (0-1) comfort scores
- dvorak9_score: Average of all 9 Dvorak criteria
- dvorak9_hands, dvorak9_fingers, dvorak9_skip_fingers, dvorak9_dont_cross_home, dvorak9_same_row, dvorak9_home_row, dvorak9_columns, dvorak9_strum, dvorak9_strong_fingers: Individual Dvorak-9 criteria scores

Key pairs not found in the input comfort file will use the --default_comfort_score value (default: 1.0).
All floating point values are formatted to 6 decimal places.
All "_0_1" columns use smart normalization with automatic distribution detection.

This is a completely standalone script with no external dependencies beyond standard library + numpy + pandas.
        """
    )
    
    parser.add_argument(
        '--raw-comfort-scores',
        required=True,
        help="CSV file with raw comfort scores (position_pair,score format)"
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help="Output CSV file path for the key pair scoring table"
    )
    
    parser.add_argument(
        '--default-comfort-score',
        type=float,
        default=1.0,
        help="Default comfort score for missing key pairs (default: 1.0)"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed information during processing"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input files first
        validate_input_files(args.raw_comfort_scores)
        
        # Create the table
        create_keypair_score_table(
            args.raw_comfort_scores,
            args.output,
            args.default_comfort_score,
            args.verbose
        )
        
        print(f"Successfully created key pair scoring table: {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())