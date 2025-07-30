#!/usr/bin/env python3
"""
Engram Layout Scorer

A self-contained script for scoring keyboard layouts based on letter frequencies
and key comfort scores. Works with CSV input files and requires no external 
dependencies beyond pandas and numpy.

This version has all the features of the original, but when --csv is used,
it outputs ONLY CSV data with no other messages for perfect scripting integration.

Usage:
    python engram_scorer.py --items "qwertyuiop" --positions "qwertyuiop" 
    python engram_scorer.py --items "pyfgcrlaoe" --positions "qwertyuiop" --details
    python engram_scorer.py --items "pyfgcrlaoe" --positions "qwertyuiop" --ignore-cross-hand

Input Files (CSV format, in input/engram/ directory):
    - normalized_letter_frequencies_en.csv:               
      letter,frequency                           
      e,12.70   
    - normalized_letter_pair_frequencies_en.csv:
      letter_pair,frequency
      th,3.56
    - normalized_key_comfort_scores_32keys.csv:
      key,comfort_score
      D,7.2
    - normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv:
      key_pair,comfort_score
      DF,7.2

For reference:
    QWERTY: "qwertyuiopasdfghjkl;zxcvbnm,./"  
    Dvorak: "',.pyfgcrlaoeuidhtns;qjkxbmwvz"
    Colemak: "qwfpgjluy;arstdhneiozxcvbkm,./"
"""

import argparse
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional

# Default combination strategy for scores
DEFAULT_COMBINATION_STRATEGY = "multiplicative"  # item_score * item_pair_score

# Keyboard hand mapping for filtering cross-hand bigrams  
POSITION_HANDS = {
    # Left hand positions (lowercase)
    '1': 'L', '2': 'L', '3': 'L', '4': 'L', '5': 'L',
    'q': 'L', 'w': 'L', 'e': 'L', 'r': 'L', 't': 'L',
    'a': 'L', 's': 'L', 'd': 'L', 'f': 'L', 'g': 'L',
    'z': 'L', 'x': 'L', 'c': 'L', 'v': 'L', 'b': 'L',
    # Right hand positions (lowercase)
    '6': 'R', '7': 'R', '8': 'R', '9': 'R', '0': 'R',
    'y': 'R', 'u': 'R', 'i': 'R', 'o': 'R', 'p': 'R',
    'h': 'R', 'j': 'R', 'k': 'R', 'l': 'R', ';': 'R',
    'n': 'R', 'm': 'R', ',': 'R', '.': 'R', '/': 'R',
    "'": 'R', '[': 'R', ']': 'R',
}

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

def load_and_normalize_scores(item_file: str, item_pair_file: str, 
                             position_file: str, position_pair_file: str,
                             verbose: bool = True) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load and normalize all score files.
    
    Returns:
        Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
    """
    if verbose:
        print("Loading and normalizing score files...")
    
    # Load and normalize item scores
    if verbose:
        print(f"  Loading item scores from: {item_file}")
    item_df = pd.read_csv(item_file, dtype={'letter': str})
    scores = item_df['frequency'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item scores', verbose)
    item_scores = {row['letter'].lower(): float(norm_scores[i]) 
                   for i, (_, row) in enumerate(item_df.iterrows())}
    
    # Load and normalize item pair scores
    if verbose:
        print(f"  Loading item pair scores from: {item_pair_file}")
    item_pair_df = pd.read_csv(item_pair_file, dtype={'letter_pair': str})
    scores = item_pair_df['frequency'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item pair scores', verbose)
    item_pair_scores = {}
    for i, (_, row) in enumerate(item_pair_df.iterrows()):
        pair_str = str(row['letter_pair'])
        if len(pair_str) == 2:
            key = (pair_str[0].lower(), pair_str[1].lower())
            item_pair_scores[key] = float(norm_scores[i])
    
    # Load and normalize position scores
    if verbose:
        print(f"  Loading position scores from: {position_file}")
    position_df = pd.read_csv(position_file, dtype={'key': str})
    scores = position_df['comfort_score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Position scores', verbose)
    position_scores = {row['key'].lower(): float(norm_scores[i]) 
                      for i, (_, row) in enumerate(position_df.iterrows())}
    
    # Load and normalize position-pair scores
    if verbose:
        print(f"  Loading position-pair scores from: {position_pair_file}")
    position_pair_df = pd.read_csv(position_pair_file, dtype={'key_pair': str}, keep_default_na=False)
    scores = position_pair_df['comfort_score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'position-pair scores', verbose)
    position_pair_scores = {}
    for i, (_, row) in enumerate(position_pair_df.iterrows()):
        pair_str = str(row['key_pair'])
        if len(pair_str) == 2:
            key = (pair_str[0].lower(), pair_str[1].lower())
            position_pair_scores[key] = float(norm_scores[i])
    
    if verbose:
        print(f"  Loaded {len(item_scores)} item scores")
        print(f"  Loaded {len(item_pair_scores)} item pair scores")
        print(f"  Loaded {len(position_scores)} position scores")
        print(f"  Loaded {len(position_pair_scores)} position-pair scores")
    
    return item_scores, item_pair_scores, position_scores, position_pair_scores

def apply_combination(item_score: float, item_pair_score: float) -> float:
    """Apply the default combination strategy for item and item-pair scores."""
    if DEFAULT_COMBINATION_STRATEGY == "multiplicative":
        return item_score * item_pair_score
    elif DEFAULT_COMBINATION_STRATEGY == "additive":
        return item_score + item_pair_score
    elif DEFAULT_COMBINATION_STRATEGY == "weighted_additive":
        return 0.6 * item_score + 0.4 * item_pair_score
    else:
        raise ValueError(f"Unknown combination strategy: {DEFAULT_COMBINATION_STRATEGY}")

def is_same_hand_pair(pos1: str, pos2: str) -> bool:
    """Check if two positions are on the same hand of the keyboard."""
    pos1_lower = pos1.lower()
    pos2_lower = pos2.lower()
    if pos1_lower not in POSITION_HANDS or pos2_lower not in POSITION_HANDS:
        return False
    return POSITION_HANDS[pos1_lower] == POSITION_HANDS[pos2_lower]

def calculate_layout_score(items_str: str, positions_str: str,
                          normalized_scores: Tuple,
                          ignore_cross_hand: bool = False) -> Tuple[float, float, float, dict]:
    """
    Calculate complete layout score.
    
    Args:
        items_str: String of items (e.g., 'etaoinsrhl')
        positions_str: String of positions (e.g., 'FDESVRJKIL')
        normalized_scores: Tuple of (item_scores, item_pair_scores, position_scores, position_pair_scores)
        ignore_cross_hand: If True, ignore cross-hand bigrams
        
    Returns:
        Tuple of (total_score, item_component, item_pair_component, filtering_info)
    """
    item_scores, item_pair_scores, position_scores, position_pair_scores = normalized_scores
    
    # Create mapping
    items = list(items_str.lower())
    positions = list(positions_str.lower())
    n_items = len(items)
    
    if len(items) != len(positions):
        raise ValueError(f"Items length ({len(items)}) != positions length ({len(positions)})")
    
    # Calculate item component
    item_raw_score = 0.0
    for item, pos in zip(items, positions):
        item_score = item_scores.get(item, 0.0)
        pos_score = position_scores.get(pos, 0.0)
        item_raw_score += item_score * pos_score
    
    item_component = item_raw_score / n_items if n_items > 0 else 0.0
    
    # Calculate item-pair component with optional filtering
    pair_raw_score = 0.0
    pair_count = 0
    cross_hand_pairs_filtered = 0
    
    for i in range(n_items):
        for j in range(n_items):
            if i != j:  # Skip self-pairs
                item1, item2 = items[i], items[j]
                pos1, pos2 = positions[i], positions[j]
                
                # Filter cross-hand pairs if requested
                if ignore_cross_hand and not is_same_hand_pair(pos1, pos2):
                    cross_hand_pairs_filtered += 1
                    continue
                
                # Get scores with defaults
                item_pair_key = (item1, item2)
                item_pair_score = item_pair_scores.get(item_pair_key, 1.0)
                
                pos_pair_key = (pos1, pos2)
                pos_pair_score = position_pair_scores.get(pos_pair_key, 1.0)
                
                pair_raw_score += item_pair_score * pos_pair_score
                pair_count += 1
    
    pair_component = pair_raw_score / max(1, pair_count)
    total_score = apply_combination(item_component, pair_component)
    
    filtering_info = {
        'cross_hand_pairs_filtered': cross_hand_pairs_filtered,
        'pairs_used': pair_count,
        'total_possible_pairs': n_items * (n_items - 1)
    }
    
    return total_score, item_component, pair_component, filtering_info

def print_detailed_breakdown(items_str: str, positions_str: str, 
                           normalized_scores: Tuple) -> None:
    """Print detailed item-by-item scoring breakdown."""
    item_scores, item_pair_scores, position_scores, position_pair_scores = normalized_scores
    
    items = list(items_str.lower())
    positions = list(positions_str.lower())
    
    print(f"\nDetailed Item Breakdown:")
    print(f"  {'letter':<4} | {'Pos':<3} | {'Item Score':<10} | {'Pos Score':<10} | {'Combined':<10}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    item_details = []
    for item, pos in zip(items, positions):
        item_score = item_scores.get(item, 0.0)
        pos_score = position_scores.get(pos, 0.0)
        combined = item_score * pos_score
        item_details.append((item, pos, item_score, pos_score, combined))
    
    # Sort by combined score (highest first)
    item_details.sort(key=lambda x: x[4], reverse=True)
    
    for item, pos, item_score, pos_score, combined in item_details:
        print(f"  {item:<4} | {pos.upper():<3} | {item_score:<10.6f} | {pos_score:<10.6f} | {combined:<10.6f}")

def validate_positions_in_pair_file(positions_str: str, position_pair_scores: dict, 
                                   ignore_cross_hand: bool = False, verbose: bool = True) -> None:
    """
    Validate that positions have corresponding entries in the position-pair file.
    """
    if not verbose:
        return  # Skip validation output in CSV mode
        
    used_positions = set(pos.lower() for pos in positions_str)
    
    # Get positions that appear in the position-pair file
    available_positions = set()
    for (pos1, pos2) in position_pair_scores.keys():
        available_positions.add(pos1)
        available_positions.add(pos2)
    
    # Check for missing individual positions
    missing_positions = used_positions - available_positions
    
    if missing_positions:
        missing_str = ''.join(sorted(missing_positions)).upper()
        available_str = ''.join(sorted(available_positions)).upper()
        print(f"  Warning: Missing positions in pair file: {missing_str}")
        print(f"  Available positions: {available_str}")
    else:
        print(f"  ✓ All positions found in position-pair file")
    
    # Check for missing position-pairs
    missing_pairs = []
    positions_list = list(used_positions)
    
    for i in range(len(positions_list)):
        for j in range(len(positions_list)):
            if i != j:
                pos1, pos2 = positions_list[i], positions_list[j]
                
                # Skip cross-hand pairs if filtering is enabled
                if ignore_cross_hand and not is_same_hand_pair(pos1, pos2):
                    continue
                    
                if (pos1, pos2) not in position_pair_scores:
                    missing_pairs.append(f"{pos1}{pos2}".upper())

    if missing_pairs and len(missing_pairs) <= 10:  # Only show if not too many
        missing_pairs.sort()
        missing_pairs_str = ' '.join(missing_pairs)
        print(f"  Note: Missing position-pairs (using default scores): {missing_pairs_str}")
    elif missing_pairs:
        print(f"  Note: {len(missing_pairs)} missing position-pairs (using default scores)")

def filter_letter_pairs(items_str: str, positions_str: str, allow_all: bool, verbose: bool = True) -> Tuple[str, str]:
    """Filter to letter-position-pairs only unless allow_all is True."""
    if allow_all:
        return items_str, positions_str
    
    filtered_items = []
    filtered_positions = []
    removed_pairs = []
    
    for item, pos in zip(items_str, positions_str):
        if item.isalpha():
            filtered_items.append(item)
            filtered_positions.append(pos)
        else:
            removed_pairs.append(f"{item}→{pos}")
    
    if removed_pairs and verbose:
        print(f"Note: Removed non-letter pairs: {removed_pairs}")
    
    return ''.join(filtered_items), ''.join(filtered_positions)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Score keyboard layouts based on letter frequencies and key comfort.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scoring
  python engram_scorer.py --items "etaoinsrhl" --positions "qwertyuiop"
  
  # With detailed breakdown
  python engram_scorer.py --items "etaoinsrhl" --positions "qwertyuiop" --details
  
  # Ignore cross-hand bigrams (keyboard-specific)
  python engram_scorer.py --items "etaoinsrhl" --positions "qwertyuiop" --ignore-cross-hand
  
  # CSV output format (ONLY CSV data, no other messages)
  python engram_scorer.py --items "etaoinsrhl" --positions "qwertyuiop" --csv

Input CSV files should be in the input/engram/ directory:
  - input/engram/normalized_letter_frequencies_en.csv (letter frequencies)
  - normalized_letter_pair_frequencies_en.csv (bigram frequencies)  
  - normalized_key_comfort_scores_32keys.csv (key comfort scores)
  - normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv (key-pair comfort scores)
        """
    )
    
    parser.add_argument("--items", required=True,
                       help="String of items (e.g., 'etaoinsrhl')")
    parser.add_argument("--positions", required=True,
                       help="String of positions (e.g., 'qwertyuiop')")
    
    # Input file arguments
    parser.add_argument("--item-scores", default="input/engram/normalized_letter_frequencies_en.csv",
                       help="Item scores CSV file (default: input/engram/normalized_letter_frequencies_en.csv)")
    parser.add_argument("--item-pair-scores", default="input/engram/normalized_letter_pair_frequencies_en.csv",
                       help="Item pair scores CSV file (default: input/engram/normalized_letter_pair_frequencies_en.csv)")
    parser.add_argument("--position-scores", default="input/engram/normalized_key_comfort_scores_32keys.csv",
                       help="Position scores CSV file (default: input/engram/normalized_key_comfort_scores_32keys.csv)")
    parser.add_argument("--position-pair-scores", default="input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv",
                       help="position-pair scores CSV file (default: input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv)")

    # Options
    parser.add_argument("--details", action="store_true",
                       help="Show detailed scoring breakdown")
    parser.add_argument("--csv", action="store_true",
                       help="Output ONLY CSV format (total_score,item_score,item_pair_score)")
    parser.add_argument("--nonletter-items", action="store_true",
                       help="Allow non-letter characters in --items (default: letters only)")
    parser.add_argument("--ignore-cross-hand", action="store_true",
                       help="Ignore bigrams that cross hands (keyboard-specific)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()
    
    try:
        # Key change: verbose is False when --csv is used (for clean CSV output)
        verbose = not args.csv and not args.quiet
        
        # Check if files exist - but only show error if not in CSV mode
        required_files = [args.item_scores, args.item_pair_scores, 
                         args.position_scores, args.position_pair_scores]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            if not args.csv:
                print(f"Error: Missing required files: {missing_files}")
                print("\nRequired CSV files:")
                print(f"  --item-scores: {args.item_scores}")
                print(f"  --item-pair-scores: {args.item_pair_scores}")
                print(f"  --position-scores: {args.position_scores}")
                print(f"  --position-pair-scores: {args.position_pair_scores}")
            return 1
        
        if verbose:
            print("Engram Layout Scorer")
            print("=" * 50)

        # Validate input lengths
        if len(args.items) != len(args.positions):
            if not args.csv:
                print(f"Error: Item count ({len(args.items)}) != Position count ({len(args.positions)})")
            return 1

        # Filter to letter pairs if requested
        valid_items, valid_positions = filter_letter_pairs(
            args.items, args.positions, args.nonletter_items, verbose)

        if len(valid_items) == 0:
            if not args.csv:
                print("Error: No letters found in items string")
            return 1
        
        # Load and normalize scores
        normalized_scores = load_and_normalize_scores(
            args.item_scores, args.item_pair_scores,
            args.position_scores, args.position_pair_scores,
            verbose=verbose
        )
        
        # Validate positions
        if verbose:
            print("\nValidating positions...")
        validate_positions_in_pair_file(
            valid_positions, normalized_scores[3], args.ignore_cross_hand, verbose)
        
        # Display layout
        if verbose:
            print(f"\nLayout: {valid_items} → {valid_positions.upper()}")
        
        # Calculate scores
        total_score, item_score, item_pair_score, filtering_info = calculate_layout_score(
            valid_items, valid_positions, normalized_scores, args.ignore_cross_hand
        )

        if args.csv:
            # CSV output format - ONLY CSV data, no other output
            print("total_score,item_score,item_pair_score")
            print(f"{total_score:.12f},{item_score:.12f},{item_pair_score:.12f}")
        else:
            # Human-readable output
            print(f"\nLayout Score Results:")
            print(f"  Total score:         {total_score:.12f}")
            print(f"  Item component:      {item_score:.12f}")
            print(f"  Item-pair component: {item_pair_score:.12f}")
            print(f"  Combination strategy: {DEFAULT_COMBINATION_STRATEGY}")
        
        # Show filtering information (only if verbose)
        if args.ignore_cross_hand and verbose:
            info = filtering_info
            total_possible = info['total_possible_pairs']
            filtered = info['cross_hand_pairs_filtered']
            used = info['pairs_used']
            print(f"\nCross-hand filtering:")
            print(f"  Total possible pairs:      {total_possible}")
            print(f"  Cross-hand pairs filtered: {filtered}")
            print(f"  Same-hand pairs used:      {used}")
            if total_possible > 0:
                print(f"  Filtering ratio:           {filtered/total_possible*100:.1f}%")
        
        # Show detailed breakdown if requested (only if verbose)
        if args.details and verbose:
            print_detailed_breakdown(valid_items, valid_positions, normalized_scores)
        
        return 0
                    
    except FileNotFoundError as e:
        if not args.csv:
            print(f"Error: {e}")
            print("Please check that all CSV input files exist.")
        return 1
    except ValueError as e:
        if not args.csv:
            print(f"Input Error: {e}")
        return 1
    except Exception as e:
        if not args.csv:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())