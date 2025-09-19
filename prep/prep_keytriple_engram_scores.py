#!/usr/bin/env python3
"""
Generate precomputed Engram scores for all possible QWERTY key-triples.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes an Engram 3-key score for for every possible triple of QWERTY keys.

The output file contains all possible key-triples (e.g., "QWE", "QAS", "ASD") with
their corresponding finger order scores, where finger order refers to the 
finger sequence toward the thumb in the same row.

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

Scoring criteria for typing trigrams:

    1. Finger sequence/switchbacks (type in one direction vs. switch to the opposite)

Each criterion score for a layout is the average score across all trigrams.

Usage:
    python prep_keytriple_engram4of4_scores.py

Output:
    ../tables/engram_3key_scores_order.csv  - CSV with columns: key_triple, engram_score
"""

import csv
from pathlib import Path
from typing import Dict

# QWERTY keyboard layout with (row, column, finger, hand, homekey) mapping
# Finger numbering: 1=pinky, 2=ring, 3=middle, 4=index
QWERTY_LAYOUT = {
    # Number row (row 0)
    '1': (0, 1, 1, 'L', 0), '2': (0, 2, 2, 'L', 0), '3': (0, 3, 3, 'L', 0), '4': (0, 4, 4, 'L', 0), '5': (0, 5, 4, 'L', 0),
    '6': (0, 5, 4, 'R', 0), '7': (0, 4, 4, 'R', 0), '8': (0, 3, 3, 'R', 0), '9': (0, 2, 2, 'R', 0), '0': (0, 1, 1, 'R', 0),

    # Top row (row 1)
    'Q': (1, 1, 1, 'L', 0), 'W': (1, 2, 2, 'L', 0), 'E': (1, 3, 3, 'L', 0), 'R': (1, 4, 4, 'L', 0), 'T': (1, 5, 4, 'L', 0),
    'Y': (1, 5, 4, 'R', 0), 'U': (1, 4, 4, 'R', 0), 'I': (1, 3, 3, 'R', 0), 'O': (1, 2, 2, 'R', 0), 'P': (1, 1, 1, 'R', 0),

    # Home row (row 2)
    'A': (2, 1, 1, 'L', 1), 'S': (2, 2, 2, 'L', 1), 'D': (2, 3, 3, 'L', 1), 'F': (2, 4, 4, 'L', 1), 'G': (2, 5, 4, 'L', 0),
    'H': (2, 5, 4, 'R', 0), 'J': (2, 4, 4, 'R', 1), 'K': (2, 3, 3, 'R', 1), 'L': (2, 2, 2, 'R', 1), ';': (2, 1, 1, 'R', 1),

    # Bottom row (row 3)
    'Z': (3, 1, 1, 'L', 0), 'X': (3, 2, 2, 'L', 0), 'C': (3, 3, 3, 'L', 0), 'V': (3, 4, 4, 'L', 0), 'B': (3, 5, 4, 'L', 0),
    'N': (3, 5, 4, 'R', 0), 'M': (3, 4, 4, 'R', 0), ',': (3, 3, 3, 'R', 0), '.': (3, 2, 2, 'R', 0), '/': (3, 1, 1, 'R', 0),

    # Additional common keys
    "'": (2, 0, 1, 'R', 0), '[': (1, 0, 1, 'R', 0),
}

# Define finger column assignments (1=pinky, 4=index)
FINGER_COLUMNS = {
    'L': {
        1: ['Q', 'A', 'Z'],    # Pinky
        2: ['W', 'S', 'X'],    # Ring
        3: ['E', 'D', 'C'],    # Middle
        4: ['R', 'F', 'V'] #, 'T', 'G', 'B']  # Index (not including T, G, B)
    },
    'R': {
        4: ['U', 'J', 'M'],    # Index (not including Y, H, N)
        3: ['I', 'K', ','],    # Middle
        2: ['O', 'L', '.'],    # Ring
        1: ['P', ';', '/'] #, "'", '[']  # Pinky (not including ' and [)
    }
}

criteria = ['order'] 
ncriteria = len(criteria)

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

def score_trigram(trigram: str) -> Dict[str, float]:
    """Calculate Engram criteria scores for a trigram."""
    if len(trigram) != 3:
        raise ValueError("Trigram must be exactly 3 characters long")

    char1, char2, char3 = trigram[0].upper(), trigram[1].upper(), trigram[2].upper()
    
    # Get key information
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    key_info3 = get_key_info(char3)

    if key_info1 is None or key_info2 is None or key_info3 is None:
        raise ValueError(f"Invalid keys: {char1}, {char2}, {char3}")

    row1, column1, finger1, hand1, homekey1 = key_info1
    row2, column2, finger2, hand2, homekey2 = key_info2
    row3, column3, finger3, hand3, homekey3 = key_info3

    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    in_column3 = is_finger_in_column(char3, finger3, hand3)
    
    scores = {}

    #----------------------------------------------------------------------------------
    # Engram's trigram scoring criteria
    #----------------------------------------------------------------------------------    
    # 1. Finger sequence/switchbacks (type in one direction vs. switch to the opposite)
    #----------------------------------------------------------------------------------    
    # 1. Finger sequence
    #    1.0: 2 hands
    #    1.0: inward (toward thumb)
    #    1.0: outward (away from thumb)
    #    0.0: mixed patterns, same finger, unhandled cases
    scores['order'] = 0.0  # Default for mixed patterns, same finger, unhandled cases
    if hand1 != hand2 and hand1 == hand3:
        scores['order'] = 1.0          # two hands
    elif hand1 == hand2 == hand3:
        if finger1 < finger2 < finger3:
            scores['order'] = 1.0      # inward
        elif finger1 > finger2 > finger3:
            scores['order'] = 1.0      # outward
        #elif char1 == char3 and finger1 != finger2:  # switchback to same key
        #    scores['order'] = 1.0      # rock back to same key
    elif hand1 == hand2 and hand2 != hand3:
        if finger1 < finger2:
            scores['order'] = 1.0      # inward
        elif finger1 > finger2:
            scores['order'] = 1.0      # outward
    elif hand1 != hand2 and hand2 == hand3:
        if finger2 < finger3:
            scores['order'] = 1.0      # inward
        elif finger2 > finger3:
            scores['order'] = 1.0      # outward

    return scores

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing."""
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def generate_all_key_triples():
    """Generate all possible QWERTY key-triple combinations."""
    keys = get_all_qwerty_keys()
    key_triples = []
    
    for key1 in keys:
        for key2 in keys:
            for key3 in keys:
                key_triples.append(key1 + key2 + key3)
    
    return key_triples

def compute_key_triple_scores():
    """Compute Engram scores for all key-triples."""
    key_triples = generate_all_key_triples()
    results = {}
    
    # Initialize results for overall and individual criteria
    results['overall'] = []
    
    for criterion in criteria:
        results[criterion] = []

    print(f"Computing Engram scores for {len(key_triples)} key-triples...")

    for i, key_triple in enumerate(key_triples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_triples)} ({i/len(key_triples)*100:.1f}%)")
        
        # Compute individual Engram criteria scores using the scorer's function
        trigram_scores = score_trigram(key_triple)
        
        # Calculate sum (baseline Engram score)
        engram_score = sum(trigram_scores.values())
        
        # Store overall score
        results['overall'].append({
                'key_triple': key_triple,
                'engram_score': engram_score
        })
    
        # Store individual criterion scores
        for criterion in criteria:
            results[criterion].append({
                'key_triple': key_triple,
                f'engram_{criterion}': trigram_scores[criterion]
            })

    return results

def save_all_score_files(results, output_dir="../tables"):
    """Save overall and individual criterion scores to separate CSV files."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save overall scores
    #overall_file = f"{output_dir}/engram_3key_scores.csv"
    #overall_results = sorted(results['overall'], key=lambda x: x['key_triple'])
    #with open(overall_file, 'w', newline='', encoding='utf-8') as f:
    #    writer = csv.DictWriter(f, fieldnames=['key_triple', 'engram_score'])
    #    writer.writeheader()
    #    writer.writerows(overall_results)
    #print(f"✅ Saved overall scores to: {overall_file}")
    
    for criterion in criteria:
        criterion_file = f"{output_dir}/engram_3key_scores_{criterion}.csv"
        criterion_results = sorted(results[criterion], key=lambda x: x['key_triple'])
        
        with open(criterion_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key_triple', f'engram_{criterion}'])
            writer.writeheader()
            writer.writerows(criterion_results)
        
        print(f"✅ Saved {criterion} scores to: {criterion_file}")

def test_scoring_completeness():
    """Test that all criteria are always populated."""
    test_trigrams = ['QQQ', 'ABC', 'FJK', 'ZXC']
    for trigram in test_trigrams:
        scores = score_trigram(trigram)
        expected_keys = {'order'}
        actual_keys = set(scores.keys())
        if expected_keys != actual_keys:
            print(f"Missing keys for {trigram}: {expected_keys - actual_keys}")

def main():
    """Main entry point."""
    print("Prepare Engram key-triple scores (overall + individual criteria)")
    print("=" * 70)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-triples to compute: {len(keys)**3}")
    print()
    
    # Compute scores
    results = compute_key_triple_scores()
    
    # Save results
    output_dir = "../tables"
    save_all_score_files(results, output_dir)
    
    test_scoring_completeness()

    print(f"\n✅ Engram key-triple score generation complete!")
    print(f"   Overall scores: {output_dir}/engram_3key_scores.csv")
    print(f"   Individual criteria: {output_dir}/engram_3key_scores_*.csv")

if __name__ == "__main__":
    main()