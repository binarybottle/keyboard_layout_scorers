#!/usr/bin/env python3
"""
Generate precomputed Dvorak-9 scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes the unweighted average Dvorak-9 score for every possible
combination of QWERTY keys and saves them to output/keypair_dvorak9_scores.csv.

The output file contains all possible key-pairs (e.g., "QW", "QE", "AS") with
their corresponding Dvorak-9 scores (average of the 9 individual criteria).

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

The 9 scoring criteria for typing bigrams are derived from Dvorak's
"Typing Behavior" book and patent (1936) (0-1, higher = better performance):
    1. Hands - favor alternating hands over same hand
    2. Fingers - avoid same finger repetition  
    3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
    4. Don't cross home - avoid crossing over the home row (hurdling)
    5. Same row - favor typing within the same row
    6. Home row - favor using the home row
    7. Columns - favor fingers staying in their designated columns
    8. Strum - favor inward rolls over outward rolls (same hand)
    9. Strong fingers - favor stronger fingers over weaker ones

Usage:
    python prep_keypair_dvorak9_scores.py

Output:
    output/keypair_dvorak9_scores.csv - CSV with columns: key_pair, dvorak9_score
"""

import csv
import os
from pathlib import Path
from typing import Dict

# QWERTY keyboard layout with (row, finger, hand) mapping (same as original)
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

# Define finger strength and home row
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

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing."""
    # Use the same key set as the original generator (excludes numbers)
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def generate_all_key_pairs():
    """Generate all possible QWERTY key-pair combinations."""
    keys = get_all_qwerty_keys()
    key_pairs = []
    
    for key1 in keys:
        for key2 in keys:
            key_pairs.append(key1 + key2)
    
    return key_pairs

def compute_key_pair_scores():
    """Compute Dvorak-9 scores for all key-pairs."""
    key_pairs = generate_all_key_pairs()
    results = []
    
    print(f"Computing Dvorak-9 scores for {len(key_pairs)} key-pairs...")
    
    for i, key_pair in enumerate(key_pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_pairs)} ({i/len(key_pairs)*100:.1f}%)")
        
        # Compute individual Dvorak-9 criteria scores using the scorer's function
        bigram_scores = score_bigram_dvorak9(key_pair)
        
        # Calculate unweighted average (baseline Dvorak-9 score)
        dvorak9_score = sum(bigram_scores.values()) / len(bigram_scores)
        
        results.append({
            'key_pair': key_pair,
            'dvorak9_score': dvorak9_score
        })
    
    return results

def save_key_pair_scores(results, output_file="output/keypair_dvorak9_scores.csv"):
    """Save key-pair scores to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'dvorak9_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def validate_output(output_file="output/keypair_dvorak9_scores.csv"):
    """Validate the generated output file."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Validation Results:")
    print(f"   Total key-pairs: {len(rows)}")
    
    # Check for expected number of combinations
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    print(f"   Expected count: {expected_count}")
    print(f"   Match: {'✅' if len(rows) == expected_count else '❌'}")
    
    # Check score range (should be 0-1)
    scores = [float(row['dvorak9_score']) for row in rows]
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.3f} to {max_score:.3f}")
    print(f"   Average score: {avg_score:.3f}")
    print(f"   Valid range (0-1): {'✅' if 0 <= min_score and max_score <= 1 else '❌'}")
    
    # Show some examples
    print(f"\n📝 Sample key-pairs and scores:")
    for i in range(0, min(10, len(rows)), max(1, len(rows)//10)):
        row = rows[i]
        print(f"   {row['key_pair']}: {float(row['dvorak9_score']):.3f}")
    
    return True

def main():
    """Main entry point."""
    print("Prepare Dvorak-9 key-pair scores")
    print("=" * 50)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print()
    
    # Compute scores
    results = compute_key_pair_scores()
    
    # Save results
    output_file = "output/keypair_dvorak9_scores.csv"
    save_key_pair_scores(results, output_file)
    
    # Validate output
    validate_output(output_file)
    
    print(f"\n✅ Dvorak-9 key-pair score generation complete: {output_file}")

if __name__ == "__main__":
    main()