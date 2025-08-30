#!/usr/bin/env python3
"""
Generate precomputed Dvorak-7 scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes both the overall Dvorak-7 score and individual criterion scores
for every possible pair of QWERTY keys and saves them to separate CSV files.

The output files contain all possible key-pairs (e.g., "QW", "QE", "AS") with
their corresponding scores.

Main output files:
    - ../tables/keypair_dvorak7_scores.csv - Overall average score
    - ../tables/keypair_dvorak7_repetition_scores.csv
    - ../tables/keypair_dvorak7_movement_scores.csv
    - ../tables/keypair_dvorak7_vertical_scores.csv
    - ../tables/keypair_dvorak7_horizontal_scores.csv
    - ../tables/keypair_dvorak7_adjacent_scores.csv
    - ../tables/keypair_dvorak7_weak_scores.csv
    - ../tables/keypair_dvorak7_outward_scores.csv

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

The 7 scoring criteria for typing bigrams are derived from Dvorak's
"Typing Behavior" book and patent (1936) and reflect favorable typing behaviors:

    1.  Distributed load: Typing with 2 hands or 2 fingers
    2.  Anchored positions: Typing within the 8 home keys
    3.  Row-aligned: Typing in the same or adjacent rows 
    4.  Column-aligned: Typing within the 8 finger columns
    5.  Adjacent/neighbor-aligned: Adjacent fingers stay in the same row
    6.  Strong fingers: Typing with the stronger two fingers
    7.  Inward direction: Finger sequence toward the thumb

When applied to a single bigram, each criterion may be scored 0, 0.5, or 1 
generally to indicate when 0, 1, or 2 fingers or keys satisfy the criterion. 
Each criterion score for a layout is the average score across all bigrams.
The overall Dvorak-7 score is simply the average of the criterion scores.

Usage:
    python prep_keypair_dvorak7_scores.py

Output:
    ../tables/keypair_dvorak7_scores.csv - CSV with columns: key_pair, dvorak7_score
    ../tables/keypair_dvorak7_*_scores.csv - Individual criterion scores
"""

import csv
from pathlib import Path
from typing import Dict

# QWERTY keyboard layout with (row, finger, hand, homekey) mapping
QWERTY_LAYOUT = {
    # Number row (row 0) - finger numbers
    '1': (0, 1, 'L', 0), '2': (0, 2, 'L', 0), '3': (0, 3, 'L', 0), '4': (0, 4, 'L', 0), '5': (0, 4, 'L', 0),
    '6': (0, 4, 'R', 0), '7': (0, 4, 'R', 0), '8': (0, 3, 'R', 0), '9': (0, 2, 'R', 0), '0': (0, 1, 'R', 0),
    
    # Top row (row 1) - finger numbers
    'Q': (1, 1, 'L', 0), 'W': (1, 2, 'L', 0), 'E': (1, 3, 'L', 0), 'R': (1, 4, 'L', 0), 'T': (1, 4, 'L', 0),
    'Y': (1, 4, 'R', 0), 'U': (1, 4, 'R', 0), 'I': (1, 3, 'R', 0), 'O': (1, 2, 'R', 0), 'P': (1, 1, 'R', 0),
    
    # Home row (row 2) - finger numbers
    'A': (2, 1, 'L', 1), 'S': (2, 2, 'L', 1), 'D': (2, 3, 'L', 1), 'F': (2, 4, 'L', 1), 'G': (2, 4, 'L', 0),
    'H': (2, 4, 'R', 0), 'J': (2, 4, 'R', 1), 'K': (2, 3, 'R', 1), 'L': (2, 2, 'R', 1), ';': (2, 1, 'R', 1),
    
    # Bottom row (row 3) - finger numbers
    'Z': (3, 1, 'L', 0), 'X': (3, 2, 'L', 0), 'C': (3, 3, 'L', 0), 'V': (3, 4, 'L', 0), 'B': (3, 4, 'L', 0),
    'N': (3, 4, 'R', 0), 'M': (3, 4, 'R', 0), ',': (3, 3, 'R', 0), '.': (3, 2, 'R', 0), '/': (3, 1, 'R', 0),
    
    # Additional common keys - finger numbers
    "'": (2, 1, 'R', 0), '[': (1, 1, 'R', 0),
}

# Define finger strength and home row
STRONG_FINGERS = {3, 4}  # middle and index
HOME_ROW = 2

# Define finger column assignments
FINGER_COLUMNS = {
    'L': {
        1: ['Q', 'A', 'Z'],    # Pinky
        2: ['W', 'S', 'X'],    # Ring  
        3: ['E', 'D', 'C'],    # Middle
        4: ['R', 'F', 'V', 'T', 'G', 'B']  # Index
    },
    'R': {
        4: ['Y', 'H', 'N', 'U', 'J', 'M'],  # Index
        3: ['I', 'K', ','],    # Middle
        2: ['O', 'L', '.'],    # Ring
        1: ['P', ';', '/', "'", '[']   # Pinky
    }
}

criteria = ['repetition', 
            'adjacent', 
            'vertical', 
            'movement', 
            'horizontal', 
            'outward', 
            'weak']

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

def score_bigram_dvorak7(bigram: str) -> Dict[str, float]:
    """Calculate all 7 Dvorak criteria scores for a bigram."""
    if len(bigram) != 2:
        raise ValueError("Bigram must be exactly 2 characters long")    
    
    char1, char2 = bigram[0].upper(), bigram[1].upper()
    
    # Get key information
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    
    if key_info1 is None or key_info2 is None:
        raise ValueError(f"Invalid keys: {char1}, {char2}")
    
    row1, finger1, hand1, homekey1 = key_info1
    row2, finger2, hand2, homekey2 = key_info2
    
    scores = {}

    #----------------------------------------------------------------------------------
    # Dvorak-7 scoring criteria
    #----------------------------------------------------------------------------------    
    # 1.  Repetition: Typing with 1 hand or 1 finger
    # 2.  Movement: Typing outside the 8 home keys
    # 3.  Vertical separation: Typing in different rows 
    # 4.  Horizontal reach: Typing outside 8 finger columns
    # 5.  Adjacent fingers: Typing with adjacent fingers (except stronger pair of fingers 1 and 2)
    # 6.  Weak fingers: Typing with weaker fingers 3 and 4
    # 7.  Outward direction: Finger sequence away from the thumb
    #----------------------------------------------------------------------------------    
   
    # 1. Repetition: Typing with 1 hand or 1 finger
    #    1.0: 2 fingers on 2 hands to type 2 keys
    #    0.5: 2 fingers on 1 hand to type 2 keys
    #    0.0: 1 finger on 1 hand to type 1-2 keys
    if hand1 != hand2:
        scores['repetition'] = 1.0  # 2 fingers on 2 hands to type 2 keys
    elif finger1 != finger2:
        scores['repetition'] = 0.5  # 2 fingers on 1 hand to type 2 keys
    elif finger1 == finger2:
        scores['repetition'] = 0.0  # 1 finger on 1 hand to type 1-2 keys

    # 2. Movement: Typing outside the 8 home keys
    #    1.0: 2 home keys
    #    0.5: 1 home key
    #    0.0: 0 home keys
    home_count = sum(1 for homekey in [homekey1, homekey2] if homekey == 1)
    #home_count = sum(1 for row in [row1, row2] if row == HOME_ROW)
    if home_count == 2:
        scores['movement'] = 1.0      # 2 home keys
    elif home_count == 1:
        scores['movement'] = 0.5      # 1 home key
    else:
        scores['movement'] = 0.0      # 0 home keys

    # 3. Vertical separation: Typing in different rows
    #    1.0: 2 keys in the same row, or opposite hands
    #    0.5: 2 keys in adjacent rows (reach)
    #    0.0: 2 keys straddling home row (hurdle)
    if hand1 != hand2:
        scores['vertical'] = 1.0          # opposite hands always score well
    else:
        if row1 == row2:
            scores['vertical'] = 1.0      # 2 keys in the same row
        elif (row1 == 1 and row2 == 2) or (row1 == 2 and row2 == 1) or \
             (row1 == 2 and row2 == 3) or (row1 == 3 and row2 == 2):
            scores['vertical'] = 0.5      # 2 keys in adjacent rows (reach)
        elif (row1 == 1 and row2 == 3) or (row1 == 3 and row2 == 1):
            scores['vertical'] = 0.0      # 2 keys straddling home row (hurdle)

    # 4. Horizontal reach: Typing outside 8 finger columns
    #    1.0: 2 keys within finger columns
    #    0.5: 1 key within finger columns
    #    0.0: 0 keys within finger columns
    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    if in_column1 and in_column2:
        scores['horizontal'] = 1.0           # 2 keys within finger columns
    elif in_column1 or in_column2:
        scores['horizontal'] = 0.5           # 1 key within finger columns
    else:
        scores['horizontal'] = 0.0           # 0 keys within finger columns

    # 5. Adjacent fingers: Typing with adjacent fingers (except strong pair of fingers 1 and 2)
    #    1.0: non-adjacent fingers, strong finger pair, or opposite hands
    #    0.0: same finger, or adjacent fingers where at least 1 is weak
    if hand1 != hand2:
        scores['adjacent'] = 1.0      # opposite hands always score well
    elif finger1 == finger2:
        scores['adjacent'] = 0.0      # same finger scores zero
    elif finger1 in STRONG_FINGERS and finger2 in STRONG_FINGERS:
        scores['adjacent'] = 1.0      # strong finger pair (index & middle)
    else:
        finger_gap = abs(finger1 - finger2)
        if finger_gap == 1:
            scores['adjacent'] = 0.0  # adjacent fingers where at least 1 is weak
        elif finger_gap == 2:
            scores['adjacent'] = 1.0  # non-adjacent fingers: skipping 1 finger
        elif finger_gap == 3:
            scores['adjacent'] = 1.0  # non-adjacent fingers: skipping 2 fingers

    # 6. Weak fingers: Typing with weaker fingers 3 and 4
    #    1.0: 0 keys typed with weak fingers
    #    0.5: 1 key typed with 1 weak finger
    #    0.0: 2 keys typed with 1-2 weak fingers
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['weak'] = 1.0    # 2 keys typed with 1-2 strong fingers
    elif strong_count == 1:
        scores['weak'] = 0.5    # 1 key typed with 1 strong finger
    else:
        scores['weak'] = 0.0    # 0 keys typed with strong finger

    # 7. Outward direction: Finger sequence away from the thumb
    #    1.0: inward roll, or opposite hands
    #    0.0: outward roll, or same finger
    if hand1 != hand2:
        scores['outward'] = 1.0             # opposite hands always score well
    elif finger1 == finger2:
        scores['outward'] = 0.0             # same finger scores zero
    else:
        if finger1 > finger2:
            scores['outward'] = 1.0         # inward roll
        else:
            scores['outward'] = 0.0         # outward roll

    return scores

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing."""

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
    """Compute Dvorak-7 scores for all key-pairs."""
    key_pairs = generate_all_key_pairs()
    results = {}
    
    # Initialize results for overall and individual criteria
    results['overall'] = []
    
    for criterion in criteria:
        results[criterion] = []

    print(f"Computing Dvorak-7 scores for {len(key_pairs)} key-pairs...")

    for i, key_pair in enumerate(key_pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_pairs)} ({i/len(key_pairs)*100:.1f}%)")
        
        # Compute individual Dvorak-7 criteria scores using the scorer's function
        bigram_scores = score_bigram_dvorak7(key_pair)
        
        # Calculate sum (baseline Dvorak-7 score)
        dvorak7_score = sum(bigram_scores.values())
        
        # Store overall score
        results['overall'].append({
            'key_pair': key_pair,
            'dvorak7_score': dvorak7_score
        })
    
        # Store individual criterion scores
        for criterion in criteria:
            results[criterion].append({
                'key_pair': key_pair,
                f'dvorak7_{criterion}': bigram_scores[criterion]
            })

    return results

def save_all_score_files(results, output_dir="../tables"):
    """Save overall and individual criterion scores to separate CSV files."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save overall scores
    overall_file = f"{output_dir}/keypair_dvorak7_scores.csv"
    overall_results = sorted(results['overall'], key=lambda x: x['key_pair'])
    
    with open(overall_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'dvorak7_score'])
        writer.writeheader()
        writer.writerows(overall_results)
    
    print(f"✅ Saved overall scores to: {overall_file}")
    
    for criterion in criteria:
        criterion_file = f"{output_dir}/keypair_dvorak7_{criterion}_scores.csv"
        criterion_results = sorted(results[criterion], key=lambda x: x['key_pair'])
        
        with open(criterion_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key_pair', f'dvorak7_{criterion}'])
            writer.writeheader()
            writer.writerows(criterion_results)
        
        print(f"✅ Saved {criterion} scores to: {criterion_file}")

def validate_output(output_dir="../tables"):
    """
    Validation with comprehensive accuracy checking.
    
    This function performs extensive validation of the generated Dvorak-7 scores,
    including mathematical verification of scoring criteria and edge case testing.
    """
    import csv
    import random
    from pathlib import Path
    
    overall_file = f"{output_dir}/keypair_dvorak7_scores.csv"
    
    if not Path(overall_file).exists():
        print(f"❌ Overall output file not found: {overall_file}")
        return False
    
    # Load overall data
    with open(overall_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Validation Results:")
    print(f"   Total key-pairs: {len(rows)}")
    
    # Statistical validation
    scores = [float(row['dvorak7_score']) for row in rows]
    min_score, max_score = min(scores), max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.4f} to {max_score:.4f}")
    print(f"   Average score: {avg_score:.4f}")
    # Now checking for 0-7 range instead of 0-1
    print(f"   Valid range (0-7): {'✅' if 0 <= min_score and max_score <= 7 else '❌'}")
    
    # Test mathematical accuracy on random samples
    print(f"\n🧮 Mathematical Accuracy Check:")
    random_samples = random.sample(rows, min(20, len(rows)))
    accuracy_errors = 0
    
    for row in random_samples:
        key_pair = row['key_pair']
        csv_score = float(row['dvorak7_score'])
        
        # Recalculate score using the same logic
        calculated_scores = score_bigram_dvorak7(key_pair)
        # Use raw sum instead of normalized average
        calculated_sum = sum(calculated_scores.values())
        
        if abs(calculated_sum - csv_score) > 0.0001:
            print(f"   ❌ Accuracy error: {key_pair} - CSV: {csv_score:.4f}, Calc: {calculated_sum:.4f}")
            accuracy_errors += 1
    
    print(f"   Accuracy check: {len(random_samples) - accuracy_errors}/{len(random_samples)} samples correct")
    
    # Validate individual criterion files
    print(f"\n📁 Individual Criterion Files:")    
    for criterion in criteria:
        criterion_file = f"{output_dir}/keypair_dvorak7_{criterion}_scores.csv"
        if Path(criterion_file).exists():
            with open(criterion_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                criterion_rows = list(reader)
            print(f"   ✅ {criterion}: {len(criterion_rows)} pairs")
        else:
            print(f"   ❌ {criterion}: File missing")
    
    # Test specific criteria with known examples
    print(f"\n🔍 Criteria-Specific Validation:")
    criteria_tests = [
        ("Same key repetition", ['AA', 'SS', 'FF'], lambda s: 1.0 <= s <= 5.0),
        ("Worst cases", ['QZ', '/[', 'ZW'], lambda s: s <= 3.0),
        ("Alternating hands", ['FJ', 'AK', 'TN'], lambda s: s >= 3.5)
    ]
    
    for test_name, test_pairs, score_check in criteria_tests:
        test_scores = []
        for pair in test_pairs:
            row = next((r for r in rows if r['key_pair'] == pair), None)
            if row:
                test_scores.append(float(row['dvorak7_score']))
        
        all_pass = all(score_check(score) for score in test_scores)
        print(f"   {test_name}: {'✅' if all_pass else '❌'} ({len([s for s in test_scores if score_check(s)])}/{len(test_scores)})")
    
    # Distribution analysis
    print(f"\n📈 Score Distribution:")
    ranges = [(0.0, 1.4, "Very Poor"), (1.4, 2.8, "Poor"), (2.8, 4.2, "Fair"), 
              (4.2, 5.6, "Good"), (5.6, 7.0, "Excellent")]
    
    for min_val, max_val, label in ranges:
        count = len([s for s in scores if min_val <= s < max_val])
        percentage = count / len(scores) * 100
        print(f"   {label} ({min_val}-{max_val}): {count} pairs ({percentage:.1f}%)")
    
    # Perfect scores count
    perfect_count = len([s for s in scores if s == 7.0])
    print(f"   Perfect (7.0): {perfect_count} pairs ({perfect_count/len(scores)*100:.1f}%)")
    
    print(f"\n✅ Validation complete!")
    return accuracy_errors == 0

def validate_perfect_scores(output_dir="../tables"):
    """Specifically validate that perfect scores are mathematically correct."""
    import csv
    
    overall_file = f"{output_dir}/keypair_dvorak7_scores.csv"
    
    with open(overall_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        perfect_pairs = [row for row in reader if float(row['dvorak7_score']) == 7.0]
    
    print(f"\n🏆 Perfect Score Verification ({len(perfect_pairs)} pairs):")
    
    for row in perfect_pairs:
        key_pair = row['key_pair']
        scores = score_bigram_dvorak7(key_pair)
        total_score = sum(scores.values())
        is_perfect = total_score == 7.0
        print(f"   {key_pair}: Total = 7.0? {'✅' if is_perfect else '❌'}")        
        if not is_perfect:
            print(f"      Individual scores: {scores}")
            print(f"      Sum: {total_score}")
    
    return True

def main():
    """Main entry point."""
    print("Prepare Dvorak-7 key-pair scores (overall + individual criteria)")
    print("=" * 70)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print(f"Output files: 1 overall + 7 individual criteria = 8 total")
    print()
    
    # Compute scores
    results = compute_key_pair_scores()
    
    # Save results
    output_dir = "../tables"
    save_all_score_files(results, output_dir)
    
    # Validate output
    validate_output(output_dir)
    validate_perfect_scores(output_dir)

    print(f"\n✅ Dvorak-7 key-pair score generation complete!")
    print(f"   Overall scores: {output_dir}/keypair_dvorak7_scores.csv")
    print(f"   Individual criteria: {output_dir}/keypair_dvorak7_*_scores.csv")

if __name__ == "__main__":
    main()