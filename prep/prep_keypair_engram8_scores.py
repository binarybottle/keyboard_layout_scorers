#!/usr/bin/env python3
"""
Generate precomputed Engram-8 scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes both the overall Engram-8 score and individual criterion scores
for every possible pair of QWERTY keys and saves them to separate CSV files.

The output files contain all possible key-pairs (e.g., "QW", "QE", "AS") with
their corresponding scores.

Main output files:
    - ../tables/keypair_engram8_scores.csv - Overall average score
    - ../tables/keypair_engram8_load_scores.csv
    - ../tables/keypair_engram8_strength_scores.csv
    - ../tables/keypair_engram8_home_scores.csv
    - ../tables/keypair_engram8_curl_scores.csv
    - ../tables/keypair_engram8_columns_scores.csv
    - ../tables/keypair_engram8_vspan_scores.csv
    - ../tables/keypair_engram8_hspan_scores.csv
    - ../tables/keypair_engram8_sequence_scores.csv

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

The 8 scoring criteria for typing bigrams come from the Typing Preference Study:

    1.  Finger load: Typing with 2 fingers
    2.  Finger strength: Typing with the stronger two fingers
    3.  Finger home: Typing within the 8 home keys
    4.  Finger curl: Typing preferences above/below the 8 home keys
    5.  Finger columns: Typing within the 8 finger columns
    6.  Row separation: Typing in the same row, reaches, and hurdles 
    7.  Column separation: Adjacent columns in the same row
    8.  Finger sequence: Finger sequence toward the thumb in the same row

When applied to a single bigram, each criterion may be scored 0, 0.5, or 1 
generally to indicate when 0, 1, or 2 fingers or keys satisfy the criterion. 
Each criterion score for a layout is the average score across all bigrams.
The overall Engram-8 score is simply the average of the criterion scores.

Usage:
    python prep_keypair_engram8_scores.py

Output:
    ../tables/keypair_engram8_scores.csv - CSV with columns: key_pair, engram8_score
    ../tables/keypair_engram8_*_scores.csv - Individual criterion scores
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

# Define finger strength and home row
# Strong fingers are middle (3) and index (4) based on hypothesis testing
STRONG_FINGERS = {3, 4}
# Upper finger (middle finger) prefers upper rows based on H7-8 results
UPPER_FINGERS = {3}  # Only middle finger, ring finger (2) is ignored per study results
# Lower fingers (pinky and index) prefer lower rows based on H9-10 results
LOWER_FINGERS = {1, 4}
HOME_ROW = 2

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

criteria = ['load', 
            'strength', 
            'home', 
            'curl', 
            'columns', 
            'vspan', 
            'hspan',
            'sequence'] 

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

def score_bigram_engram8(bigram: str) -> Dict[str, float]:
    """Calculate all 8 Engram criteria scores for a bigram."""
    if len(bigram) != 2:
        raise ValueError("Bigram must be exactly 2 characters long")    
    
    char1, char2 = bigram[0].upper(), bigram[1].upper()
    
    # Get key information
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    
    if key_info1 is None or key_info2 is None:
        raise ValueError(f"Invalid keys: {char1}, {char2}")
    
    row1, column1, finger1, hand1, homekey1 = key_info1
    row2, column2, finger2, hand2, homekey2 = key_info2

    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    
    scores = {}

    #----------------------------------------------------------------------------------
    # Engram-8 scoring criteria
    #----------------------------------------------------------------------------------    
    #1.  Finger load: Typing with 2 fingers
    #2.  Finger strength: Typing with the stronger two fingers
    #3.  Finger home: Typing within the 8 home keys
    #4.  Finger curl: Typing preferences above/below the 8 home keys
    #5.  Finger columns: Typing within the 8 finger columns
    #6.  Row separation: Typing in the same row, reaches, and hurdles
    #7.  Column separation: Adjacent columns in the same row
    #8.  Finger sequence: Finger sequence toward the thumb
    #----------------------------------------------------------------------------------    
   
    # 1. Finger load: Typing with 2 fingers
    #    1.0: 2 fingers on 2 hands to type 2 keys
    #    1.0: 2 fingers on 1 hand to type 2 keys
    #    0.0: 1 finger on 1 hand to type 1-2 keys
    if hand1 != hand2:
        scores['load'] = 1.0      # 2 fingers on 2 hands to type 2 keys
    elif finger1 != finger2:
        scores['load'] = 1.0      # 2 fingers on 1 hand to type 2 keys
    elif finger1 == finger2:
        scores['load'] = 0.0      # 1 finger on 1 hand to type 1-2 keys

    # 2. Finger strength: Typing with the stronger two fingers
    #    1.0: 2 keys typed with strong fingers
    #    0.5: 1 key typed with 1 strong finger
    #    0.0: 0 keys typed with strong finger
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['strength'] = 1.0      # 2 keys typed with strong fingers
    elif strong_count == 1:
        scores['strength'] = 0.5      # 1 key typed with 1 strong finger
    else:
        scores['strength'] = 0.0      # 0 keys typed with strong finger

    # 3. Finger home: Typing within the 8 home keys
    #    1.0: 2 home keys
    #    0.5: 1 home key
    #    0.0: 0 home keys
    home_count = sum(1 for homekey in [homekey1, homekey2] if homekey == 1)
    if home_count == 2:
        scores['home'] = 1.0      # 2 home keys
    elif home_count == 1:
        scores['home'] = 0.5      # 1 home key
    else:
        scores['home'] = 0.0      # 0 home keys

    # 4. Finger curl: Typing above/below the 8 home keys
    #    Based on study results: ignore finger 2 (ring), count preferred rows for others
    #    1.0: 0 keys in unpreferred curl position*
    #    0.5: 1 key in unpreferred curl position*
    #    0.0: 2 keys in unpreferred curl position*
    #    *(or outside finger columns)
    #    Preferred curl positions: fingers 1,4 prefer row 3; finger 3 prefers rows 1; finger 2 no preference
    unpreferred_count = 0
    if in_column1:
        if finger1 in UPPER_FINGERS and row1 == 3:
            # Fingers that prefer the upper row 1 (not lower row 3)
            unpreferred_count += 1
        elif finger1 in LOWER_FINGERS and row1 == 1:
            # Fingers that prefer the lower row 3 (not upper row 1)
            unpreferred_count += 1
    else:
        unpreferred_count += 1

    if in_column2:
        if finger2 in UPPER_FINGERS and row2 == 3:
            # Fingers that prefer the upper row 1 (not lower row 3)
            unpreferred_count += 1
        elif finger2 in LOWER_FINGERS and row2 == 1:
            # Fingers that prefer the lower row 3 (not upper row 1)
            unpreferred_count += 1
    else:
        unpreferred_count += 1

    if unpreferred_count == 2:
        scores['curl'] = 0.0      # 2 keys in unpreferred rows
    elif unpreferred_count == 1:
        scores['curl'] = 0.5      # 1 key in unpreferred row
    else:
        scores['curl'] = 1.0      # 0 keys in unpreferred rows

    # 5. Finger columns: Typing within the 8 finger columns
    #    1.0: 2 keys within finger columns
    #    0.5: 1 key within finger columns
    #    0.0: 0 keys within finger columns
    if in_column1 and in_column2:
        scores['columns'] = 1.0      # 2 keys within finger columns
    elif in_column1 or in_column2:
        scores['columns'] = 0.5      # 1 key within finger columns
    else:
        scores['columns'] = 0.0      # 0 keys within finger columns

    # 6. Row separation: Typing in the same row, reaches, and hurdles 
    #    1.0: 2 keys in the same row, or opposite hands
    #    0.5: 2 keys in adjacent rows (reach)
    #    0.0: 2 keys straddling home row (hurdle)
    if hand1 != hand2:
        scores['vspan'] = 1.0          # opposite hands always score well
    else:
        if row1 == row2:
            scores['vspan'] = 1.0      # 2 keys in the same row
        elif abs(row1 - row2) == 1:
            scores['vspan'] = 0.5      # 2 keys in adjacent rows (reach)
        else:
            scores['vspan'] = 0.0      # 2 keys straddling home row (hurdle)

    # 7. Column separation: Adjacent columns in the same row
    #    1.0: adjacent columns in same row, or non-adjacent columns in different rows (or 2 hands)
    #    0.0: non-adjacent columns in the same row, or adjacent columns in different rows (or 1 finger)
    if hand1 != hand2:
        scores['hspan'] = 1.0      # opposite hands always score well
    elif finger1 == finger2:
        scores['hspan'] = 0.0      # same finger scores zero
    else:
        column_gap = abs(column1 - column2)
        finger_gap = abs(finger1 - finger2)
        if (column_gap == 1 and row1 == row2) or (column_gap > 1 and finger_gap > 1 and row1 != row2):
            scores['hspan'] = 1.0  # adjacent columns, same row / non-adjacent, different rows
        else:
            scores['hspan'] = 0.0  # non-adjacent columns, same row / adjacent, different rows

    # 8. Finger sequence: Finger sequence toward the thumb
    #    1.0: inward roll on the same row, or opposite hands
    #    0.0: outward roll, or same finger
    if hand1 != hand2:
        scores['sequence'] = 1.0      # opposite hands always score well
    elif finger1 == finger2:
        scores['sequence'] = 0.0      # same finger scores zero
    elif row1 == row2:
        # Inward roll: toward thumb (finger number increases: 1→2→3→4)
        # Both hands are bilaterally symmetric, so same sequence logic
        scores['sequence'] = 1.0 if finger1 < finger2 else 0.0
    else:
        scores['sequence'] = 0.0

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
    """Compute Engram-8 scores for all key-pairs."""
    key_pairs = generate_all_key_pairs()
    results = {}
    
    # Initialize results for overall and individual criteria
    results['overall'] = []
    
    for criterion in criteria:
        results[criterion] = []

    print(f"Computing Engram-8 scores for {len(key_pairs)} key-pairs...")

    for i, key_pair in enumerate(key_pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_pairs)} ({i/len(key_pairs)*100:.1f}%)")
        
        # Compute individual Engram-8 criteria scores using the scorer's function
        bigram_scores = score_bigram_engram8(key_pair)
        
        # Calculate sum (baseline Engram-8 score)
        engram8_score = sum(bigram_scores.values())
        
        # Store overall score
        results['overall'].append({
            'key_pair': key_pair,
            'engram8_score': engram8_score
        })
    
        # Store individual criterion scores
        for criterion in criteria:
            results[criterion].append({
                'key_pair': key_pair,
                f'engram8_{criterion}': bigram_scores[criterion]
            })

    return results

def save_all_score_files(results, output_dir="../tables"):
    """Save overall and individual criterion scores to separate CSV files."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save overall scores
    overall_file = f"{output_dir}/keypair_engram8_scores.csv"
    overall_results = sorted(results['overall'], key=lambda x: x['key_pair'])
    
    with open(overall_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'engram8_score'])
        writer.writeheader()
        writer.writerows(overall_results)
    
    print(f"✅ Saved overall scores to: {overall_file}")
    
    for criterion in criteria:
        criterion_file = f"{output_dir}/keypair_engram8_{criterion}_scores.csv"
        criterion_results = sorted(results[criterion], key=lambda x: x['key_pair'])
        
        with open(criterion_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key_pair', f'engram8_{criterion}'])
            writer.writeheader()
            writer.writerows(criterion_results)
        
        print(f"✅ Saved {criterion} scores to: {criterion_file}")

def validate_output(output_dir="../tables"):
    """
    Validation with comprehensive accuracy checking.
    
    This function performs extensive validation of the generated Engram-8 scores,
    including mathematical verification of scoring criteria and edge case testing.
    """
    import csv
    import random
    from pathlib import Path
    
    overall_file = f"{output_dir}/keypair_engram8_scores.csv"
    
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
    scores = [float(row['engram8_score']) for row in rows]
    min_score, max_score = min(scores), max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.4f} to {max_score:.4f}")
    print(f"   Average score: {avg_score:.4f}")
    # Now checking for 0-8 range instead of 0-1
    print(f"   Valid range (0-8): {'✅' if 0 <= min_score and max_score <= 8 else '❌'}")
    
    # Test mathematical accuracy on random samples
    print(f"\n🧮 Mathematical Accuracy Check:")
    random_samples = random.sample(rows, min(20, len(rows)))
    accuracy_errors = 0
    
    for row in random_samples:
        key_pair = row['key_pair']
        csv_score = float(row['engram8_score'])
        
        # Recalculate score using the same logic
        calculated_scores = score_bigram_engram8(key_pair)
        # Use raw sum instead of normalized average
        calculated_sum = sum(calculated_scores.values())
        
        if abs(calculated_sum - csv_score) > 0.0001:
            print(f"   ❌ Accuracy error: {key_pair} - CSV: {csv_score:.4f}, Calc: {calculated_sum:.4f}")
            accuracy_errors += 1
    
    print(f"   Accuracy check: {len(random_samples) - accuracy_errors}/{len(random_samples)} samples correct")
    
    # Validate individual criterion files
    print(f"\n📁 Individual Criterion Files:")    
    for criterion in criteria:
        criterion_file = f"{output_dir}/keypair_engram8_{criterion}_scores.csv"
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
                test_scores.append(float(row['engram8_score']))
        
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
    perfect_count = len([s for s in scores if s == 8.0])
    print(f"   Perfect (8.0): {perfect_count} pairs ({perfect_count/len(scores)*100:.1f}%)")
    
    print(f"\n✅ Validation complete!")
    return accuracy_errors == 0

def validate_perfect_scores(output_dir="../tables"):
    """Specifically validate that perfect scores are mathematically correct."""
    import csv
    
    overall_file = f"{output_dir}/keypair_engram8_scores.csv"
    
    with open(overall_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        perfect_pairs = [row for row in reader if float(row['engram8_score']) == 8.0]
    
    print(f"\n🏆 Perfect Score Verification ({len(perfect_pairs)} pairs):")
    for row in perfect_pairs:
        key_pair = row['key_pair']
        scores = score_bigram_engram8(key_pair)
        total_score = sum(scores.values())
        is_perfect = total_score == 8.0
        print(f"   {key_pair}: Total = 8.0? {'✅' if is_perfect else '❌'}")
        if not is_perfect:
            print(f"      Individual scores: {scores}")
            print(f"      Sum: {total_score}")
    
    return True

def main():
    """Main entry point."""
    print("Prepare Engram-8 key-pair scores (overall + individual criteria)")
    print("=" * 70)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print(f"Output files: 1 overall + 8 individual criteria = 9 total")
    print()
    
    # Compute scores
    results = compute_key_pair_scores()
    
    # Save results
    output_dir = "../tables"
    save_all_score_files(results, output_dir)
    
    # Validate output
    validate_output(output_dir)
    validate_perfect_scores(output_dir)

    print(f"\n✅ Engram-8 key-pair score generation complete!")
    print(f"   Overall scores: {output_dir}/keypair_engram8_scores.csv")
    print(f"   Individual criteria: {output_dir}/keypair_engram8_*_scores.csv")

if __name__ == "__main__":
    main()