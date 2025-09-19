#!/usr/bin/env python3
"""
Generate precomputed Engram scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes both the overall Engram score and individual criterion scores
for every possible pair of QWERTY keys and saves them to separate CSV files.

Scoring criteria from the Bigram Typing Preference Study for use as MOO objectives:
    1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    3. Same-row finger order and column separation
       - (empirical analysis of left key-pairs toward vs. away from the thumb)
       - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 

Scoring criteria isolated from above:
    4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns; see #3)
    5. Outside reach (empirical analysis of left-hand lateral stretches outside finger-columns; see #1)

Additional simple scoring criteria considered problematic by the Alternative Keyboard Layout communities:
    6. Scissor (adjacent column hurdle; ignores all other awkward finger mechanics)
    7. Half-scissor (adjacent column reach; ignores all other awkward finger mechanics)

Collectively, these criteria address finger mechanics generally considered problematic:
  - uncomfortable key/finger positions: #1
  - uncomfortable transitions:
    - reaches and hurdles: #2
    - skipping within a row: #3
    - outward rolls: #3 (within a row)
    - same-finger bigrams: #4 (and #3)
    - lateral stretches: #5 (and #1, #3)
    - scissors: #6, #7 (row separation is also explicitly penalized by #2)
  - uncomfortable patterns:
    - redirects: addressed by the trigram scoring criteria
   
Each criterion score for a layout is the average score across all bigrams.
The overall Engram score is simply the average of the criterion scores.

The output files contain all possible key-pairs (e.g., "QW", "QE", "AS") with
their corresponding scores.

Main output files:
    - ../tables/engram_2key_scores.csv - Overall average score
    - ../tables/engram_2key_scores_key_preference.csv
    - ../tables/engram_2key_scores_row_separation.csv
    - ../tables/engram_2key_scores_same_row.csv
    - ../tables/engram_2key_scores_same_finger.csv
    - ../tables/engram_2key_scores_outside.csv
    - ../tables/engram_2key_scores_scissor.csv
    - ../tables/engram_2key_scores_half_scissor.csv

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

Usage:
    python prep_keypair_engram3of4_scores.py

Output:
    ../tables/engram_2key_scores.csv - CSV with columns: key_pair, engram_score
    ../tables/engram_2key_scores_*.csv - Individual criterion scores
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
    '[': (1, 0, 1, 'R', 0), "'": (2, 0, 1, 'R', 0), 
}

qwerty_home_blocks = ['Q', 'W', 'E', 'R', 'U', 'I', 'O', 'P', 
                      'A', 'S', 'D', 'F', 'J', 'K', 'L', ';', 
                      'Z', 'X', 'C', 'V', 'M', ',', '.', '/']

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

criteria = ['key_preference', 'row_separation', 'same_row',
            'outside', 'same_finger',
            'scissor', 'half_scissor'] 
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

def score_bigram(bigram: str) -> Dict[str, float]:
    """Calculate Engram criteria scores for a bigram."""
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

    row_gap = abs(row1 - row2)
    column_gap = abs(column1 - column2)
    
    inside_columns1 = is_finger_in_column(char1, finger1, hand1)
    inside_columns2 = is_finger_in_column(char2, finger2, hand2)
    
    scores = {}

    #----------------------------------------------------------------------------------
    # Engram's bigram scoring criteria
    #----------------------------------------------------------------------------------    
    # 1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    # 2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    # 3. Same-row finger order and column separation
    #    - (empirical analysis of left key-pairs toward vs. away from the thumb)
    #    - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 
    # 4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns)
    # 5. Outside reach (empirical analysis of left-hand lateral stretches outside finger-columns)
    # 6. Scissor (adjacent column hurdle; ignores all other awkward finger mechanics)
    # 7. Half-scissor (adjacent column reach; ignores all other awkward finger mechanics)
    #----------------------------------------------------------------------------------    
   
    # 1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    #    0.137 - 1.000: keys inside the 8 finger-columns
    #    0.000: keys outside the 8 finger-columns 
    tier_values = {
        'F': 1.000, 'J': 1.000,
        'D': 0.870, 'K': 0.870,
        'E': 0.646, 'I': 0.646,
        'S': 0.646, 'L': 0.646,
        'V': 0.568, 'M': 0.568,
        'R': 0.568, 'U': 0.568,
        'W': 0.472, 'O': 0.472,
        'A': 0.410, ';': 0.410,
        'C': 0.410, ',': 0.410,
        'Z': 0.137, '/': 0.137,
        'Q': 0.137, 'P': 0.137,
        'X': 0.137, '.': 0.137
    }

    key_score = 0
    for key in [char1, char2]:
        key_score += tier_values.get(key, 0)  # Get tier value or 0 if not found

    scores['key_preference'] = key_score / 2.0  # Average over 2 keys

    # 2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    #    1.000: 2 hands
    #    1.000: 2 keys in the same row
    #    0.588: 2 keys in adjacent rows (reach)
    #    0.000: 2 keys straddling home row (hurdle)
    if hand1 != hand2:
        scores['row_separation'] = 1.0        # Two hands
    else:
        if row_gap == 0:
            scores['row_separation'] = 1.0    # Same row
        elif row_gap == 1:
            scores['row_separation'] = 0.588  # Adjacent row (reach)
        else:
            scores['row_separation'] = 0.0    # Skip row (hurdle)

    # 3. Same-row finger order and column separation
    #    - (empirical analysis of left key-pairs toward vs. away from the thumb)
    #    - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 
    #    1.000: 2 hands
    #    1.000: adjacent columns, inward roll, in the same row
    #    0.779: adjacent columns, outward roll, in the same row
    #    0.811: remote columns, inward roll, in the same row
    #    0.779 x 0.811: remote columns, outward roll, in the same row
    #    0.500: different rows, different fingers
    #    0.000: same finger
    if hand1 != hand2:
        scores['same_row'] = 1.0        # Two hands
    elif finger1 == finger2:
        scores['same_row'] = 0.0        # Same finger
    elif row_gap == 0:  # Same row logic

        # Apply same-row finger order/direction (stronger effect)
        if finger2 > finger1:           # Inward
            scores['same_row'] = 1.0
        elif finger2 < finger1:         # Outward  
            scores['same_row'] = 0.779
        
        # Apply column separation penalty (weaker effect)
        if column_gap >= 2:             # Remote columns
            scores['same_row'] *= 0.811

    else:
        scores['same_row'] = 0.5        # Different rows, different fingers

    # 4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns)
    #    1.0: 2 hands
    #    1.0: 2 fingers
    #    0.0: 1 finger
    if hand1 != hand2:
        scores['same_finger'] = 1.0          # Two hands
    elif finger1 != finger2:
        scores['same_finger'] = 1.0          # Two fingers
    else:
        scores['same_finger'] = 0.0          # Same finger

    # 5. Outside reach (empirical analysis of left-hand lateral stretches outside finger-columns)
    #    1.000: 0 outside keys
    #    0.846: 1 outside key
    #    0.716: 2 outside keys
    qwerty_home_blocks_set = set(qwerty_home_blocks)  # Convert to set for O(1) lookup performance

    # Count how many keys are outside the home blocks
    outside_count = sum(1 for key in [char1, char2] if key not in qwerty_home_blocks_set)

    # Apply score based on count
    outside_scores = {0: 1.0, 1: 0.846, 2: 0.716}
    scores['outside'] = outside_scores[outside_count]

    # 6. Scissor (adjacent column hurdle; ignores all other awkward finger mechanics)
    #    1.0: 2 hands
    #    1.0: non-scissor
    #    0.0: scissor
    scores['scissor'] = 1.0              # Default score
    if hand1 != hand2:
        scores['scissor'] = 1.0          # Two hands
    elif finger1 != finger2:
        if column_gap == 1 and row_gap == 2:
            scores['scissor'] = 0.0      # Adjacent hurdle

    # 7. Half-scissor (adjacent column reach; ignores all other awkward finger mechanics)
    #    1.0: 2 hands
    #    1.0: non-half-scissor
    #    0.0: half-scissor
    scores['half_scissor'] = 1.0              # Default score
    if hand1 != hand2:
        scores['half_scissor'] = 1.0          # Two hands
    elif finger1 != finger2:
        if column_gap == 1 and row_gap == 1:
            scores['half_scissor'] = 0.0      # Adjacent reach


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
    """Compute Engram scores for all key-pairs."""
    key_pairs = generate_all_key_pairs()
    results = {}
    
    # Initialize results for overall and individual criteria
    results['overall'] = []
    
    for criterion in criteria:
        results[criterion] = []

    print(f"Computing Engram scores for {len(key_pairs)} key-pairs...")

    for i, key_pair in enumerate(key_pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_pairs)} ({i/len(key_pairs)*100:.1f}%)")
        
        # Compute individual Engram criteria scores using the scorer's function
        bigram_scores = score_bigram(key_pair)
        
        # Calculate sum (baseline Engram score)
        engram_score = sum(bigram_scores.values())
        
        # Store overall score
        results['overall'].append({
                'key_pair': key_pair,
                'engram_score': engram_score
        })
    
        # Store individual criterion scores
        for criterion in criteria:
            results[criterion].append({
                'key_pair': key_pair,
                f'engram_{criterion}': bigram_scores[criterion]
            })

    return results

def save_all_score_files(results, output_dir="../tables"):
    """Save overall and individual criterion scores to separate CSV files."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save overall scores
    overall_file = f"{output_dir}/engram_2key_scores.csv"
    overall_results = sorted(results['overall'], key=lambda x: x['key_pair'])
    
    with open(overall_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'engram_score'])
        writer.writeheader()
        writer.writerows(overall_results)
    
    print(f"✅ Saved overall scores to: {overall_file}")
    
    for criterion in criteria:
        criterion_file = f"{output_dir}/engram_2key_scores_{criterion}.csv"
        criterion_results = sorted(results[criterion], key=lambda x: x['key_pair'])
        
        with open(criterion_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key_pair', f'engram_{criterion}'])
            writer.writeheader()
            writer.writerows(criterion_results)
        
        print(f"✅ Saved {criterion} scores to: {criterion_file}")

def validate_output(output_dir="../tables"):
    """
    Validation with comprehensive accuracy checking.
    
    This function performs extensive validation of the generated Engram scores,
    including mathematical verification of scoring criteria and edge case testing.
    """
    import csv
    import random
    from pathlib import Path
    
    overall_file = f"{output_dir}/engram_2key_scores.csv"
    
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
    scores = [float(row['engram_score']) for row in rows]
    min_score, max_score = min(scores), max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.4f} to {max_score:.4f}")
    print(f"   Average score: {avg_score:.4f}")
    # Now checking for 0-ncriteria range instead of 0-1
    print(f"   Valid range (0-ncriteria): {'✅' if 0 <= min_score and max_score <= ncriteria else '❌'}")

    # Test mathematical accuracy on random samples
    print(f"\n🧮 Mathematical Accuracy Check:")
    random_samples = random.sample(rows, min(20, len(rows)))
    accuracy_errors = 0
    
    for row in random_samples:
        key_pair = row['key_pair']
        csv_score = float(row['engram_score'])
        
        # Recalculate score using the same logic
        calculated_scores = score_bigram(key_pair)
        # Use raw sum instead of normalized average
        calculated_sum = sum(calculated_scores.values())
        
        if abs(calculated_sum - csv_score) > 0.0001:
            print(f"   ❌ Accuracy error: {key_pair} - CSV: {csv_score:.4f}, Calc: {calculated_sum:.4f}")
            accuracy_errors += 1
    
    print(f"   Accuracy check: {len(random_samples) - accuracy_errors}/{len(random_samples)} samples correct")
    
    # Validate individual criterion files
    print(f"\n📁 Individual Criterion Files:")    
    for criterion in criteria:
        criterion_file = f"{output_dir}/engram_2key_scores_{criterion}.csv"
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
                test_scores.append(float(row['engram_score']))
        
        all_pass = all(score_check(score) for score in test_scores)
        print(f"   {test_name}: {'✅' if all_pass else '❌'} ({len([s for s in test_scores if score_check(s)])}/{len(test_scores)})")
    
    # Perfect scores count
    perfect_count = len([s for s in scores if s == ncriteria])
    print(f"   Perfect: {perfect_count} pairs ({perfect_count/len(scores)*100:.1f}%)")

    print(f"\n✅ Validation complete!")
    return accuracy_errors == 0

def validate_perfect_scores(output_dir="../tables"):
    """Specifically validate that perfect scores are mathematically correct."""
    import csv
    
    overall_file = f"{output_dir}/engram_2key_scores.csv"
    
    with open(overall_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        perfect_pairs = [row for row in reader if float(row['engram_score']) == ncriteria]
    
    print(f"\n🏆 Perfect Score Verification ({len(perfect_pairs)} pairs):")
    for row in perfect_pairs:
        key_pair = row['key_pair']
        scores = score_bigram(key_pair)
        total_score = sum(scores.values())
        is_perfect = total_score == ncriteria
        print(f"   {key_pair}: {'✅' if is_perfect else '❌'}")
        if not is_perfect:
            print(f"      Individual scores: {scores}")
            print(f"      Sum: {total_score}")
    
    return True

def main():
    """Main entry point."""
    print("Prepare Engram key-pair scores (overall + individual criteria)")
    print("=" * 70)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print()
    
    # Compute scores
    results = compute_key_pair_scores()
    
    # Save results
    output_dir = "../tables"
    save_all_score_files(results, output_dir)
    
    # Validate output
    validate_output(output_dir)
    validate_perfect_scores(output_dir)

    print(f"\n✅ Engram key-pair score generation complete!")
    print(f"   Overall scores: {output_dir}/engram_2key_scores.csv")
    print(f"   Individual criteria: {output_dir}/engram_2key_scores_*.csv")

if __name__ == "__main__":
    main()
