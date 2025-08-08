#!/usr/bin/env python3
"""
Generate complete comfort scores for all possible QWERTY key-pairs.

This script takes the existing comfort scores for 24-key combinations and
generates scores for the remaining key-pairs to create a complete dataset
of all 1024 possible key-pairs from the 32 QWERTY keys.

The script uses different strategies for different types of key-pairs:

For ONE HAND key-pairs (same hand):
1. Key-pairs with one lateral stretch key (TYGHBN['): 
   - If lateral key is 1st: use minimum comfort score where the non-lateral 
     key appears as 2nd key in existing data
   - If lateral key is 2nd: use minimum comfort score where the non-lateral 
     key appears as 1st key in existing data
2. Key-pairs with two lateral stretch keys: use minimum score from all 
   key-pairs (existing + phase 1 generated) containing either lateral key

For BOTH HANDS key-pairs (different hands):
1. Find maximum comfort score for key-pairs containing the 1st key
2. Find maximum comfort score for key-pairs containing the 2nd key  
3. Average these two maximum scores

Key-pairs that cannot be calculated due to insufficient data are skipped.
Generated key-pairs have empty uncertainty values.

Input:
    input/engram/comfort_keypair_scores_24keys.csv - CSV with existing 24-key comfort scores

Output:
    output/keypair_comfort_scores.csv - Complete CSV with all calculable key-pair scores

Usage:
    python prep_keypair_comfort_scores.py
"""

import csv
import os
from pathlib import Path
from typing import Dict, Set, Tuple, List

# QWERTY keyboard layout with (row, finger, hand) mapping
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

# Define the lateral stretch keys that require finger stretching
LATERAL_STRETCH_KEYS = set("TYGHBN['")

def get_key_hand(key: str) -> str:
    """Get the hand (L or R) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key][2]
    return None

def is_same_hand(key1: str, key2: str) -> bool:
    """Check if two keys are on the same hand."""
    hand1 = get_key_hand(key1)
    hand2 = get_key_hand(key2)
    return hand1 == hand2 and hand1 is not None

def load_existing_scores(input_file: str = "input/engram/comfort_keypair_scores_24keys.csv") -> Dict[str, Tuple[float, float]]:
    """Load existing comfort scores from CSV file."""
    scores = {}
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key_pair = row['key_pair']
            comfort_score = float(row['comfort_score'])
            uncertainty = float(row['uncertainty'])
            scores[key_pair] = (comfort_score, uncertainty)
    
    print(f"📖 Loaded {len(scores)} existing key-pair scores from {input_file}")
    return scores

def get_all_qwerty_keys() -> List[str]:
    """Get all standard QWERTY keys for testing."""
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def analyze_existing_scores(scores: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
    """Analyze existing scores to extract statistics needed for generation."""
    stats = {
        'min_scores_first_pos': {},  # Min scores when key is in first position
        'min_scores_second_pos': {}, # Min scores when key is in second position
        'max_scores_any_pos': {},    # Max scores when key appears in any position
        'min_scores_any_pos': {}     # Min scores when key appears in any position
    }
    
    # Get unique keys from existing data
    existing_keys = set()
    for key_pair in scores.keys():
        existing_keys.add(key_pair[0])
        existing_keys.add(key_pair[1])
    
    print(f"📊 Analyzing scores for {len(existing_keys)} existing keys")
    
    # Calculate statistics for each key
    for key in existing_keys:
        first_pos_scores = []
        second_pos_scores = []
        any_pos_scores = []
        
        for key_pair, (score, _) in scores.items():
            if key_pair[0] == key:
                first_pos_scores.append(score)
                any_pos_scores.append(score)
            if key_pair[1] == key:
                second_pos_scores.append(score)
                any_pos_scores.append(score)
        
        if first_pos_scores:
            stats['min_scores_first_pos'][key] = min(first_pos_scores)
        if second_pos_scores:
            stats['min_scores_second_pos'][key] = min(second_pos_scores)
        if any_pos_scores:
            stats['max_scores_any_pos'][key] = max(any_pos_scores)
            stats['min_scores_any_pos'][key] = min(any_pos_scores)
    
    return stats

def update_stats_with_new_scores(original_stats: Dict[str, Dict], 
                                new_scores: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
    """Update statistics to include newly generated scores."""
    updated_stats = {
        'min_scores_first_pos': dict(original_stats['min_scores_first_pos']),
        'min_scores_second_pos': dict(original_stats['min_scores_second_pos']),
        'max_scores_any_pos': dict(original_stats['max_scores_any_pos']),
        'min_scores_any_pos': dict(original_stats['min_scores_any_pos'])
    }
    
    # Get all keys that appear in new scores
    new_keys = set()
    for key_pair in new_scores.keys():
        new_keys.add(key_pair[0])
        new_keys.add(key_pair[1])
    
    # Calculate statistics for each new key
    for key in new_keys:
        first_pos_scores = []
        second_pos_scores = []
        any_pos_scores = []
        
        for key_pair, (score, _) in new_scores.items():
            if key_pair[0] == key:
                first_pos_scores.append(score)
                any_pos_scores.append(score)
            if key_pair[1] == key:
                second_pos_scores.append(score)
                any_pos_scores.append(score)
        
        # Update or add new statistics
        if first_pos_scores:
            existing_min = updated_stats['min_scores_first_pos'].get(key)
            new_min = min(first_pos_scores)
            updated_stats['min_scores_first_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
        
        if second_pos_scores:
            existing_min = updated_stats['min_scores_second_pos'].get(key)
            new_min = min(second_pos_scores)
            updated_stats['min_scores_second_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
        
        if any_pos_scores:
            existing_max = updated_stats['max_scores_any_pos'].get(key)
            existing_min = updated_stats['min_scores_any_pos'].get(key)
            new_max = max(any_pos_scores)
            new_min = min(any_pos_scores)
            
            updated_stats['max_scores_any_pos'][key] = max(existing_max, new_max) if existing_max is not None else new_max
            updated_stats['min_scores_any_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
    
    return updated_stats

def generate_missing_scores(existing_scores: Dict[str, Tuple[float, float]], 
                          stats: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
    """Generate comfort scores for missing key-pairs based on the rules."""
    all_keys = get_all_qwerty_keys()
    missing_scores = {}
    
    print(f"🔄 Generating scores for missing key-pairs in phases...")
    
    # Phase 1: Same-hand pairs with one lateral stretch key
    print("   Phase 1: Same-hand pairs with one lateral stretch key")
    phase1_count = 0
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            
            # Skip if we already have this key-pair
            if key_pair in existing_scores:
                continue
            
            # Check if this is a same-hand pair with exactly one lateral key
            same_hand = is_same_hand(key1, key2)
            key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
            key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
            
            if same_hand and (key1_is_lateral + key2_is_lateral == 1):
                if key1_is_lateral:
                    # Lateral key is 1st, non-lateral is 2nd
                    # Use min score where non-lateral key appears in 2nd position
                    min_score = stats['min_scores_second_pos'].get(key2)
                else:
                    # Lateral key is 2nd, non-lateral is 1st  
                    # Use min score where non-lateral key appears in 1st position
                    min_score = stats['min_scores_first_pos'].get(key1)
                
                if min_score is not None:
                    missing_scores[key_pair] = (min_score, None)
                    phase1_count += 1
    
    print(f"      Generated {phase1_count} same-hand pairs with one lateral key")
    
    # Phase 2: Same-hand pairs with two lateral stretch keys
    print("   Phase 2: Same-hand pairs with two lateral stretch keys")
    phase2_count = 0
    
    # Combine existing scores with phase 1 results for finding minimums
    combined_scores = {**existing_scores, **missing_scores}
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            
            # Skip if we already have this key-pair
            if key_pair in existing_scores or key_pair in missing_scores:
                continue
            
            # Check if this is a same-hand pair with two lateral keys
            same_hand = is_same_hand(key1, key2)
            key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
            key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
            
            if same_hand and key1_is_lateral and key2_is_lateral:
                # Find minimum score among all pairs containing either key
                min_scores = []
                
                for existing_pair, (score, _) in combined_scores.items():
                    if key1 in existing_pair or key2 in existing_pair:
                        min_scores.append(score)
                
                if min_scores:
                    missing_scores[key_pair] = (min(min_scores), None)
                    phase2_count += 1
    
    print(f"      Generated {phase2_count} same-hand pairs with two lateral keys")
    
    # Update stats to include Phase 1 and Phase 2 results for Phase 3
    print("   Updating statistics with Phase 1 & 2 results...")
    updated_stats = update_stats_with_new_scores(stats, missing_scores)
    
    # Phase 3: Different-hand pairs
    print("   Phase 3: Different-hand pairs")
    phase3_count = 0
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            
            # Skip if we already have this key-pair
            if key_pair in existing_scores or key_pair in missing_scores:
                continue
            
            # Check if this is a different-hand pair
            same_hand = is_same_hand(key1, key2)
            
            if not same_hand:
                # Use maximum scores for each key and average them
                max1 = updated_stats['max_scores_any_pos'].get(key1)
                max2 = updated_stats['max_scores_any_pos'].get(key2)
                
                if max1 is not None and max2 is not None:
                    avg_score = (max1 + max2) / 2
                    missing_scores[key_pair] = (avg_score, None)
                    phase3_count += 1
    
    print(f"      Generated {phase3_count} different-hand pairs")
    
    total_generated = phase1_count + phase2_count + phase3_count
    total_possible = len(all_keys) ** 2 - len(existing_scores)
    skipped = total_possible - total_generated
    
    print(f"✅ Generated {total_generated} missing key-pair scores")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} key-pairs (insufficient data to calculate score)")
    
    return missing_scores

def save_complete_scores(existing_scores: Dict[str, Tuple[float, float]], 
                        generated_scores: Dict[str, Tuple[float, float]], 
                        output_file: str = "output/keypair_comfort_scores.csv"):
    """Save complete comfort scores to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Combine existing and generated scores
    all_scores = {**existing_scores, **generated_scores}
    
    # Convert to list and sort by key-pair for consistent ordering
    results = []
    for key_pair, (comfort_score, uncertainty) in all_scores.items():
        results.append({
            'key_pair': key_pair,
            'comfort_score': comfort_score,
            'uncertainty': uncertainty if uncertainty is not None else ''
        })
    
    results.sort(key=lambda x: x['key_pair'])
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'comfort_score', 'uncertainty'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"💾 Saved {len(results)} total key-pair scores to: {output_file}")
    return len(results)

def validate_output(output_file: str = "output/keypair_comfort_scores.csv"):
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
    print(f"   Expected max count: {expected_count}")
    if len(rows) == expected_count:
        print(f"   Status: ✅ Complete (all key-pairs generated)")
    else:
        missing = expected_count - len(rows)
        print(f"   Status: ⚠️  Partial ({missing} key-pairs skipped due to insufficient data)")
    
    # Check score statistics
    scores = [float(row['comfort_score']) for row in rows]
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.3f} to {max_score:.3f}")
    print(f"   Average score: {avg_score:.3f}")
    
    # Count scores by type
    existing_count = 0
    same_hand_one_lateral = 0
    same_hand_two_lateral = 0
    diff_hand_generated = 0
    empty_uncertainty_count = 0
    
    for row in rows:
        key1, key2 = row['key_pair'][0], row['key_pair'][1]
        if row['uncertainty'] == '':
            empty_uncertainty_count += 1
        
        key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
        key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
        lateral_count = key1_is_lateral + key2_is_lateral
        
        if lateral_count == 0:
            existing_count += 1
        elif is_same_hand(key1, key2):
            if lateral_count == 1:
                same_hand_one_lateral += 1
            else:  # lateral_count == 2
                same_hand_two_lateral += 1
        else:  # different hands with lateral key(s)
            diff_hand_generated += 1
    
    print(f"   Existing scores (no lateral keys): {existing_count}")
    print(f"   Same-hand, one lateral key: {same_hand_one_lateral}")
    print(f"   Same-hand, two lateral keys: {same_hand_two_lateral}")
    print(f"   Different-hand with lateral key(s): {diff_hand_generated}")
    print(f"   Empty uncertainties: {empty_uncertainty_count}")
    
    # Show some examples
    print(f"\n📝 Sample key-pairs and scores:")
    for i in range(0, min(15, len(rows)), max(1, len(rows)//15)):
        row = rows[i]
        key1, key2 = row['key_pair'][0], row['key_pair'][1]
        hand_type = "same" if is_same_hand(key1, key2) else "diff"
        
        key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
        key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
        lateral_count = key1_is_lateral + key2_is_lateral
        
        if lateral_count == 0:
            type_desc = "existing"
        elif lateral_count == 1:
            type_desc = "1-lateral"
        else:
            type_desc = "2-lateral"
        
        print(f"   {row['key_pair']}: {float(row['comfort_score']):.3f} ({hand_type} hand, {type_desc})")
    
    return True

def main():
    """Main entry point."""
    print("Generate Complete QWERTY Key-pair Comfort Scores")
    print("=" * 60)
    
    # Load existing scores
    input_file = "input/engram/comfort_keypair_scores_24keys.csv"
    existing_scores = load_existing_scores(input_file)
    
    # Analyze existing scores to get statistics
    stats = analyze_existing_scores(existing_scores)
    
    # Show what keys we're working with
    all_keys = get_all_qwerty_keys()
    existing_keys = set()
    for key_pair in existing_scores.keys():
        existing_keys.add(key_pair[0])
        existing_keys.add(key_pair[1])
    
    missing_keys = set(all_keys) - existing_keys
    
    print(f"📋 Key Analysis:")
    print(f"   Total QWERTY keys: {len(all_keys)}")
    print(f"   Keys in existing data: {len(existing_keys)} ({''.join(sorted(existing_keys))})")
    print(f"   Missing keys: {len(missing_keys)} ({''.join(sorted(missing_keys))})")
    print(f"   Lateral stretch keys: {''.join(sorted(LATERAL_STRETCH_KEYS))}")
    print(f"   Missing = Lateral stretch: {'✅' if missing_keys == LATERAL_STRETCH_KEYS else '❌'}")
    print()
    
    # Generate missing scores
    generated_scores = generate_missing_scores(existing_scores, stats)
    
    # Save complete dataset
    output_file = "output/keypair_comfort_scores.csv"
    total_count = save_complete_scores(existing_scores, generated_scores, output_file)
    
    # Validate output
    validate_output(output_file)
    
    print(f"\n✅ Comfort score generation complete!")
    print(f"   Input: {len(existing_scores)} existing key-pairs")
    print(f"   Generated: {len(generated_scores)} new key-pairs")
    print(f"     - Phase 1 (same-hand, one lateral): included in total")
    print(f"     - Phase 2 (same-hand, two lateral): included in total") 
    print(f"     - Phase 3 (different-hand): included in total")
    print(f"   Output: {total_count} total key-pairs in {output_file}")

if __name__ == "__main__":
    main()