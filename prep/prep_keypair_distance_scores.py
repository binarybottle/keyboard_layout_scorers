#!/usr/bin/env python3
"""
Generate precomputed distance scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes theoretical distance scores for every possible 
combination of QWERTY keys using only keyboard interkey distances.
This version provides truly layout-agnostic distance scores based only on:
1. Compute theoretical distances for ALL 1024 key-pairs 
2. Fingers start from home except when using the same finger
4. No text processing or frequency weighting

Usage:
    python prep_keypair_distance_scores.py --output ../tables/keypair_distance_scores.csv
    
Output:
    ../tables/keypair_distance_scores.csv - CSV with columns: key_pair, distance_score
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from math import sqrt

# Physical keyboard layout definitions (unchanged from original)
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

FINGER_MAP = {
    'q': 4, 'w': 3, 'e': 2, 'r': 1, 't': 1,
    'a': 4, 's': 3, 'd': 2, 'f': 1, 'g': 1,
    'z': 4, 'x': 3, 'c': 2, 'v': 1, 'b': 1,
    'y': 1, 'u': 1, 'i': 2, 'o': 3, 'p': 4,
    'h': 1, 'j': 1, 'k': 2, 'l': 3, ';': 4, 
    'n': 1, 'm': 1, ',': 2, '.': 3, '/': 4,
    '[': 4, "'": 4
}

COLUMN_MAP = {
    'q': 1, 'w': 2, 'e': 3, 'r': 4, 't': 5, 
    'a': 1, 's': 2, 'd': 3, 'f': 4, 'g': 5, 
    'z': 1, 'x': 2, 'c': 3, 'v': 4, 'b': 5,
    'y': 6, 'u': 7, 'i': 8, 'o': 9, 'p': 10, 
    'h': 6, 'j': 7, 'k': 8, 'l': 9, ';': 10, 
    'n': 6, 'm': 7, ',': 8, '.': 9, '/': 10,
    '[': 11, "'": 11
}

# Home row positions for each finger (where fingers start)
HOME_ROW_POSITIONS = {
    'L4': 'a',  # Left pinky
    'L3': 's',  # Left ring
    'L2': 'd',  # Left middle
    'L1': 'f',  # Left index
    'R1': 'j',  # Right index
    'R2': 'k',  # Right middle
    'R3': 'l',  # Right ring
    'R4': ';'   # Right pinky
}

def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions in mm."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return sqrt(dx * dx + dy * dy)

def get_finger_id(char: str) -> Optional[str]:
    """Get unique finger identifier for a character (combines hand and finger number)."""
    char_lower = char.lower()
    if char_lower not in FINGER_MAP or char_lower not in COLUMN_MAP:
        return None
    
    hand = 'L' if COLUMN_MAP[char_lower] < 6 else 'R'
    finger_num = FINGER_MAP[char_lower]
    return f"{hand}{finger_num}"

def get_physical_position(qwerty_key: str) -> Optional[Tuple[float, float]]:
    """Get the physical position of a QWERTY key."""
    return STAGGERED_POSITION_MAP.get(qwerty_key.lower())

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for analysis."""
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def generate_all_key_pairs():
    """Generate all possible QWERTY key-pair combinations."""
    keys = get_all_qwerty_keys()
    key_pairs = []
    
    for key1 in keys:
        for key2 in keys:
            key_pairs.append(key1 + key2)
    
    return key_pairs

class FingerTracker:
    """Track finger positions with proper reset logic."""
    
    def __init__(self):
        self.finger_positions = {}
        self.reset_all_fingers()
    
    def reset_all_fingers(self):
        """Reset all fingers to their home row positions."""
        self.finger_positions = {}
        for finger_id, home_key in HOME_ROW_POSITIONS.items():
            self.finger_positions[finger_id] = home_key.upper()
    
    def calculate_distance_and_move_finger(self, target_key: str) -> float:
        """
        Calculate distance for finger to move to target key and update position.
        Also resets all other fingers to home positions.
        
        Returns:
            Distance traveled by the finger responsible for target_key (0.0 if no movement needed)
        """
        target_key = target_key.upper()
        
        # Get the finger responsible for this key
        finger_id = get_finger_id(target_key.lower())
        if finger_id is None:
            return 0.0
        
        # Get finger's current position
        current_key = self.finger_positions.get(finger_id, HOME_ROW_POSITIONS[finger_id].upper())
        
        # If finger is already at target key, no movement needed
        if current_key == target_key:
            # Still need to reset other fingers to home positions
            for fid, home_key in HOME_ROW_POSITIONS.items():
                if fid != finger_id:
                    self.finger_positions[fid] = home_key.upper()
            return 0.0
        
        # Calculate distance only if finger needs to move
        current_pos = get_physical_position(current_key)
        target_pos = get_physical_position(target_key)
        
        if current_pos is None or target_pos is None:
            return 0.0
        
        distance = calculate_euclidean_distance(current_pos, target_pos)
        
        # Update this finger's position
        self.finger_positions[finger_id] = target_key
        
        # Reset all OTHER fingers to home positions
        for fid, home_key in HOME_ROW_POSITIONS.items():
            if fid != finger_id:
                self.finger_positions[fid] = home_key.upper()
        
        return distance

def compute_theoretical_keypair_distance(key1: str, key2: str) -> float:
    """
    Compute theoretical distance for a key-pair assuming fingers start from home.
    This will correctly return 0.0 for home row pairs where no movement is needed.
    
    Args:
        key1, key2: The two keys in the pair
        
    Returns:
        Total distance for typing this key-pair
    """
    finger_tracker = FingerTracker()
    
    # Calculate distance for first keystroke
    distance1 = finger_tracker.calculate_distance_and_move_finger(key1)
    
    # Calculate distance for second keystroke  
    distance2 = finger_tracker.calculate_distance_and_move_finger(key2)
    
    return distance1 + distance2

def compute_all_theoretical_distances() -> Dict[str, float]:
    """
    Compute PURE THEORETICAL distances for ALL possible key-pairs.
    Assumes all fingers start from home row positions.
    
    This is FREQUENCY-INDEPENDENT - treats all key-pairs equally.
    
    Returns:
        Dictionary mapping all key-pairs to theoretical distance scores
    """
    print("\n🔵 Computing PURE THEORETICAL distances for all key-pairs")
    print("   (No text processing, no frequency bias)")
    
    all_key_pairs = generate_all_key_pairs()
    theoretical_scores = {}
    
    print(f"  Computing distances for {len(all_key_pairs)} key-pairs...")
    
    for i, key_pair in enumerate(all_key_pairs):
        if i % 100 == 0:
            print(f"    Progress: {i}/{len(all_key_pairs)} ({i/len(all_key_pairs)*100:.1f}%)")
        
        char1, char2 = key_pair[0], key_pair[1]
        distance = compute_theoretical_keypair_distance(char1, char2)
        theoretical_scores[key_pair] = distance
    
    print(f"  ✅ Computed theoretical distances for all {len(theoretical_scores)} key-pairs")
    
    # Show some statistics
    distances = list(theoretical_scores.values())
    zero_count = sum(1 for d in distances if d == 0.0)
    avg_distance = sum(distances) / len(distances)
    max_distance = max(distances)
    min_distance = min(distances)
    
    print(f"  📊 Statistics:")
    print(f"    Zero distances: {zero_count}")
    print(f"    Min distance: {min_distance:.2f}mm")
    print(f"    Average distance: {avg_distance:.2f}mm")
    print(f"    Max distance: {max_distance:.2f}mm")
    
    return theoretical_scores

def compute_all_key_pair_scores():
    """Compute distance scores for all key-pairs using PURE THEORETICAL approach."""
    print("Computing distance scores using PURE THEORETICAL approach...")
    print("🎯 This version is FREQUENCY-INDEPENDENT")
    print("   - No text processing")
    print("   - No frequency weighting") 
    print("   - Pure biomechanical evaluation")
    
    # Compute theoretical distances for all key-pairs
    theoretical_scores = compute_all_theoretical_distances()
    
    # Convert to results format - use theoretical distances only
    results = []
    for key_pair, distance in sorted(theoretical_scores.items()):
        results.append({
            'key_pair': key_pair,
            'distance_score': distance
        })
    
    print(f"\n✅ Total key-pairs computed: {len(results)}")
    print("🎯 All scores are frequency-independent and layout-agnostic")
    
    return results

def save_key_pair_scores(results, output_file="../tables/keypair_distance_scores.csv"):
    """Save key-pair scores to CSV file with distance_score column."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'distance_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def verify_distance_calculation_model():
    """Verify that distance calculation correctly handles same-finger vs different-finger cases."""
    
    print(f"\n🔍 DISTANCE CALCULATION MODEL VERIFICATION")
    print("=" * 60)
    print("Expected behavior:")
    print("  Different fingers: home→key1 + home→key2")
    print("  Same finger: home→key1 + key1→key2")
    
    # Test cases
    test_cases = [
        # (key1, key2, expected_behavior, description)
        ('E', 'R', 'different', 'E(middle) + R(index) = (D→E) + (F→R)'),
        ('T', 'G', 'same', 'T+G(both index) = (F→T) + (T→G)'),
        ('A', 'S', 'different', 'A(pinky) + S(ring) = (A→A) + (S→S) = 0 + 0'),
        ('F', 'G', 'same', 'F+G(both index) = (F→F) + (F→G) = 0 + (F→G)'),
        ('Q', 'W', 'different', 'Q(pinky) + W(ring) = (A→Q) + (S→W)'),
        ('R', 'T', 'same', 'R+T(both index) = (F→R) + (R→T)'),
    ]
    
    print(f"\n📊 Test Results:")
    print(f"{'Pair':<4} | {'Finger':<10} | {'Calculated':<10} | {'Components':<25} | {'Status'}")
    print("-" * 75)
    
    for key1, key2, expected_type, description in test_cases:
        # Get finger assignments
        finger1 = get_finger_id(key1.lower())
        finger2 = get_finger_id(key2.lower())
        same_finger = (finger1 == finger2)
        
        # Calculate distance using our function
        total_distance = compute_theoretical_keypair_distance(key1, key2)
        
        # Manually verify the calculation
        if same_finger:
            # Same finger: home→key1 + key1→key2
            home_key = HOME_ROW_POSITIONS.get(finger1)
            home_pos = get_physical_position(home_key)
            key1_pos = get_physical_position(key1.lower())
            key2_pos = get_physical_position(key2.lower())
            
            expected_dist1 = calculate_euclidean_distance(home_pos, key1_pos)
            expected_dist2 = calculate_euclidean_distance(key1_pos, key2_pos)
            expected_total = expected_dist1 + expected_dist2
            components = f"({home_key.upper()}→{key1}) + ({key1}→{key2})"
        else:
            # Different fingers: home→key1 + home→key2  
            home_key1 = HOME_ROW_POSITIONS.get(finger1)
            home_key2 = HOME_ROW_POSITIONS.get(finger2)
            home_pos1 = get_physical_position(home_key1)
            home_pos2 = get_physical_position(home_key2)
            key1_pos = get_physical_position(key1.lower())
            key2_pos = get_physical_position(key2.lower())
            
            expected_dist1 = calculate_euclidean_distance(home_pos1, key1_pos)
            expected_dist2 = calculate_euclidean_distance(home_pos2, key2_pos)
            expected_total = expected_dist1 + expected_dist2
            components = f"({home_key1.upper()}→{key1}) + ({home_key2.upper()}→{key2})"
        
        # Check if calculation matches expectation
        matches = abs(total_distance - expected_total) < 0.001
        finger_type = "same" if same_finger else "diff"
        status = "✅" if matches and (finger_type == expected_type[:4]) else "❌"
        
        print(f"{key1+key2:<4} | {finger_type:<10} | {total_distance:<10.2f} | {components:<25} | {status}")
    
    print(f"\n🎯 Model verification complete!")
    return True

def validate_frequency_independence(output_file="../tables/keypair_distance_scores.csv"):
    """Validate that the scores are truly frequency-independent."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 FREQUENCY-INDEPENDENCE VALIDATION")
    print("=" * 50)
    
    # Check that we have scores for ALL possible key-pairs
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    actual_count = len(rows)
    
    print(f"Expected key-pairs: {expected_count}")
    print(f"Actual key-pairs: {actual_count}")
    print(f"Complete coverage: {'✅' if actual_count == expected_count else '❌'}")
    
    # Verify scores are based on theoretical calculations
    distance_scores = [float(row['distance_score']) for row in rows]
    print(f"\nDistance score statistics:")
    print(f"  Min: {min(distance_scores):.2f}mm")
    print(f"  Max: {max(distance_scores):.2f}mm")
    print(f"  Average: {sum(distance_scores)/len(distance_scores):.2f}mm")
    
    # Check some expected patterns
    print(f"\n🔍 Validation checks:")
    
    # Same-key home row pairs should be 0
    home_row_keys = ['A', 'S', 'D', 'F', 'J', 'K', 'L', ';']
    home_same_key_pairs = [row for row in rows if row['key_pair'][0] == row['key_pair'][1] and row['key_pair'][0] in home_row_keys]
    zero_home_pairs = [row for row in home_same_key_pairs if float(row['distance_score']) == 0.0]
    print(f"  Home row same-key pairs with 0 distance: {len(zero_home_pairs)}/{len(home_same_key_pairs)} {'✅' if len(zero_home_pairs) == len(home_same_key_pairs) else '❌'}")
    
    # Adjacent keys should have shorter distances than distant keys
    qw_distance = next((float(row['distance_score']) for row in rows if row['key_pair'] == 'QW'), None)
    qp_distance = next((float(row['distance_score']) for row in rows if row['key_pair'] == 'QP'), None)
    
    if qw_distance and qp_distance:
        print(f"  QW (adjacent): {qw_distance:.2f}mm")
        print(f"  QP (distant): {qp_distance:.2f}mm")
        print(f"  Distance ordering correct: {'✅' if qp_distance > qw_distance else '❌'}")
    
    # Check that common English bigrams don't have suspiciously low scores
    common_english = ['TH', 'HE', 'IN', 'ER', 'AN']
    random_pairs = ['XZ', 'QJ', 'WK', 'YB', 'PV']
    
    common_avg = sum(float(row['distance_score']) for row in rows if row['key_pair'] in common_english) / len(common_english)
    random_avg = sum(float(row['distance_score']) for row in rows if row['key_pair'] in random_pairs) / len(random_pairs)
    
    print(f"  Common English avg: {common_avg:.2f}mm")
    print(f"  Random pairs avg: {random_avg:.2f}mm")
    print(f"  No English bias: {'✅' if abs(common_avg - random_avg) < (random_avg * 0.3) else '⚠️'}")
    
    print(f"\n✅ Frequency-independence validation complete!")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate FREQUENCY-INDEPENDENT distance scores for QWERTY key-pairs',
        epilog="""
This version computes PURE THEORETICAL distance scores with:
- No text processing
- No frequency weighting
- Complete layout-agnostic evaluation

Perfect for dual framework analysis where you want to separate
design quality from language-specific optimization.
        """
    )
    parser.add_argument('--output', default='../tables/keypair_distance_scores.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("Generate FREQUENCY-INDEPENDENT distance-based key-pair scores")
    print("=" * 70)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print("🎯 Using PURE THEORETICAL approach (no frequency bias)")
    
    # Compute scores using pure theoretical approach
    results = compute_all_key_pair_scores()
    
    # Save results
    save_key_pair_scores(results, args.output)
    
    # Verify distance calculation model
    verify_distance_calculation_model()
    
    # Validate frequency independence
    validate_frequency_independence(args.output)
    
    print(f"\n✅ Frequency-independent distance generation complete: {args.output}")
    print("🎯 These scores are ready for dual framework analysis!")
    
    return 0

if __name__ == "__main__":
    exit(main())