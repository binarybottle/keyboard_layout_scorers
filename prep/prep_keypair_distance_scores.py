#!/usr/bin/env python3
"""
Generate precomputed distance scores for all possible QWERTY key-pairs with breakdown.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes theoretical distance scores for every possible 
combination of QWERTY keys with four components:
1. distance_setup: Distance to position finger(s) for first key
2. distance_interval: Distance to move from first key to second key
3. distance_return: Distance to return finger(s) to home positions  
4. distance_total: Sum of all three components

Usage:
    python prep_keypair_distance_scores.py --output ../tables/keypair_distance_scores.csv
    
Output:
    ../tables/keypair_distance_scores.csv - CSV with distance breakdown
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from math import sqrt

class DistanceBreakdown(NamedTuple):
    """Container for distance breakdown."""
    setup: float
    interval: float
    return_distance: float
    total: float

# Physical keyboard layout definitions
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
    'q': 1, 'w': 2, 'e': 3, 'r': 4, 't': 4,  # 1=pinky, 4=index
    'a': 1, 's': 2, 'd': 3, 'f': 4, 'g': 4,
    'z': 1, 'x': 2, 'c': 3, 'v': 4, 'b': 4,
    'y': 4, 'u': 4, 'i': 3, 'o': 2, 'p': 1,
    'h': 4, 'j': 4, 'k': 3, 'l': 2, ';': 1, 
    'n': 4, 'm': 4, ',': 3, '.': 2, '/': 1,
    '[': 1, "'": 1
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

# Home row positions for each finger (1=pinky, 4=index):
HOME_ROW_POSITIONS = {
    'L1': 'a',  # Left pinky
    'L2': 's',  # Left ring
    'L3': 'd',  # Left middle
    'L4': 'f',  # Left index
    'R4': 'j',  # Right index
    'R3': 'k',  # Right middle
    'R2': 'l',  # Right ring
    'R1': ';'   # Right pinky
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

def get_home_key_for_finger(finger_id: str) -> str:
    """Get the home row key for a given finger."""
    return HOME_ROW_POSITIONS.get(finger_id, '').upper()

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

def compute_detailed_keypair_distance(key1: str, key2: str) -> DistanceBreakdown:
    """
    Compute distance breakdown for a key-pair.
    
    Returns breakdown with:
    - setup: Distance to position finger(s) for first key
    - interval: Distance to move from first key to second key
    - return_distance: Distance to return finger(s) to home
    - total: Sum of all components
    """
    key1 = key1.upper()
    key2 = key2.upper()
    
    # Get finger assignments
    finger1_id = get_finger_id(key1.lower())
    finger2_id = get_finger_id(key2.lower())
    
    if finger1_id is None or finger2_id is None:
        return DistanceBreakdown(0.0, 0.0, 0.0, 0.0)
    
    # Get physical positions
    key1_pos = get_physical_position(key1.lower())
    key2_pos = get_physical_position(key2.lower())
    home1_key = get_home_key_for_finger(finger1_id)
    home2_key = get_home_key_for_finger(finger2_id)
    home1_pos = get_physical_position(home1_key.lower())
    home2_pos = get_physical_position(home2_key.lower())
    
    if None in [key1_pos, key2_pos, home1_pos, home2_pos]:
        return DistanceBreakdown(0.0, 0.0, 0.0, 0.0)
    
    same_finger = (finger1_id == finger2_id)
    
    if same_finger:
        # Same finger: home→key1→key2→home
        setup_dist = calculate_euclidean_distance(home1_pos, key1_pos)
        interval_dist = calculate_euclidean_distance(key1_pos, key2_pos)
        return_dist = calculate_euclidean_distance(key2_pos, home1_pos)
    else:
        # Different fingers: 
        # Setup: finger1 home→key1, finger2 stays home
        # Interval: finger1 stays at key1, finger2 home→key2  
        # Return: finger1 key1→home, finger2 key2→home
        setup_dist = calculate_euclidean_distance(home1_pos, key1_pos)
        interval_dist = calculate_euclidean_distance(home2_pos, key2_pos)
        return_dist = (calculate_euclidean_distance(key1_pos, home1_pos) + 
                      calculate_euclidean_distance(key2_pos, home2_pos))
    
    total_dist = setup_dist + interval_dist + return_dist
    
    return DistanceBreakdown(
        setup=setup_dist,
        interval=interval_dist, 
        return_distance=return_dist,
        total=total_dist
    )

def compute_all_detailed_distances() -> Dict[str, DistanceBreakdown]:
    """
    Compute distance breakdowns for ALL possible key-pairs.
    
    Returns:
        Dictionary mapping all key-pairs to distance breakdowns
    """
    print("\n🔵 Computing distance breakdowns for all key-pairs")
    print("   Components: setup, interval, return, total")
    
    all_key_pairs = generate_all_key_pairs()
    detailed_scores = {}
    
    print(f"  Computing distances for {len(all_key_pairs)} key-pairs...")
    
    for i, key_pair in enumerate(all_key_pairs):
        if i % 100 == 0:
            print(f"    Progress: {i}/{len(all_key_pairs)} ({i/len(all_key_pairs)*100:.1f}%)")
        
        char1, char2 = key_pair[0], key_pair[1]
        breakdown = compute_detailed_keypair_distance(char1, char2)
        detailed_scores[key_pair] = breakdown
    
    print(f"  ✅ Computed breakdowns for all {len(detailed_scores)} key-pairs")
    
    # Show some statistics
    totals = [breakdown.total for breakdown in detailed_scores.values()]
    setups = [breakdown.setup for breakdown in detailed_scores.values()]
    intervals = [breakdown.interval for breakdown in detailed_scores.values()]
    returns = [breakdown.return_distance for breakdown in detailed_scores.values()]
    
    print(f"  📊 Statistics:")
    print(f"    Setup distances - avg: {sum(setups)/len(setups):.2f}mm, max: {max(setups):.2f}mm")
    print(f"    Interval distances - avg: {sum(intervals)/len(intervals):.2f}mm, max: {max(intervals):.2f}mm")
    print(f"    Return distances - avg: {sum(returns)/len(returns):.2f}mm, max: {max(returns):.2f}mm")
    print(f"    Total distances - avg: {sum(totals)/len(totals):.2f}mm, max: {max(totals):.2f}mm")
    
    return detailed_scores

def compute_all_key_pair_scores():
    """Compute distance scores for all key-pairs."""
    print("Computing distance scores for all key-pairs...")
    print("🎯 Four-component breakdown:")
    print("   - distance_setup: Position finger(s) for first key")
    print("   - distance_interval: Move from first to second key")
    print("   - distance_return: Return finger(s) to home")
    print("   - distance_total: Sum of all components")
    
    # Compute distances for all key-pairs
    detailed_scores = compute_all_detailed_distances()
    
    # Convert to results format
    results = []
    for key_pair, breakdown in sorted(detailed_scores.items()):
        results.append({
            'key_pair': key_pair,
            'distance_setup': breakdown.setup,
            'distance_interval': breakdown.interval,
            'distance_return': breakdown.return_distance,
            'distance_total': breakdown.total
        })
    
    print(f"\n✅ Total key-pairs computed: {len(results)}")
    print("🎯 All scores provide motion breakdown")
    
    return results

def save_key_pair_scores(results, output_file="../tables/keypair_distance_scores.csv"):
    """Save key-pair scores to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    fieldnames = ['key_pair', 'distance_setup', 'distance_interval', 'distance_return', 'distance_total']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def verify_detailed_calculation_model():
    """Verify that distance calculation works correctly."""
    
    print(f"\n🔍 DISTANCE CALCULATION MODEL VERIFICATION")
    print("=" * 70)
    
    # Test cases with expected behavior
    test_cases = [
        ('F', 'F', 'Home row same key - all zeros except return may be 0'),
        ('F', 'G', 'Same finger (index) - sequential motion'),
        ('F', 'J', 'Different fingers (both index but different hands)'),
        ('A', 'S', 'Different fingers (pinky to ring) - home row'),
        ('Q', 'P', 'Different fingers - maximum distance'),
        ('T', 'G', 'Same finger - off home row'),
    ]
    
    print(f"\n📊 Test Results:")
    print(f"{'Pair':<4} | {'Setup':<8} | {'Interval':<8} | {'Return':<8} | {'Total':<8} | {'Description'}")
    print("-" * 75)
    
    for key1, key2, description in test_cases:
        breakdown = compute_detailed_keypair_distance(key1, key2)
        
        print(f"{key1+key2:<4} | {breakdown.setup:<8.2f} | {breakdown.interval:<8.2f} | "
              f"{breakdown.return_distance:<8.2f} | {breakdown.total:<8.2f} | {description}")
    
    # Verify that total equals sum of components
    print(f"\n🔍 Component sum verification:")
    for key1, key2, description in test_cases[:3]:  # Test a few cases
        breakdown = compute_detailed_keypair_distance(key1, key2)
        calculated_total = breakdown.setup + breakdown.interval + breakdown.return_distance
        matches = abs(breakdown.total - calculated_total) < 0.001
        status = "✅" if matches else "❌"
        print(f"  {key1+key2}: {breakdown.total:.2f} = {breakdown.setup:.2f} + {breakdown.interval:.2f} + {breakdown.return_distance:.2f} {status}")
    
    print(f"\n🎯 Model verification complete!")
    return True

def validate_detailed_output(output_file="../tables/keypair_distance_scores.csv"):
    """Validate the output file."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 OUTPUT VALIDATION")
    print("=" * 50)
    
    # Check that we have all expected columns
    expected_columns = ['key_pair', 'distance_setup', 'distance_interval', 'distance_return', 'distance_total']
    actual_columns = reader.fieldnames if reader.fieldnames else []
    missing_columns = set(expected_columns) - set(actual_columns)
    
    print(f"Expected columns: {expected_columns}")
    print(f"Missing columns: {list(missing_columns) if missing_columns else 'None ✅'}")
    
    # Check total count
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    actual_count = len(rows)
    
    print(f"Expected key-pairs: {expected_count}")
    print(f"Actual key-pairs: {actual_count}")
    print(f"Complete coverage: {'✅' if actual_count == expected_count else '❌'}")
    
    # Validate numeric values and component relationships
    valid_rows = 0
    for row in rows:
        try:
            setup = float(row['distance_setup'])
            interval = float(row['distance_interval'])
            return_dist = float(row['distance_return'])
            total = float(row['distance_total'])
            
            # Check that total equals sum of components (within floating point precision)
            calculated_total = setup + interval + return_dist
            if abs(total - calculated_total) < 0.001:
                valid_rows += 1
        except (ValueError, KeyError):
            continue
    
    print(f"Valid numeric rows: {valid_rows}/{actual_count}")
    print(f"Component sum consistency: {'✅' if valid_rows == actual_count else '❌'}")
    
    # Show some statistics
    if valid_rows > 0:
        setup_values = [float(row['distance_setup']) for row in rows]
        interval_values = [float(row['distance_interval']) for row in rows]
        return_values = [float(row['distance_return']) for row in rows]
        total_values = [float(row['distance_total']) for row in rows]
        
        print(f"\nDistance component statistics:")
        print(f"  Setup - min: {min(setup_values):.2f}, max: {max(setup_values):.2f}, avg: {sum(setup_values)/len(setup_values):.2f}")
        print(f"  Interval - min: {min(interval_values):.2f}, max: {max(interval_values):.2f}, avg: {sum(interval_values)/len(interval_values):.2f}")
        print(f"  Return - min: {min(return_values):.2f}, max: {max(return_values):.2f}, avg: {sum(return_values)/len(return_values):.2f}")
        print(f"  Total - min: {min(total_values):.2f}, max: {max(total_values):.2f}, avg: {sum(total_values)/len(total_values):.2f}")

    print(f"\n✅ Output validation complete!")

    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate distance scores for QWERTY key-pairs',
        epilog="""
This version computes distance breakdowns with four components:
- distance_setup: Position finger(s) for first key
- distance_interval: Move from first to second key  
- distance_return: Return finger(s) to home positions
- distance_total: Sum of all components

Perfect for analyzing different aspects of typing motion.
        """
    )
    parser.add_argument('--output', default='../tables/keypair_distance_scores.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("Generate distance-based key-pair scores")
    print("=" * 70)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print("🎯 Four-component breakdown")
    
    # Compute scores
    results = compute_all_key_pair_scores()
    
    # Save results
    save_key_pair_scores(results, args.output)
    
    # Verify calculation model
    verify_detailed_calculation_model()
    
    # Validate output
    validate_detailed_output(args.output)
    
    print(f"\n✅ Distance generation complete: {args.output}")
    print("🎯 Four-component breakdown ready for analysis!")
    
    return 0

if __name__ == "__main__":
    exit(main())