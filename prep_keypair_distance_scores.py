#!/usr/bin/env python3
"""
Generate precomputed distance scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes distance scores for every possible combination of QWERTY keys
using a 2-stage approach:

1. Compute theoretical distances for ALL 1024 key-pairs (fingers start from home)
2. Process text files to find actual key-pairs and replace with averaged real-world scores

For each key-pair instance in text:
1. Spaces reset all fingers to home positions  
2. When a finger moves, all other fingers reset to home positions
3. Distance is calculated as sum of both keystrokes in the pair

Usage:
    python prep_keypair_distance_scores.py --text-files file1.txt,file2.txt
    python prep_keypair_distance_scores.py --text-files corpus.txt

Output:
    output/keypair_distance_scores.csv - CSV with columns: key_pair, distance_score
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from math import sqrt
from collections import defaultdict

# Physical keyboard layout definitions (from distance_scorer.py)
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
    if char not in FINGER_MAP or char not in COLUMN_MAP:
        return None
    
    hand = 'L' if COLUMN_MAP[char] < 6 else 'R'
    finger_num = FINGER_MAP[char]
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
    Stage 1: Compute theoretical distances for ALL possible key-pairs.
    Assumes all fingers start from home row positions.
    
    Returns:
        Dictionary mapping all key-pairs to theoretical distance scores
    """
    print("\n🔵 Stage 1: Computing theoretical distances for all key-pairs")
    
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
    
    print(f"  Statistics: {zero_count} zero distances, avg={avg_distance:.2f}mm, max={max_distance:.2f}mm")
    
    return theoretical_scores

def extract_bigrams_and_distances_from_text(text: str) -> Dict[str, List[float]]:
    """
    Extract bigrams from text and compute their distances with proper finger tracking.
    
    For each word, simulates typing character by character and records the distance
    for each bigram as it occurs in context.
    
    Example: "EDF"
    - Type E: distance = home→E (finger 2: D→E), prev_distance = home→E  
    - Type D: distance = E→D (finger 2: E→D), ED bigram = (home→E) + (E→D)
    - Type F: distance = home→F = 0 (finger 1: F→F), DF bigram = (E→D) + 0
    
    The DF bigram gets distance (E→D) + 0 because finger 2 is at E when we start 
    typing the DF bigram in this context.
    
    Args:
        text: Input text (should be uppercase)
        
    Returns:
        Dictionary mapping bigrams to lists of distance values
    """
    bigram_distances = defaultdict(list)
    
    # Split text by spaces to handle space resets
    words = text.split()
    
    for word in words:
        # Filter to valid QWERTY characters
        valid_chars = []
        for char in word:
            if char in STAGGERED_POSITION_MAP:
                valid_chars.append(char)
        
        if len(valid_chars) < 2:
            continue  # Need at least 2 characters for bigrams
        
        # Simulate typing this word character by character
        finger_tracker = FingerTracker()  # Start with all fingers at home
        prev_distance = 0.0
        
        for i, char in enumerate(valid_chars):
            # Calculate distance to type this character
            distance = finger_tracker.calculate_distance_and_move_finger(char)
            
            # If this forms a bigram, record the bigram distance
            if i > 0:
                prev_char = valid_chars[i - 1]
                bigram = prev_char + char
                
                # Bigram distance is the sum of distances for both characters
                bigram_distance = prev_distance + distance
                bigram_distances[bigram].append(bigram_distance)
            
            # Remember this distance for the next bigram
            prev_distance = distance
    
    return bigram_distances

def compute_text_based_distances(text_files: List[str]) -> Dict[str, float]:
    """
    Stage 2: Process text files to find actual key-pairs and compute averaged distances.
    
    Args:
        text_files: List of text file paths
        
    Returns:
        Dictionary mapping key-pairs found in text to averaged distance scores
    """
    print("\n🟡 Stage 2: Processing text files for real-world key-pair usage")
    
    all_bigram_distances = defaultdict(list)
    
    for text_file in text_files:
        print(f"  Processing: {text_file}")
        
        try:
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().upper()  # Convert entire text to uppercase
        except FileNotFoundError:
            print(f"    Warning: File not found: {text_file}")
            continue
        except Exception as e:
            print(f"    Warning: Error reading {text_file}: {e}")
            continue
        
        print(f"    Text length: {len(text)} characters")
        
        # Extract bigrams and their distances with proper finger tracking
        bigram_distances = extract_bigrams_and_distances_from_text(text)
        
        # Combine results from this file
        for bigram, distances in bigram_distances.items():
            all_bigram_distances[bigram].extend(distances)
        
        total_bigram_instances = sum(len(distances) for distances in bigram_distances.values())
        print(f"    Found {total_bigram_instances} total bigram instances")
        print(f"    Unique bigrams: {len(bigram_distances)}")
    
    # Calculate averages for pairs found in text
    text_based_scores = {}
    for key_pair, distances in all_bigram_distances.items():
        text_based_scores[key_pair] = sum(distances) / len(distances)
    
    print(f"  ✅ Computed averaged distances for {len(text_based_scores)} key-pairs from text")
    
    # Show some examples
    if text_based_scores:
        print(f"  Sample text-based scores:")
        sample_pairs = list(text_based_scores.items())[:5]
        for pair, distance in sample_pairs:
            instances = len(all_bigram_distances[pair])
            print(f"    {pair}: {distance:.2f}mm (avg of {instances} instances)")
    
    return text_based_scores

def combine_theoretical_and_text_scores(theoretical_scores: Dict[str, float], 
                                       text_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Combine theoretical and text-based scores.
    Text-based scores override theoretical scores when available.
    
    Args:
        theoretical_scores: All key-pairs with theoretical distances
        text_scores: Key-pairs found in text with averaged distances
        
    Returns:
        Final combined scores for all key-pairs
    """
    print("\n🟢 Stage 3: Combining theoretical and text-based scores")
    
    final_scores = theoretical_scores.copy()
    
    # Override with text-based scores where available
    overridden_count = 0
    for key_pair, text_distance in text_scores.items():
        if key_pair in final_scores:
            final_scores[key_pair] = text_distance
            overridden_count += 1
    
    print(f"  ✅ Overrode {overridden_count} theoretical scores with text-based averages")
    print(f"  Final scores: {len(final_scores)} total key-pairs")
    
    # Statistics
    text_pairs = len(text_scores)
    theoretical_pairs = len(final_scores) - text_pairs
    print(f"    - {text_pairs} from text analysis")
    print(f"    - {theoretical_pairs} from theoretical calculation")
    
    return final_scores

def compute_all_key_pair_scores(text_files: List[str]):
    """Compute distance scores for all key-pairs using theoretical + text approach."""
    print("Computing distance scores using theoretical + text approach...")
    
    # Stage 1: Compute theoretical distances for all key-pairs
    theoretical_scores = compute_all_theoretical_distances()
    
    # Stage 2: Process text files for real-world usage patterns
    text_scores = compute_text_based_distances(text_files)
    
    # Stage 3: Combine scores (text overrides theoretical)
    final_scores = combine_theoretical_and_text_scores(theoretical_scores, text_scores)
    
    # Convert to results format
    results = []
    for key_pair, score in sorted(final_scores.items()):
        results.append({
            'key_pair': key_pair,
            'distance_score': score
        })
    
    print(f"\n✅ Total key-pairs computed: {len(results)}")
    
    return results

def save_key_pair_scores(results, output_file="output/keypair_distance_scores.csv"):
    """Save key-pair scores to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'distance_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def validate_output(output_file="output/keypair_distance_scores.csv"):
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
    
    # Check score range
    scores = [float(row['distance_score']) for row in rows]
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.3f} to {max_score:.3f} mm")
    print(f"   Average score: {avg_score:.3f} mm")
    
    # Count zero scores (should be home row key-pairs where no movement needed)
    zero_count = sum(1 for score in scores if score == 0.0)
    print(f"   Zero scores (no finger movement): {zero_count}")
    
    # Show some examples including zero scores
    print(f"\n📝 Sample key-pairs and scores:")
    zero_examples = [row for row in rows if float(row['distance_score']) == 0.0][:5]
    non_zero_examples = [row for row in rows if float(row['distance_score']) > 0.0][:5]
    
    print("   Zero distance pairs (no finger movement):")
    for row in zero_examples:
        print(f"     {row['key_pair']}: {float(row['distance_score']):.3f} mm")
    
    print("   Non-zero distance pairs:")
    for row in non_zero_examples:
        print(f"     {row['key_pair']}: {float(row['distance_score']):.3f} mm")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate precomputed distance scores for QWERTY key-pairs')
    parser.add_argument('--text-files', required=True,
                        help='Comma-separated list of text files to analyze')
    parser.add_argument('--output', default='output/keypair_distance_scores.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Parse text files
    text_files = [f.strip() for f in args.text_files.split(',')]
    
    # Validate text files exist
    valid_files = []
    for text_file in text_files:
        if os.path.exists(text_file):
            valid_files.append(text_file)
        else:
            print(f"Warning: Text file not found: {text_file}")
    
    if not valid_files:
        print("Error: No valid text files found")
        return 1
    
    print("Prepare distance-based key-pair scores")
    print("=" * 50)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print(f"Text files to analyze: {len(valid_files)}")
    for f in valid_files:
        print(f"  - {f}")
    
    # Compute scores using theoretical + text approach
    results = compute_all_key_pair_scores(valid_files)
    
    # Save results
    save_key_pair_scores(results, args.output)
    
    # Validate output
    validate_output(args.output)
    
    print(f"\n✅ Distance key-pair score generation complete: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())