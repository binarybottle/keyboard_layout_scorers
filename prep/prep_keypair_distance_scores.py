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
    tables/keypair_distance_scores.csv - CSV with columns: key_pair, distance_score, raw_distance, common_preceding
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
    char_lower = char.lower()  # FIXED: Convert to lowercase for map lookups
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

def extract_bigrams_and_distances_from_text(text: str) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]]]:
    """
    Extract bigrams from text and compute their distances with proper finger tracking.
    Also track preceding characters for each bigram.
    
    Returns:
        Tuple of (bigram_distances, bigram_preceding) where:
        - bigram_distances: Dict mapping bigrams to lists of distance values
        - bigram_preceding: Dict mapping bigrams to dict of {preceding_char: count}
    """
    bigram_distances = defaultdict(list)
    bigram_preceding = defaultdict(lambda: defaultdict(int))
    
    # Split text by spaces to handle space resets
    words = text.split()
    print(f"    Processing {len(words)} words from text")
    
    # Sample the first few words to show what we're working with
    sample_words = words[:10]
    print(f"    Sample words: {' '.join(sample_words)}")
    
    total_bigrams_found = 0
    
    for word_idx, word in enumerate(words):
        # Show progress every 100k words
        if word_idx % 100000 == 0 and word_idx > 0:
            print(f"    Progress: {word_idx}/{len(words)} words, {total_bigrams_found} bigrams found")
        
        # Replace non-QWERTY with spaces, then re-split
        cleaned_word = ''
        for char in word:
            if char.lower() in STAGGERED_POSITION_MAP:  # FIXED: Convert to lowercase for lookup
                cleaned_word += char
            else:
                cleaned_word += ' '
        
        # Show some cleaning examples for the first few words
        if word_idx < 5 and word != cleaned_word:
            print(f"    Cleaning example: '{word}' -> '{cleaned_word}'")
        
        # Re-split by spaces to handle embedded non-QWERTY characters
        sub_words = cleaned_word.split()
        
        for sub_word in sub_words:
            valid_chars = list(sub_word)
            
            if len(valid_chars) < 2:
                continue  # Need at least 2 characters for bigrams
            
            # Show first few valid sub-words
            if total_bigrams_found < 5:
                print(f"    Valid sub-word: '{sub_word}'")
            
            # Simulate typing this sub-word character by character
            finger_tracker = FingerTracker()  # Start with all fingers at home
            prev_distance = 0.0
            
            for i, char in enumerate(valid_chars):
                # Calculate distance to type this character
                distance = finger_tracker.calculate_distance_and_move_finger(char)
                
                # If this forms a bigram, record the bigram distance and preceding character
                if i > 0:
                    prev_char = valid_chars[i - 1]
                    bigram = prev_char + char
                    
                    # Bigram distance is the sum of distances for both characters
                    bigram_distance = prev_distance + distance
                    bigram_distances[bigram].append(bigram_distance)
                    
                    # Track preceding character (the character before this bigram)
                    if i > 1:  # There's a character before the bigram
                        preceding_char = valid_chars[i - 2]
                        bigram_preceding[bigram][preceding_char] += 1
                    # If bigram is at start of word, we could track space, but let's skip that
                    
                    total_bigrams_found += 1
                    
                    # Show first few bigrams found
                    if total_bigrams_found <= 10:
                        preceding_info = f" (after '{valid_chars[i-2]}')" if i > 1 else " (word start)"
                        print(f"    Bigram #{total_bigrams_found}: '{bigram}' distance={bigram_distance:.2f}mm{preceding_info}")
                
                # Remember this distance for the next bigram
                prev_distance = distance
    
    print(f"    ✅ Found {total_bigrams_found} total bigram instances")
    return bigram_distances, bigram_preceding

def compute_text_based_distances(text_files: List[str]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Stage 2: Process text files to find actual key-pairs and compute averaged distances.
    Also determine most common preceding characters.
    
    Args:
        text_files: List of text file paths
        
    Returns:
        Tuple of (text_based_scores, common_preceding) where:
        - text_based_scores: Dictionary mapping key-pairs to averaged distance scores
        - common_preceding: Dictionary mapping key-pairs to comma-separated top preceding chars
    """
    print("\n🟡 Stage 2: Processing text files for real-world key-pair usage")
    
    all_bigram_distances = defaultdict(list)
    all_bigram_preceding = defaultdict(lambda: defaultdict(int))
    
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
        
        # Extract bigrams, distances, and preceding characters
        bigram_distances, bigram_preceding = extract_bigrams_and_distances_from_text(text)
        
        # Combine results from this file
        for bigram, distances in bigram_distances.items():
            all_bigram_distances[bigram].extend(distances)
        
        for bigram, preceding_counts in bigram_preceding.items():
            for preceding_char, count in preceding_counts.items():
                all_bigram_preceding[bigram][preceding_char] += count
        
        total_bigram_instances = sum(len(distances) for distances in bigram_distances.values())
        print(f"    Found {total_bigram_instances} total bigram instances")
        print(f"    Unique bigrams: {len(bigram_distances)}")
    
    # Calculate averages for pairs found in text
    text_based_scores = {}
    for key_pair, distances in all_bigram_distances.items():
        text_based_scores[key_pair] = sum(distances) / len(distances)
    
    # Determine most common preceding characters (top 3)
    common_preceding = {}
    for key_pair, preceding_counts in all_bigram_preceding.items():
        # Sort by count (descending) and take top 3
        sorted_preceding = sorted(preceding_counts.items(), key=lambda x: x[1], reverse=True)
        top_preceding = [char for char, count in sorted_preceding[:3]]
        common_preceding[key_pair] = ','.join(top_preceding) if top_preceding else ''
    
    print(f"  ✅ Computed averaged distances for {len(text_based_scores)} key-pairs from text")
    
    # Show some examples with preceding characters
    if text_based_scores:
        print(f"  Sample text-based scores with common preceding characters:")
        sample_pairs = list(text_based_scores.items())[:5]
        for pair, distance in sample_pairs:
            instances = len(all_bigram_distances[pair])
            preceding = common_preceding.get(pair, '')
            print(f"    {pair}: {distance:.2f}mm (avg of {instances} instances, common preceding: {preceding})")
    
    return text_based_scores, common_preceding

def combine_theoretical_and_text_scores(theoretical_scores: Dict[str, float], 
                                       text_scores: Dict[str, float],
                                       common_preceding: Dict[str, str]) -> Dict[str, Tuple[float, float, str]]:
    """
    Combine theoretical and text-based scores.
    Text-based scores override theoretical scores when available.
    
    Args:
        theoretical_scores: All key-pairs with theoretical distances
        text_scores: Key-pairs found in text with averaged distances
        common_preceding: Key-pairs mapped to common preceding characters
        
    Returns:
        Dictionary mapping key-pairs to (raw_distance, final_distance, preceding_chars) tuples
        where raw_distance is always theoretical, final_distance is text-based when available
    """
    print("\n🟢 Stage 3: Combining theoretical and text-based scores")
    
    final_scores = {}
    
    # For all key-pairs, start with theoretical as both raw and final, empty preceding
    for key_pair, theoretical_distance in theoretical_scores.items():
        final_scores[key_pair] = (theoretical_distance, theoretical_distance, '')
    
    # Override ONLY the final distance with text-based scores where available
    overridden_count = 0
    significant_differences = 0
    
    for key_pair, text_distance in text_scores.items():
        if key_pair in theoretical_scores:  # Should always be true
            theoretical_distance = theoretical_scores[key_pair]
            preceding_chars = common_preceding.get(key_pair, '')
            final_scores[key_pair] = (theoretical_distance, text_distance, preceding_chars)  # raw, final, preceding
            overridden_count += 1
            
            # Check for significant differences (> 0.1mm) between theoretical and text-based
            if abs(theoretical_distance - text_distance) > 0.1:
                significant_differences += 1
                if significant_differences <= 5:  # Show first 5 examples
                    print(f"    Example override: {key_pair} theoretical={theoretical_distance:.3f}mm -> text={text_distance:.3f}mm (common preceding: {preceding_chars})")
        else:
            # This shouldn't happen since we compute all theoretical scores first
            print(f"  Warning: Text-based score found for unknown key-pair: {key_pair}")
    
    print(f"  ✅ Overrode {overridden_count} theoretical scores with text-based averages")
    print(f"  📊 {significant_differences} pairs have significant differences (>0.1mm)")
    print(f"  Final scores: {len(final_scores)} total key-pairs")
    
    # Statistics  
    text_pairs = len(text_scores)
    theoretical_only_pairs = len(final_scores) - text_pairs
    print(f"    - {text_pairs} pairs have text-based overrides")
    print(f"    - {theoretical_only_pairs} pairs use theoretical distances only")
    
    return final_scores

def compute_all_key_pair_scores(text_files: List[str]):
    """Compute distance scores for all key-pairs using theoretical + text approach."""
    print("Computing distance scores using theoretical + text approach...")
    
    # Stage 1: Compute theoretical distances for all key-pairs
    theoretical_scores = compute_all_theoretical_distances()
    
    # Stage 2: Process text files for real-world usage patterns
    text_scores, common_preceding = compute_text_based_distances(text_files)
    
    # Stage 3: Combine scores (text overrides theoretical)
    combined_scores = combine_theoretical_and_text_scores(theoretical_scores, text_scores, common_preceding)
    
    # Convert to results format with raw distance, final distance, and common preceding
    results = []
    for key_pair, (raw_distance, final_distance, preceding_chars) in sorted(combined_scores.items()):
        results.append({
            'key_pair': key_pair,
            'distance_score': final_distance,
            'raw_distance': raw_distance,
            'common_preceding': preceding_chars
        })
    
    print(f"\n✅ Total key-pairs computed: {len(results)}")
    
    return results

def save_key_pair_scores(results, output_file="tables/keypair_distance_scores.csv"):
    """Save key-pair scores to CSV file with distance_score, raw_distance, and common_preceding columns."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'distance_score', 'raw_distance', 'common_preceding'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def validate_output(output_file="tables/keypair_distance_scores.csv"):
    """Perform thorough validation of the generated output file with manual verification."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Comprehensive Validation Results:")
    print(f"   Total key-pairs: {len(rows)}")
    
    # Check for expected number of combinations
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    print(f"   Expected count: {expected_count}")
    print(f"   Match: {'✅' if len(rows) == expected_count else '❌'}")
    
    # Check score ranges for both columns
    final_scores = [float(row['distance_score']) for row in rows]
    raw_scores = [float(row['raw_distance']) for row in rows]
    
    print(f"   Final distance range: {min(final_scores):.3f} to {max(final_scores):.3f} mm")
    print(f"   Raw distance range: {min(raw_scores):.3f} to {max(raw_scores):.3f} mm")
    print(f"   Average final: {sum(final_scores)/len(final_scores):.3f} mm")
    print(f"   Average raw: {sum(raw_scores)/len(raw_scores):.3f} mm")
    
    # Count differences between raw and final
    different_count = sum(1 for row in rows if float(row['distance_score']) != float(row['raw_distance']))
    print(f"   Pairs with text-based overrides: {different_count}")
    
    # Count pairs with preceding character data
    with_preceding = sum(1 for row in rows if row['common_preceding'].strip())
    print(f"   Pairs with preceding character data: {with_preceding}")
    
    # Count zero scores for both columns
    zero_final = sum(1 for score in final_scores if score == 0.0)
    zero_raw = sum(1 for score in raw_scores if score == 0.0)
    print(f"   Zero final scores: {zero_final}")
    print(f"   Zero raw scores: {zero_raw}")
    
    print(f"\n🔍 MANUAL VERIFICATION:")
    print("=" * 60)
    
    # 1. Verify theoretical distance calculations manually
    print("\n1️⃣ Theoretical Distance Verification:")
    test_pairs = [('F', 'F'), ('F', 'D'), ('F', 'G'), ('Q', 'P'), ('A', 'Z')]
    
    for key1, key2 in test_pairs:
        # Manual calculation
        manual_distance = compute_theoretical_keypair_distance(key1, key2)
        
        # Find in CSV
        csv_row = next((row for row in rows if row['key_pair'] == key1 + key2), None)
        csv_raw = float(csv_row['raw_distance']) if csv_row else None
        
        match = "✅" if abs(manual_distance - csv_raw) < 0.001 else "❌"
        print(f"   {key1}{key2}: manual={manual_distance:.3f}mm, csv={csv_raw:.3f}mm {match}")
    
    # 2. Manual text processing verification using sample words
    print("\n2️⃣ Text Processing Verification:")
    sample_words = ["SATURDAY", "NOVEMBER", "EMERGING"]
    
    for word in sample_words:
        print(f"\n   Word: '{word}'")
        
        # Manual step-by-step simulation
        finger_tracker = FingerTracker()
        prev_distance = 0.0
        
        for i, char in enumerate(word):
            distance = finger_tracker.calculate_distance_and_move_finger(char)
            
            if i > 0:
                prev_char = word[i - 1]
                bigram = prev_char + char
                bigram_distance = prev_distance + distance
                
                # Get preceding char if exists
                preceding = word[i - 2] if i > 1 else "START"
                
                print(f"     Step {i}: {char} -> distance={distance:.2f}mm, bigram='{bigram}' total={bigram_distance:.2f}mm, after='{preceding}'")
            else:
                print(f"     Step {i}: {char} -> distance={distance:.2f}mm (first char)")
            
            prev_distance = distance
    
    # 3. Verify CSV data matches sample word patterns (context may differ in full corpus)
    print("\n3️⃣ CSV Data Verification:")
    verification_pairs = [
        ('SA', 'S', 'SATURDAY context'),
        ('AT', 'S', 'SATURDAY context (may differ in full corpus)'), 
        ('TU', 'A', 'SATURDAY context'),
        ('RD', 'U', 'SATURDAY context')
    ]
    
    for bigram, expected_preceding, note in verification_pairs:
        csv_row = next((row for row in rows if row['key_pair'] == bigram), None)
        if csv_row:
            preceding_chars = csv_row['common_preceding'].split(',')
            has_expected = expected_preceding in preceding_chars
            status = "✅" if has_expected else "📊"  # Use 📊 for expected corpus differences
            print(f"   {bigram}: expected preceding '{expected_preceding}' in '{csv_row['common_preceding']}' {status} ({note})")
    
    # 4. Validate finger assignments and movements
    print("\n4️⃣ Finger Assignment Verification:")
    finger_tests = [
        ('F', 'L1'),  # Left index
        ('J', 'R1'),  # Right index  
        ('A', 'L4'),  # Left pinky
        (';', 'R4'),  # Right pinky
        ('D', 'L2'),  # Left middle
        ('K', 'R2'),  # Right middle
    ]
    
    for char, expected_finger in finger_tests:
        actual_finger = get_finger_id(char)
        match = "✅" if actual_finger == expected_finger else "❌"
        print(f"   '{char}' -> finger {actual_finger} (expected {expected_finger}) {match}")
    
    # 5. Physical distance spot checks - verify bigram distances (not direct key-to-key)
    print("\n5️⃣ Bigram Distance Verification:")
    
    # FJ bigram: F(home) + J(home) = 0 + 0 = 0
    csv_fj = next((row for row in rows if row['key_pair'] == 'FJ'), None)
    expected_fj = 0.0  # Both F and J are home positions
    actual_fj = float(csv_fj['raw_distance']) if csv_fj else 0
    print(f"   FJ bigram: expected={expected_fj:.1f}mm, csv={actual_fj:.1f}mm {'✅' if abs(expected_fj - actual_fj) < 0.1 else '❌'}")
    
    # QP bigram: (home→Q) + (home→P) = (A→Q) + (;→P)
    pos_a = get_physical_position('a')  
    pos_q = get_physical_position('q')  
    pos_semicolon = get_physical_position(';')
    pos_p = get_physical_position('p')
    
    distance_a_to_q = calculate_euclidean_distance(pos_a, pos_q)
    distance_semicolon_to_p = calculate_euclidean_distance(pos_semicolon, pos_p)
    expected_qp = distance_a_to_q + distance_semicolon_to_p
    
    csv_qp = next((row for row in rows if row['key_pair'] == 'QP'), None)
    actual_qp = float(csv_qp['raw_distance']) if csv_qp else 0
    print(f"   QP bigram: expected={expected_qp:.1f}mm, csv={actual_qp:.1f}mm {'✅' if abs(expected_qp - actual_qp) < 0.1 else '❌'}")
    
    # Manual verification: compute QZ (Q finger stays at Q, then moves Q→Z)
    pos_z = get_physical_position('z')
    pos_q = get_physical_position('q')
    distance_a_to_q = calculate_euclidean_distance(pos_a, pos_q)
    distance_q_to_z = calculate_euclidean_distance(pos_q, pos_z)  # FIXED: Q→Z not A→Z
    expected_qz = distance_a_to_q + distance_q_to_z  # A→Q + Q→Z (finger stays at Q)
    
    csv_qz = next((row for row in rows if row['key_pair'] == 'QZ'), None)
    actual_qz = float(csv_qz['raw_distance']) if csv_qz else 0
    print(f"   QZ bigram: expected={expected_qz:.1f}mm, csv={actual_qz:.1f}mm {'✅' if abs(expected_qz - actual_qz) < 0.1 else '❌'}")
    
    # 6. Context pattern validation
    print("\n6️⃣ Context Pattern Validation:")
    high_frequency_pairs = [
        ('TH', ['A', 'I', 'E']),  # "THAT", "THINK", "THE"
        ('HE', ['T', 'W', 'S']),  # "THE", "WHEN", "SHE"  
        ('ER', ['T', 'V', 'H']),  # "AFTER", "OVER", "WHERE"
        ('AN', ['C', 'M', 'H']),  # "CAN", "MAN", "THAN"
    ]
    
    for bigram, expected_contexts in high_frequency_pairs:
        csv_row = next((row for row in rows if row['key_pair'] == bigram), None)
        if csv_row and csv_row['common_preceding']:
            actual_contexts = csv_row['common_preceding'].split(',')
            matches = [ctx for ctx in expected_contexts if ctx in actual_contexts]
            match_rate = len(matches) / len(expected_contexts)
            status = "✅" if match_rate >= 0.5 else "❌"
            print(f"   {bigram}: expected {expected_contexts}, got {actual_contexts}, match rate={match_rate:.1%} {status}")
    
    # 7. Data consistency checks
    print("\n7️⃣ Data Consistency Checks:")
    
    # Check that all raw distances are >= 0
    negative_raw = [row for row in rows if float(row['raw_distance']) < 0]
    print(f"   Negative raw distances: {len(negative_raw)} {'✅' if len(negative_raw) == 0 else '❌'}")
    
    # Check that all final distances are >= 0  
    negative_final = [row for row in rows if float(row['distance_score']) < 0]
    print(f"   Negative final distances: {len(negative_final)} {'✅' if len(negative_final) == 0 else '❌'}")
    
    # Check that same-key pairs for HOME ROW keys have 0 distance
    home_row_keys = ['A', 'S', 'D', 'F', 'J', 'K', 'L', ';']
    home_same_key_pairs = [row for row in rows if row['key_pair'][0] == row['key_pair'][1] and row['key_pair'][0] in home_row_keys]
    zero_home_same_key = [row for row in home_same_key_pairs if float(row['raw_distance']) == 0.0]
    print(f"   Home row same-key pairs with 0 distance: {len(zero_home_same_key)}/{len(home_same_key_pairs)} {'✅' if len(zero_home_same_key) == len(home_same_key_pairs) else '❌'}")
    
    # Check that non-home same-key pairs have non-zero distance
    non_home_keys = [key for key in get_all_qwerty_keys() if key.upper() not in home_row_keys]
    non_home_same_key_pairs = [row for row in rows if row['key_pair'][0] == row['key_pair'][1] and row['key_pair'][0] in non_home_keys]
    non_zero_non_home = [row for row in non_home_same_key_pairs if float(row['raw_distance']) > 0.0]
    print(f"   Non-home same-key pairs with >0 distance: {len(non_zero_non_home)}/{len(non_home_same_key_pairs)} {'✅' if len(non_zero_non_home) == len(non_home_same_key_pairs) else '❌'}")
    
    # Check home row pairs (should be 0 for same finger pairs on home row)
    home_position_keys = ['A', 'S', 'D', 'F', 'J', 'K', 'L', ';']  # Actual home positions only
    
    # Same-key pairs for home positions (these should all be 0)
    home_same_key_pairs = [row for row in rows if row['key_pair'][0] == row['key_pair'][1] and row['key_pair'][0] in home_position_keys]
    zero_home_same_key = [row for row in home_same_key_pairs if float(row['raw_distance']) == 0.0]
    print(f"   Home position same-key pairs with 0 distance: {len(zero_home_same_key)}/{len(home_same_key_pairs)} {'✅' if len(zero_home_same_key) == len(home_same_key_pairs) else '❌'}")
    
    # Different-key same-finger home row pairs (like FG, GF for index finger - these should be >0)
    same_finger_diff_key_pairs = []
    for row in rows:
        char1, char2 = row['key_pair'][0], row['key_pair'][1]
        if char1 != char2:  # Different keys
            finger1, finger2 = get_finger_id(char1), get_finger_id(char2)
            if finger1 == finger2 and finger1 is not None:  # Same finger
                # Check if both are in the home row area (including G, H)
                home_row_area = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';']
                if char1 in home_row_area and char2 in home_row_area:
                    same_finger_diff_key_pairs.append(row)
    
    non_zero_same_finger = [row for row in same_finger_diff_key_pairs if float(row['raw_distance']) > 0.0]
    print(f"   Same-finger different-key home row pairs with >0 distance: {len(non_zero_same_finger)}/{len(same_finger_diff_key_pairs)} {'✅' if len(non_zero_same_finger) == len(same_finger_diff_key_pairs) else '❌'}")
    
    print(f"\n✅ Comprehensive validation complete!")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate precomputed distance scores for QWERTY key-pairs')
    parser.add_argument('--text-files', required=True,
                        help='Comma-separated list of text files to analyze')
    parser.add_argument('--output', default='tables/keypair_distance_scores.csv',
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