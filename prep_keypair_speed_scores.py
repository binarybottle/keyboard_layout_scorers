#!/usr/bin/env python3
"""
Generate precomputed speed scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes speed scores for every possible combination of QWERTY keys
using CSV typing data with a comprehensive fallback strategy:

1. Extract timing data for all key-to-key movements from CSV typing data
2. For each bigram, calculate:
   - key1_time: time from key1's home position to key1 (e.g., D→E for key1=E)
   - key1_to_key2_time: time from key1 to key2 (e.g., E→R)
   - total_time: key1_time + key1_to_key2_time
   - speed_score: 2000ms / total_time (2 keystrokes per total time)
3. For missing data, use fallback in this priority order:
   - Mirror pair data (e.g., for ./ use XZ, for ;' use SA)
   - Special cases: for '[ use ;P data, for [' use P; data
   - Left-hand data if available
   - Minimum speed from all empirical data

Processing approach:
- Analyzes consecutive keystrokes within words (do not span across spaces)
- Example: "HELLO" produces bigrams HE, EL, LL, LO with their respective timing components
- Uses actual CSV keystroke timing data only (no theoretical models)

Usage:
    python prep_keypair_speed_scores.py --input-dir /path/to/csv/files/

Output:
    output/keypair_speed_scores.csv - CSV with columns: 
    key_pair, speed_score, key1_time, key1_to_key2_time, total_time, fallback
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict

# Left home block keys (from analyze_raw_data.py)
LEFT_HOME_KEYS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']

# Mirror pairs (from analyze_raw_data.py)
MIRROR_PAIRS = [
    ('q', 'p'), ('w', 'o'), ('e', 'i'), ('r', 'u'),
    ('a', ';'), ('s', 'l'), ('d', 'k'), ('f', 'j'), ('g', 'h'),
    ('z', '/'), ('x', '.'), ('c', ','), ('v', 'm')
]

# Home row positions for each key (which finger types which key)
HOME_KEY_MAP = {
    # Left hand
    'q': 'a', 'w': 's', 'e': 'd', 'r': 'f', 't': 'f',
    'a': 'a', 's': 's', 'd': 'd', 'f': 'f', 'g': 'f', 
    'z': 'a', 'x': 's', 'c': 'd', 'v': 'f', 'b': 'f',
    # Right hand  
    'y': 'j', 'u': 'j', 'i': 'k', 'o': 'l', 'p': ';',
    'h': 'j', 'j': 'j', 'k': 'k', 'l': 'l', ';': ';',
    'n': 'j', 'm': 'j', ',': 'k', '.': 'l', '/': ';',
    '[': ';', "'": ';'
}

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

def process_typing_data(data):
    """Process typing data to calculate typing time between consecutive keystrokes."""
    data = data.copy()
    
    # Convert expectedKey and typedKey to string
    data.loc[:, 'expectedKey'] = data['expectedKey'].astype(str)
    data.loc[:, 'typedKey'] = data['typedKey'].astype(str)
    
    # Replace 'nan' strings with empty strings
    data.loc[:, 'expectedKey'] = data['expectedKey'].replace('nan', '')
    data.loc[:, 'typedKey'] = data['typedKey'].replace('nan', '')
    
    # Ensure isCorrect is boolean
    data.loc[:, 'isCorrect'] = data['isCorrect'].map(lambda x: str(x).lower() == 'true')
    
    processed = []
    
    # Sort by user and timestamp
    sorted_data = data.sort_values(by=['user_id', 'keydownTime'])
    
    # Group by user_id
    for user_id, user_data in sorted_data.groupby('user_id'):
        user_rows = user_data.to_dict('records')
        
        # Calculate typing time for each keystroke
        for i in range(1, len(user_rows)):
            current = user_rows[i]
            previous = user_rows[i-1]
            
            # Calculate time difference in milliseconds
            typing_time = current['keydownTime'] - previous['keydownTime']
            
            processed.append({
                'user_id': user_id,
                'trialId': current['trialId'],
                'expectedKey': current['expectedKey'],
                'typedKey': current['typedKey'],
                'isCorrect': current['isCorrect'],
                'typingTime': typing_time,
                'keydownTime': current['keydownTime'],
                'prevKey': previous['expectedKey']
            })
    
    return pd.DataFrame(processed)

def extract_key_to_key_timings(data):
    """
    Extract timing data for all key-to-key movements from CSV data.
    Returns dictionary mapping (key1, key2) -> list of timing values.
    """
    key_timings = defaultdict(list)
    
    # Group by user_id and trialId
    grouped = data.groupby(['user_id', 'trialId'])
    
    for (user_id, trial_id), trial_data in grouped:
        # Sort by timestamp
        sorted_trial = trial_data.sort_values('keydownTime').reset_index(drop=True)
        
        # Process consecutive keystrokes within words
        for i in range(len(sorted_trial) - 1):
            current = sorted_trial.iloc[i]
            next_key = sorted_trial.iloc[i+1]
            
            key1 = current['expectedKey']
            key2 = next_key['expectedKey']
            
            # Skip if either key is not a valid single character or is a space
            if (not isinstance(key1, str) or len(key1) != 1 or
                not isinstance(key2, str) or len(key2) != 1 or
                key1 == ' ' or key2 == ' '):
                continue
            
            # Skip same-key transitions
            if key1 == key2:
                continue
            
            # Calculate timing if both keys are correct
            if current['isCorrect'] and next_key['isCorrect']:
                timing = next_key['keydownTime'] - current['keydownTime']
                if 50 <= timing <= 2000:  # Reasonable time range
                    key_pair = (key1.lower(), key2.lower())
                    key_timings[key_pair].append(timing)
    
    return key_timings

def calculate_median_timings(key_timings):
    """Calculate median timing for each key-to-key movement."""
    median_timings = {}
    
    for key_pair, timings in key_timings.items():
        if len(timings) >= 3:  # Need sufficient data
            median_timings[key_pair] = np.median(timings)
    
    return median_timings

def create_mirror_mapping():
    """Create a comprehensive mirror mapping for key-pairs."""
    mirror_map = {}
    
    # Add basic mirror pairs
    for left, right in MIRROR_PAIRS:
        mirror_map[left] = right
        mirror_map[right] = left
    
    # Special cases
    special_mirrors = {
        "'": ';',
        '[': 'p',
    }
    
    for key, mirror in special_mirrors.items():
        mirror_map[key] = mirror
        mirror_map[mirror] = key
    
    return mirror_map

def get_mirror_bigram(bigram, mirror_map):
    """Get the mirror bigram for a given bigram."""
    if len(bigram) != 2:
        return None
    
    key1, key2 = bigram[0].lower(), bigram[1].lower()
    
    mirror1 = mirror_map.get(key1)
    mirror2 = mirror_map.get(key2)
    
    if mirror1 and mirror2:
        return mirror1 + mirror2
    
    return None

def calculate_bigram_components(bigram, median_timings, mirror_map):
    """
    Calculate the timing components for a bigram.
    Returns (key1_time, key1_to_key2_time, total_time, fallback_used)
    """
    key1, key2 = bigram[0].lower(), bigram[1].lower()
    
    # Get home keys for the fingers that type key1 and key2
    key1_home = HOME_KEY_MAP.get(key1)
    key2_home = HOME_KEY_MAP.get(key2)
    
    if not key1_home or not key2_home:
        return 0, 0, 0, "no_mapping"
    
    # Calculate key1_time: time from home to key1
    key1_movement = (key1_home, key1)
    key1_time = median_timings.get(key1_movement, 0)
    
    # Calculate key1_to_key2_time: time from key1 to key2
    key1_to_key2_movement = (key1, key2)
    key1_to_key2_time = median_timings.get(key1_to_key2_movement, 0)
    
    fallback_used = ""
    
    # Try fallbacks if either component is missing
    if key1_time == 0:
        # Try mirror for key1_movement
        if key1_home in mirror_map and key1 in mirror_map:
            mirror_key1_home = mirror_map[key1_home]
            mirror_key1 = mirror_map[key1]
            mirror_movement = (mirror_key1_home, mirror_key1)
            key1_time = median_timings.get(mirror_movement, 0)
            if key1_time > 0:
                fallback_used += "mirror_key1_movement;"
    
    if key1_to_key2_time == 0:
        # Try mirror for key1_to_key2_movement
        if key1 in mirror_map and key2 in mirror_map:
            mirror_key1 = mirror_map[key1]
            mirror_key2 = mirror_map[key2]
            mirror_movement = (mirror_key1, mirror_key2)
            key1_to_key2_time = median_timings.get(mirror_movement, 0)
            if key1_to_key2_time > 0:
                fallback_used += "mirror_transition;"
    
    total_time = key1_time + key1_to_key2_time
    
    return key1_time, key1_to_key2_time, total_time, fallback_used.rstrip(';')

def load_and_process_csv_data(input_dir: str):
    """Load and process CSV files from a directory to extract key-to-key timings."""
    print(f"\n🔵 Loading and processing CSV files from directory: {input_dir}")
    
    # Find all CSV files in the directory
    csv_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(input_dir, file))
    
    if not csv_files:
        print(f"Error: No CSV files found in directory: {input_dir}")
        return None, None
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"  Processing: {os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                # Extract user ID from filename if not present
                if 'user_id' not in df.columns:
                    filename = os.path.basename(csv_file)
                    user_id = filename.split('_')[2] if '_' in filename else filename.split('.')[0]
                    df['user_id'] = user_id
                
                all_data.append(df)
        except Exception as e:
            print(f"    Warning: Error reading {csv_file}: {e}")
            continue
    
    if not all_data:
        print("Error: No data was successfully loaded from any CSV file.")
        return None, None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_data)} total typing data records")
    
    # Filter out intro trials
    filtered_data = combined_data[combined_data['trialId'] != 'intro-trial-1'].copy()
    print(f"After filtering: {len(filtered_data)} records")
    
    # Process the data to calculate typing times
    processed_data = process_typing_data(filtered_data)
    
    # Extract key-to-key timings
    key_timings = extract_key_to_key_timings(processed_data)
    print(f"  ✅ Found {len(key_timings)} unique key-to-key movements")
    
    # Calculate median timings
    median_timings = calculate_median_timings(key_timings)
    print(f"  ✅ Calculated median timings for {len(median_timings)} movements")
    
    return median_timings, key_timings

def compute_all_keypair_speeds(input_dir: str):
    """Compute speed scores for all key-pairs using timing components from CSV data."""
    print("Computing key-pair speeds using timing components from CSV data...")
    
    # Load and process CSV data
    median_timings, key_timings = load_and_process_csv_data(input_dir)
    
    if median_timings is None:
        return []
    
    # Create mirror mapping
    mirror_map = create_mirror_mapping()
    
    # Generate all possible key-pairs
    all_key_pairs = generate_all_key_pairs()
    
    print(f"\n🟡 Stage 2: Computing speeds for all {len(all_key_pairs)} key-pairs")
    
    results = []
    successful_calcs = 0
    fallback_used_count = 0
    no_data_count = 0
    
    # Calculate minimum total time for fallback
    all_total_times = []
    
    for i, key_pair in enumerate(all_key_pairs):
        if i % 100 == 0:
            print(f"    Progress: {i}/{len(all_key_pairs)} ({i/len(all_key_pairs)*100:.1f}%)")
        
        key1_time, key1_to_key2_time, total_time, fallback_info = calculate_bigram_components(
            key_pair, median_timings, mirror_map
        )
        
        if total_time > 0:
            all_total_times.append(total_time)
    
    # Calculate minimum time for final fallback
    minimum_total_time = min(all_total_times) if all_total_times else 400  # 400ms default
    minimum_speed = 2000.0 / minimum_total_time
    
    print(f"  ✅ Minimum total time found: {minimum_total_time:.1f}ms (speed: {minimum_speed:.3f} keys/sec)")
    
    # Now calculate all speeds
    for i, key_pair in enumerate(all_key_pairs):
        key1_time, key1_to_key2_time, total_time, fallback_info = calculate_bigram_components(
            key_pair, median_timings, mirror_map
        )
        
        fallback = ""
        
        if total_time > 0:
            speed_score = 2000.0 / total_time  # 2 keystrokes per total time
            successful_calcs += 1
            if fallback_info:
                fallback = fallback_info
                fallback_used_count += 1
        else:
            # Use minimum timing as final fallback
            key1_time = minimum_total_time / 2
            key1_to_key2_time = minimum_total_time / 2
            total_time = minimum_total_time
            speed_score = minimum_speed
            fallback = "minimum"
            no_data_count += 1
        
        results.append({
            'key_pair': key_pair.upper(),
            'speed_score': speed_score,
            'key1_time': key1_time,
            'key1_to_key2_time': key1_to_key2_time,
            'total_time': total_time,
            'fallback': fallback
        })
    
    print(f"  ✅ Speed computation complete:")
    print(f"    - {successful_calcs} pairs calculated from data")
    print(f"    - {fallback_used_count} pairs used fallback data")
    print(f"    - {no_data_count} pairs used minimum fallback ({minimum_speed:.3f} keys/sec)")
    
    return results

def save_key_pair_speeds(results, output_file="output/keypair_speed_scores.csv"):
    """Save key-pair speeds to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'speed_score', 'key1_time', 'key1_to_key2_time', 'total_time', 'fallback'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair speeds to: {output_file}")

def validate_output(output_file="output/keypair_speed_scores.csv"):
    """Perform validation of the generated output file with timing component analysis."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Timing Component Validation Results:")
    print(f"   Total key-pairs: {len(rows)}")
    
    # Check for expected number of combinations
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    print(f"   Expected count: {expected_count}")
    print(f"   Match: {'✅' if len(rows) == expected_count else '❌'}")
    
    # Analyze timing components
    speeds = [float(row['speed_score']) for row in rows]
    key1_times = [float(row['key1_time']) for row in rows]
    transition_times = [float(row['key1_to_key2_time']) for row in rows]
    total_times = [float(row['total_time']) for row in rows]
    
    print(f"   Speed range: {min(speeds):.3f} to {max(speeds):.3f} keys/sec")
    print(f"   Key1 time range: {min(key1_times):.1f} to {max(key1_times):.1f}ms")
    print(f"   Transition time range: {min(transition_times):.1f} to {max(transition_times):.1f}ms")
    print(f"   Total time range: {min(total_times):.1f} to {max(total_times):.1f}ms")
    
    # Count fallback types
    fallback_types = {}
    for row in rows:
        fallback = row['fallback']
        if fallback == "":
            fallback = "empirical"
        fallback_types[fallback] = fallback_types.get(fallback, 0) + 1
    
    print(f"   Fallback types:")
    for fallback_type, count in sorted(fallback_types.items()):
        print(f"     {fallback_type}: {count}")
    
    # Test timing component math
    print(f"\n🔍 Timing Component Math Validation:")
    math_errors = 0
    for row in rows:
        key1_time = float(row['key1_time'])
        transition_time = float(row['key1_to_key2_time'])
        total_time = float(row['total_time'])
        speed_score = float(row['speed_score'])
        
        expected_total = key1_time + transition_time
        expected_speed = 2000.0 / total_time if total_time > 0 else 0
        
        if abs(expected_total - total_time) > 0.1:
            math_errors += 1
        if abs(expected_speed - speed_score) > 0.001:
            math_errors += 1
    
    print(f"   Math consistency errors: {math_errors} {'✅' if math_errors == 0 else '❌'}")
    
    # Show examples
    print(f"\n   Sample timing breakdowns:")
    print("     Key-Pair | Speed | Key1 Time | Transition | Total | Fallback")
    print("     ---------|-------|-----------|------------|-------|----------")
    sample_pairs = ['AS', 'SD', 'DF', 'ER', 'QP', 'ZX']
    
    for pair in sample_pairs:
        for row in rows:
            if row['key_pair'] == pair:
                fallback_display = row['fallback'] if row['fallback'] else "empirical"
                print(f"     {pair:8} | {float(row['speed_score']):5.2f} | {float(row['key1_time']):9.1f} | {float(row['key1_to_key2_time']):10.1f} | {float(row['total_time']):5.1f} | {fallback_display}")
                break
    
    print(f"\n✅ Timing component validation complete!")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate timing-component speed scores for QWERTY key-pairs')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing CSV files with typing data')
    parser.add_argument('--output', default='output/keypair_speed_scores.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        return 1
    
    # Check if directory contains CSV files
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in directory: {args.input_dir}")
        return 1
    
    print("Generate timing-component key-pair speed scores from CSV data")
    print("=" * 60)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print(f"Input directory: {args.input_dir}")
    print(f"CSV files found: {len(csv_files)}")
    
    # Compute speeds using timing components
    results = compute_all_keypair_speeds(args.input_dir)
    
    # Save results
    save_key_pair_speeds(results, args.output)
    
    # Validate output
    validate_output(args.output)
    
    print(f"\n✅ Timing-component speed generation complete: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())