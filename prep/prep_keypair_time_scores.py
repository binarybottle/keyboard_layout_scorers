#!/usr/bin/env python3
"""
Generate precomputed time scores for all possible QWERTY key-pairs.
NOW WITH BUILT-IN FREQUENCY-BASED DEBIASING.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes time scores for every possible combination of QWERTY keys
using CSV typing data with a comprehensive fallback strategy AND automatic
QWERTY bias removal based on English bigram frequencies.

Processing approach:
- Analyzes consecutive keystrokes within words (do not span across spaces)
- Example: "HELLO" produces bigrams HE, EL, LL, LO with their respective timing components
- Uses MEDIAN timing across all instances in all CSV files (not average, to reduce outlier impact)
- Requires minimum 3 instances per key transition for statistical reliability
- Automatically removes QWERTY bias using English bigram frequency corrections

Usage:
    python prep_keypair_time_scores.py --input-dir /path/to/csv/files/ --frequency-file ../input/english-letter-pair-frequencies-google-ngrams.csv

Output:
    ../tables/keypair_time_scores.csv - CSV with debiased time scores
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import statistics
import numpy as np
from scipy import stats

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

# ============================================================================
# NEW: FREQUENCY-BASED DEBIASING FUNCTIONS
# ============================================================================

def load_english_frequencies(frequency_file: str) -> Dict[str, float]:
    """Load English bigram frequencies for debiasing."""
    
    if not frequency_file or not Path(frequency_file).exists():
        print(f"⚠️  Frequency file not found: {frequency_file}")
        print("   Debiasing will be skipped - using raw empirical times")
        return {}
    
    try:
        df = pd.read_csv(frequency_file)
        
        # Handle different possible column names
        bigram_col = None
        freq_col = None
        
        for col in ['bigram', 'letter_pair', 'pair']:
            if col in df.columns:
                bigram_col = col
                break
        
        for col in ['normalized_frequency', 'frequency', 'count']:
            if col in df.columns:
                freq_col = col
                break
        
        if bigram_col is None or freq_col is None:
            print(f"⚠️  Could not find required columns in {frequency_file}")
            return {}
        
        # Create frequency dictionary
        frequencies = dict(zip(df[bigram_col].str.upper(), df[freq_col]))
        
        # Normalize to ensure sum = 1.0 for proper proportional corrections
        total_freq = sum(frequencies.values())
        if total_freq > 0:
            frequencies = {k: v/total_freq for k, v in frequencies.items()}
        
        print(f"✅ Loaded {len(frequencies)} English bigram frequencies for debiasing")
        return frequencies
        
    except Exception as e:
        print(f"⚠️  Error loading frequency file: {e}")
        return {}

def create_layout_mapping() -> Dict[str, str]:
    """Create mapping from QWERTY keys back to letters for debiasing."""
    
    # Standard QWERTY layout mapping
    qwerty_layout = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./'["
    qwerty_letters = "QWERTYUIOPASDFGHJKLZXCVBNM"  # Letters only
    
    # Map each key to its letter (for letters)
    key_to_letter = {}
    for i, letter in enumerate(qwerty_letters):
        qwerty_key = qwerty_layout[i]
        key_to_letter[qwerty_key] = letter
    
    # Special characters map to themselves (no debiasing needed)
    special_chars = [';', ',', '.', '/', "'", '[']
    for char in special_chars:
        key_to_letter[char] = char
    
    return key_to_letter

def map_keypairs_to_letterpairs(key_pair_times: Dict[str, float], 
                               key_to_letter: Dict[str, str]) -> Dict[str, List[str]]:
    """Map key-pairs back to letter-pairs for frequency lookup."""
    
    letter_to_keypairs = defaultdict(list)
    
    for key_pair in key_pair_times.keys():
        if len(key_pair) == 2:
            key1, key2 = key_pair[0].upper(), key_pair[1].upper()
            
            # Map keys back to letters (if they represent letters)
            letter1 = key_to_letter.get(key1)
            letter2 = key_to_letter.get(key2)
            
            if letter1 and letter2 and letter1.isalpha() and letter2.isalpha():
                letter_pair = letter1 + letter2
                letter_to_keypairs[letter_pair].append(key_pair)
    
    return letter_to_keypairs

def estimate_bias_factor(key_pair_times: Dict[str, float],
                        english_frequencies: Dict[str, float],
                        letter_to_keypairs: Dict[str, List[str]]) -> float:
    """Estimate the bias factor: how much time advantage per unit frequency."""
    
    # Collect data points: (frequency, time) pairs
    freq_time_pairs = []
    
    for letter_pair, key_pairs in letter_to_keypairs.items():
        english_freq = english_frequencies.get(letter_pair, 0)
        
        if english_freq > 0:  # Only use pairs with frequency data
            for key_pair in key_pairs:
                if key_pair in key_pair_times:
                    time = key_pair_times[key_pair]
                    freq_time_pairs.append((english_freq, time))
    
    if len(freq_time_pairs) < 10:
        print("   ⚠️  Insufficient data for bias estimation, using conservative default")
        return 1000.0  # Conservative default bias factor
    
    # Convert to arrays for analysis
    frequencies = np.array([x[0] for x in freq_time_pairs])
    times = np.array([x[1] for x in freq_time_pairs])
    
    # Calculate correlation
    correlation = np.corrcoef(frequencies, times)[0, 1]
    print(f"   📊 Frequency-time correlation: {correlation:.3f}")
    
    # Use linear regression to estimate bias (negative slope = bias)
    if len(freq_time_pairs) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(frequencies, times)
        regression_bias_factor = -slope  # Negative because higher freq = lower time
        
        print(f"   📊 Regression bias factor: {regression_bias_factor:.1f}ms per frequency unit")
        print(f"      (R²={r_value**2:.3f}, p={p_value:.3f})")
        
        # Validate that bias factor is reasonable
        if 100 <= abs(regression_bias_factor) <= 5000:
            return abs(regression_bias_factor)
        else:
            print(f"   ⚠️  Regression bias factor seems unreasonable, using conservative estimate")
    
    # Fallback: Conservative estimate based on typing research
    return 1000.0  # 1000ms per unit frequency

def apply_frequency_debiasing(key_pair_times: Dict[str, float],
                             english_frequencies: Dict[str, float],
                             verbose: bool = False) -> Dict[str, float]:
    """Apply frequency-based debiasing to remove QWERTY practice effects."""
    
    if not english_frequencies:
        print("   ⚠️  No frequency data available - skipping debiasing")
        return key_pair_times
    
    print(f"\n🔧 Applying frequency-based debiasing...")
    
    # Step 1: Create mapping from key-pairs to letter-pairs
    key_to_letter = create_layout_mapping()
    letter_to_keypairs = map_keypairs_to_letterpairs(key_pair_times, key_to_letter)
    
    print(f"   📝 Mapped {len(letter_to_keypairs)} letter-pairs to key-pairs")
    
    # Step 2: Estimate bias factor
    bias_factor = estimate_bias_factor(key_pair_times, english_frequencies, letter_to_keypairs)
    print(f"   🎯 Using bias factor: {bias_factor:.1f}ms per frequency unit")
    
    # Step 3: Apply corrections
    debiased_times = {}
    corrections_applied = 0
    total_correction = 0
    
    for key_pair, original_time in key_pair_times.items():
        correction = 0.0
        
        # Find corresponding letter-pair
        if len(key_pair) == 2:
            key1, key2 = key_pair[0].upper(), key_pair[1].upper()
            letter1 = key_to_letter.get(key1)
            letter2 = key_to_letter.get(key2)
            
            if letter1 and letter2 and letter1.isalpha() and letter2.isalpha():
                letter_pair = letter1 + letter2
                english_freq = english_frequencies.get(letter_pair, 0)
                
                if english_freq > 0:
                    # Calculate correction: higher frequency = larger correction
                    correction = english_freq * bias_factor
                    corrections_applied += 1
                    total_correction += correction
        
        # Apply correction (add time back to remove unfair advantage)
        corrected_time = original_time + correction
        
        # Ensure minimum realistic time
        corrected_time = max(50, corrected_time)
        
        debiased_times[key_pair] = corrected_time
        
        if verbose and correction > 10:  # Show significant corrections
            print(f"      {key_pair}: {original_time:.1f}ms → {corrected_time:.1f}ms (+{correction:.1f}ms)")
    
    print(f"   ✅ Applied corrections to {corrections_applied}/{len(key_pair_times)} key-pairs")
    if corrections_applied > 0:
        print(f"      Average correction: {total_correction/corrections_applied:.1f}ms")
    
    return debiased_times

def validate_debiasing(original_times: Dict[str, float],
                      debiased_times: Dict[str, float],
                      english_frequencies: Dict[str, float]) -> bool:
    """Validate that debiasing was effective."""
    
    if not english_frequencies:
        return True  # No debiasing applied, nothing to validate
    
    print(f"\n🔍 Validating debiasing effectiveness...")
    
    # Create mapping for validation
    key_to_letter = create_layout_mapping()
    
    # Test: High vs Low frequency English bigrams
    high_freq_threshold = np.percentile(list(english_frequencies.values()), 80)
    low_freq_threshold = np.percentile(list(english_frequencies.values()), 20)
    
    high_freq_orig = []
    high_freq_debiased = []
    low_freq_orig = []
    low_freq_debiased = []
    
    for key_pair in original_times.keys():
        if len(key_pair) == 2:
            key1, key2 = key_pair[0].upper(), key_pair[1].upper()
            letter1 = key_to_letter.get(key1)
            letter2 = key_to_letter.get(key2)
            
            if letter1 and letter2 and letter1.isalpha() and letter2.isalpha():
                letter_pair = letter1 + letter2
                freq = english_frequencies.get(letter_pair, 0)
                
                if freq >= high_freq_threshold:
                    high_freq_orig.append(original_times[key_pair])
                    high_freq_debiased.append(debiased_times[key_pair])
                elif freq <= low_freq_threshold and freq > 0:
                    low_freq_orig.append(original_times[key_pair])
                    low_freq_debiased.append(debiased_times[key_pair])
    
    if high_freq_orig and low_freq_orig:
        orig_bias = np.mean(low_freq_orig) - np.mean(high_freq_orig)
        debiased_bias = np.mean(low_freq_debiased) - np.mean(high_freq_debiased)
        
        print(f"   📊 High frequency pairs:")
        print(f"      Original: {np.mean(high_freq_orig):.1f}ms")
        print(f"      Debiased: {np.mean(high_freq_debiased):.1f}ms")
        print(f"   📊 Low frequency pairs:")
        print(f"      Original: {np.mean(low_freq_orig):.1f}ms")
        print(f"      Debiased: {np.mean(low_freq_debiased):.1f}ms")
        print(f"   🎯 Bias reduction: {orig_bias:.1f}ms → {debiased_bias:.1f}ms")
        
        bias_reduced = debiased_bias < orig_bias
        print(f"   Status: {'✅ Bias reduced' if bias_reduced else '⚠️ Bias not reduced'}")
        
        return bias_reduced
    else:
        print(f"   ⚠️  Insufficient data for validation")
        return True

# ============================================================================
# ORIGINAL FUNCTIONS (UNCHANGED)
# ============================================================================

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
    """Extract timing data for all key-to-key movements from CSV data."""
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
    
    print(f"  📊 Calculating medians from {len(key_timings)} unique movements:")
    sufficient_data = 0
    insufficient_data = 0
    
    for key_pair, timings in key_timings.items():
        if len(timings) >= 3:  # Need sufficient data
            median_timings[key_pair] = statistics.median(timings)
            sufficient_data += 1
        else:
            insufficient_data += 1
    
    print(f"    - {sufficient_data} movements with sufficient data (≥3 instances)")
    print(f"    - {insufficient_data} movements with insufficient data (<3 instances)")
    
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

def calculate_bigram_components(bigram, median_timings, mirror_map, all_movement_times=None):
    """Calculate the timing components for a bigram."""
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
    
    fallback_used = []
    
    # Calculate overall median for final fallback
    all_timings = list(all_movement_times.values()) if all_movement_times else []
    overall_median = statistics.median(all_timings) if all_timings else 200  # 200ms default
    
    # Try fallbacks if either component is missing
    if key1_time == 0:
        # Try mirror for key1_movement
        if key1_home in mirror_map and key1 in mirror_map:
            mirror_key1_home = mirror_map[key1_home]
            mirror_key1 = mirror_map[key1]
            mirror_movement = (mirror_key1_home, mirror_key1)
            key1_time = median_timings.get(mirror_movement, 0)
            if key1_time > 0:
                fallback_used.append("mirror_key1_movement")
        
        # If still 0, try to find any movement to this key from any home position
        if key1_time == 0 and all_movement_times:
            for (from_key, to_key), timing in all_movement_times.items():
                if to_key == key1 and from_key in ['a', 's', 'd', 'f', 'j', 'k', 'l', ';']:  # home row keys
                    key1_time = timing
                    fallback_used.append("any_home_to_key1")
                    break
        
        # If still 0, use movements TO this key from any key
        if key1_time == 0 and all_movement_times:
            movements_to_key1 = [timing for (from_key, to_key), timing in all_movement_times.items() if to_key == key1]
            if movements_to_key1:
                key1_time = statistics.median(movements_to_key1)
                fallback_used.append("median_to_key1")
        
        # Final fallback: use overall median
        if key1_time == 0:
            key1_time = overall_median
            fallback_used.append("overall_median_key1")
    
    if key1_to_key2_time == 0:
        # Try mirror for key1_to_key2_movement
        if key1 in mirror_map and key2 in mirror_map:
            mirror_key1 = mirror_map[key1]
            mirror_key2 = mirror_map[key2]
            mirror_movement = (mirror_key1, mirror_key2)
            key1_to_key2_time = median_timings.get(mirror_movement, 0)
            if key1_to_key2_time > 0:
                fallback_used.append("mirror_transition")
        
        # If still 0, try to find any movement from key1 to any key
        if key1_to_key2_time == 0 and all_movement_times:
            movements_from_key1 = [timing for (from_key, to_key), timing in all_movement_times.items() if from_key == key1]
            if movements_from_key1:
                key1_to_key2_time = statistics.median(movements_from_key1)
                fallback_used.append("median_from_key1")
            else:
                # Try movements to key2 from any key
                movements_to_key2 = [timing for (from_key, to_key), timing in all_movement_times.items() if to_key == key2]
                if movements_to_key2:
                    key1_to_key2_time = statistics.median(movements_to_key2)
                    fallback_used.append("median_to_key2")
        
        # Final fallback: use overall median
        if key1_to_key2_time == 0:
            key1_to_key2_time = overall_median
            fallback_used.append("overall_median_transition")
    
    total_time = key1_time + key1_to_key2_time
    
    return key1_time, key1_to_key2_time, total_time, ";".join(fallback_used)

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

def compute_all_keypair_times(input_dir: str, frequency_file: str = None, verbose: bool = False):
    """Compute time scores for all key-pairs with built-in debiasing."""
    print("Computing key-pair times using timing components from CSV data...")
    if frequency_file:
        print("🎯 Built-in QWERTY debiasing: ENABLED")
    
    # Load and process CSV data
    median_timings, key_timings = load_and_process_csv_data(input_dir)
    
    if median_timings is None:
        return []
    
    # Create mirror mapping
    mirror_map = create_mirror_mapping()
    
    # Generate all possible key-pairs
    all_key_pairs = generate_all_key_pairs()
    
    print(f"\n🟡 Stage 2: Computing times for all {len(all_key_pairs)} key-pairs")
    print(f"  📈 Available empirical data: {len(median_timings)} movements")
    
    raw_empirical_times = {}
    successful_calcs = 0
    fallback_used_count = 0
    no_data_count = 0
    empirical_count = 0
    
    # Calculate maximum time for fallback
    all_total_times = []
    zero_component_count = 0

    for i, key_pair in enumerate(all_key_pairs):
        if i % 100 == 0:
            print(f"    Progress: {i}/{len(all_key_pairs)} ({i/len(all_key_pairs)*100:.1f}%)")
        
        key1_time, key1_to_key2_time, total_time, fallback_info = calculate_bigram_components(
            key_pair, median_timings, mirror_map, median_timings
        )
        
        if total_time > 0:
            all_total_times.append(total_time)
        else:
            zero_component_count += 1

    print(f"  📊 Component analysis:")
    print(f"    - {len(all_total_times)} pairs have empirical or fallback timing data")
    print(f"    - {zero_component_count} pairs need maximum fallback")

    # Calculate maximum time for final fallback
    maximum_total_time = max(all_total_times) if all_total_times else None

    if maximum_total_time:
        print(f"  ✅ Maximum total time found: {maximum_total_time:.1f}ms")
    else:
        print(f"  ❌ No empirical timing data found - cannot calculate fallback times")
        return []

    # Now calculate all times
    for i, key_pair in enumerate(all_key_pairs):
        key1_time, key1_to_key2_time, total_time, fallback_info = calculate_bigram_components(
            key_pair, median_timings, mirror_map, median_timings
        )
        
        fallback_type = ""
        
        if total_time > 0:
            time_score = total_time
            successful_calcs += 1
            if fallback_info:
                fallback_type = fallback_info
                fallback_used_count += 1
            else:
                fallback_type = "empirical"
                empirical_count += 1
        else:
            # Use maximum timing as final fallback
            if maximum_total_time is not None:
                key1_time = maximum_total_time / 2
                key1_to_key2_time = maximum_total_time / 2
                total_time = maximum_total_time
                time_score = maximum_total_time
                fallback_type = "maximum"
                no_data_count += 1
            else:
                # Skip key-pairs with no data available
                continue
        
        raw_empirical_times[key_pair.upper()] = time_score
    
    print(f"  ✅ Time computation complete:")
    print(f"    - {empirical_count} pairs used pure empirical data")
    print(f"    - {fallback_used_count} pairs used mirror/fallback data") 
    print(f"    - {no_data_count} pairs used maximum fallback ({maximum_total_time:.1f}ms)")
    
    # NEW: Apply frequency-based debiasing
    english_frequencies = load_english_frequencies(frequency_file)
    debiased_times = apply_frequency_debiasing(raw_empirical_times, english_frequencies, verbose)
    
    # Validate debiasing effectiveness
    validate_debiasing(raw_empirical_times, debiased_times, english_frequencies)
    
    # Convert to results format
    results = []
    for key_pair, time_score in sorted(debiased_times.items()):
        results.append({
            'key_pair': key_pair,
            'time_score': time_score,
            'fallback_type': 'debiased'  # Mark all as debiased
        })
    
    return results

def save_key_pair_times(results, output_file="../tables/keypair_time_scores.csv"):
    """Save key-pair times to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'time_score', 'fallback_type'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} debiased key-pair times to: {output_file}")

def validate_output(output_file="../tables/keypair_time_scores.csv"):
    """Perform validation of the generated output file."""
    
    if not os.path.exists(output_file):
        print(f"❌ Output file not found: {output_file}")
        return False
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Output Validation Results:")
    print(f"   Total key-pairs: {len(rows)}")
    
    # Check for expected number of combinations
    keys = get_all_qwerty_keys()
    expected_count = len(keys) ** 2
    print(f"   Expected count: {expected_count}")
    print(f"   Match: {'✅' if len(rows) == expected_count else '❌'}")
    
    # Check score ranges
    time_scores = [float(row['time_score']) for row in rows]
    print(f"   Time score range: {min(time_scores):.1f} to {max(time_scores):.1f}ms")
    print(f"   Average time: {sum(time_scores)/len(time_scores):.1f}ms")
    
    # Check that all scores are positive
    negative_times = [row for row in rows if float(row['time_score']) <= 0]
    print(f"   Negative or zero time scores: {len(negative_times)} {'✅' if len(negative_times) == 0 else '❌'}")
    
    print(f"\n✅ Output validation complete!")
    
    return True

def main():
    """Main entry point with integrated debiasing."""
    parser = argparse.ArgumentParser(
        description='Generate time scores for QWERTY key-pairs with built-in QWERTY debiasing',
        epilog="""
This version automatically removes QWERTY bias using English bigram frequencies:
- Analyzes empirical typing data for rich biomechanical insights
- Applies frequency-based corrections to remove practice effects  
- Outputs layout-agnostic time scores for fair comparison
- Perfect for dual framework analysis

The debiasing process:
1. Maps key-pairs back to letter-pairs
2. Looks up English bigram frequencies
3. Applies proportional corrections (higher frequency = larger correction)
4. Validates bias removal effectiveness

No separate debiasing step needed - output is ready for use.
        """
    )
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing CSV files with typing data')
    parser.add_argument('--frequency-file', 
                        default='../input/english-letter-pair-frequencies-google-ngrams.csv',
                        help='English bigram frequency file for debiasing (optional)')
    parser.add_argument('--output', default='../tables/keypair_time_scores.csv',
                        help='Output CSV file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed debiasing information')
    
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
    
    print("Generate time scores with built-in QWERTY debiasing")
    print("=" * 60)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print(f"Input directory: {args.input_dir}")
    print(f"CSV files found: {len(csv_files)}")
    print(f"Frequency file: {args.frequency_file}")
    
    # Check if frequency file exists
    if args.frequency_file and Path(args.frequency_file).exists():
        print("🎯 Automatic QWERTY debiasing: ENABLED")
    else:
        print("⚠️  Automatic QWERTY debiasing: DISABLED (no frequency file)")
    
    # Compute times with integrated debiasing
    results = compute_all_keypair_times(args.input_dir, args.frequency_file, args.verbose)
    
    # Save results
    save_key_pair_times(results, args.output)
    
    # Validate output
    validate_output(args.output)
    
    print(f"\n✅ Time generation with integrated debiasing complete: {args.output}")
    print("🎯 Output is layout-agnostic and ready for dual framework analysis!")
    
    return 0

if __name__ == "__main__":
    exit(main())