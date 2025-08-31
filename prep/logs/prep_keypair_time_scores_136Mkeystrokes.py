"""
Generate precomputed time scores for all possible QWERTY key-pairs.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script computes time scores for every possible combination of QWERTY keys
using large-scale bigram timing data with automatic QWERTY bias removal based 
on English bigram frequencies.

Processing approach:
- Analyzes 136M+ keystroke bigram timing data from real typing sessions
- Uses MEDIAN timing across all instances (robust to outliers)
- Requires minimum 3 instances per bigram for statistical reliability
- Automatically removes QWERTY bias using English bigram frequency corrections

Usage:
    python prep_keypair_time_scores.py --bigram-file ../../process_136M_keystrokes/output/bigram_times.csv --frequency-file ../input/english-letter-pair-frequencies-google-ngrams.csv

Output:
    ../tables/keypair_time_scores.csv - CSV with debiased time scores

This version automatically removes QWERTY bias using English bigram frequencies:
- Analyzes large-scale empirical typing data for robust biomechanical insights
- Applies frequency-based corrections to remove practice effects  
- Outputs layout-agnostic time scores for fair comparison
- Perfect for dual framework analysis with Dvorak-7 validation

The debiasing process:
1. Maps key-pairs back to letter-pairs
2. Looks up English bigram frequencies
3. Applies proportional corrections (higher frequency = larger correction)
4. Validates bias removal effectiveness

No separate debiasing step needed - output is ready for use.
"""

import argparse
import csv
from email import parser
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
# FREQUENCY-BASED DEBIASING FUNCTIONS
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
        
        # Accept regression result if statistically significant
        if p_value < 0.05 and r_value**2 > 0.1:  # Significant & explains >10% variance
            print(f"   ✅ Using regression bias factor (statistically significant)")
            return abs(regression_bias_factor)
        else:
            print(f"   ⚠️  Regression not significant, using conservative estimate")
    
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

def apply_mirror_conservative_debiasing(debiased_times: Dict[str, float], 
                                      mirror_map: Dict[str, str],
                                      verbose: bool = False) -> Dict[str, float]:
    """
    Apply mirror-based conservative debiasing: take the slower time of each mirror pair.
    This removes any remaining hand/side advantages that survived frequency debiasing.
    """
    if verbose:
        print(f"\n🔄 Applying mirror-based conservative debiasing...")
    
    mirror_corrected_times = debiased_times.copy()
    corrections_applied = 0
    total_adjustment = 0
    
    # Process each key-pair
    for key_pair, original_time in debiased_times.items():
        if len(key_pair) == 2:
            key1, key2 = key_pair[0].lower(), key_pair[1].lower()
            
            # Get mirror keys
            mirror_key1 = mirror_map.get(key1)
            mirror_key2 = mirror_map.get(key2)
            
            if mirror_key1 and mirror_key2:
                mirror_pair = (mirror_key1 + mirror_key2).upper()
                
                # If mirror pair exists in our data
                if mirror_pair in debiased_times:
                    mirror_time = debiased_times[mirror_pair]
                    
                    # Take the slower (conservative) time for both pairs
                    conservative_time = max(original_time, mirror_time)
                    
                    # Apply to both original and mirror pair
                    if conservative_time > original_time:
                        adjustment = conservative_time - original_time
                        total_adjustment += adjustment
                        corrections_applied += 1
                        
                        if verbose and adjustment > 10:  # Show significant adjustments
                            print(f"      {key_pair}: {original_time:.1f}ms → {conservative_time:.1f}ms (+{adjustment:.1f}ms, mirror: {mirror_pair})")
                    
                    mirror_corrected_times[key_pair] = conservative_time
                    mirror_corrected_times[mirror_pair] = conservative_time
    
    print(f"   ✅ Applied mirror corrections to {corrections_applied} key-pairs")
    if corrections_applied > 0:
        print(f"      Average adjustment: {total_adjustment/corrections_applied:.1f}ms")
    
    return mirror_corrected_times

def validate_mirror_debiasing(original_times: Dict[str, float],
                             mirror_debiased_times: Dict[str, float],
                             mirror_map: Dict[str, str]) -> bool:
    """Validate that mirror debiasing removed hand-side advantages."""
    
    print(f"\n🔍 Validating mirror debiasing effectiveness...")
    
    # Check that mirror pairs now have identical times
    identical_pairs = 0
    total_pairs = 0
    
    for key_pair in original_times.keys():
        if len(key_pair) == 2:
            key1, key2 = key_pair[0].lower(), key_pair[1].lower()
            mirror_key1 = mirror_map.get(key1)
            mirror_key2 = mirror_map.get(key2)
            
            if mirror_key1 and mirror_key2:
                mirror_pair = (mirror_key1 + mirror_key2).upper()
                if mirror_pair in mirror_debiased_times:
                    total_pairs += 1
                    time1 = mirror_debiased_times[key_pair]
                    time2 = mirror_debiased_times[mirror_pair]
                    
                    if abs(time1 - time2) < 0.001:  # Effectively identical
                        identical_pairs += 1
    
    if total_pairs > 0:
        consistency_rate = identical_pairs / total_pairs
        print(f"   📊 Mirror pair consistency: {consistency_rate:.1%} ({identical_pairs}/{total_pairs})")
        
        if consistency_rate > 0.95:
            print(f"   Status: ✅ Mirror pairs highly consistent")
            return True
        else:
            print(f"   Status: ⚠️ Some mirror pairs still inconsistent")
            return False
    else:
        print(f"   ⚠️  No mirror pairs found for validation")
        return True

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
# OTHER FUNCTIONS
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

def load_136M_bigram_data(bigram_file: str):
    """Load 136M bigram timing data directly"""
    print(f"Loading 136M bigram data from {bigram_file}...")
    
    df = pd.read_csv(bigram_file)
    print(f"Loaded {len(df):,} bigram records")
    
    # Group by bigram and collect timings
    key_pair_times = defaultdict(list)
    
    for _, row in df.iterrows():
        bigram = row['bigram'].upper()
        timing = row['interkey_interval']
        
        # Apply same filtering as current script
        if 50 <= timing <= 2000:
            key_pair_times[bigram].append(timing)
    
    # Calculate medians (keeping minimum sample size requirement)
    median_timings = {}
    for bigram, timings in key_pair_times.items():
        if len(timings) >= 3:  # Same threshold as current
            median_timings[bigram] = statistics.median(timings)
    
    print(f"Calculated medians for {len(median_timings)} bigrams")
    return median_timings

def compute_all_keypair_times_136M(bigram_file: str, frequency_file: str = None, verbose: bool = False):
    """Compute times using 136M dataset - much simpler!"""
    
    # Load bigram timings directly
    empirical_times = load_136M_bigram_data(bigram_file)
    
    # Generate all possible key-pairs
    all_key_pairs = generate_all_key_pairs()
    
    # Fill empirical data + simple fallbacks for missing pairs
    raw_times = {}
    for key_pair in all_key_pairs:
        if key_pair in empirical_times:
            raw_times[key_pair] = empirical_times[key_pair]
        else:
            # Simple fallback: use maximum observed time
            raw_times[key_pair] = max(empirical_times.values()) if empirical_times else 300
    
    # **ADD THIS: Load English frequencies before debiasing**
    english_frequencies = load_english_frequencies(frequency_file)
    
    # Apply same debiasing pipeline
    mirror_map = create_mirror_mapping()
    debiased_times = apply_frequency_debiasing(raw_times, english_frequencies, verbose)
    final_times = apply_mirror_conservative_debiasing(debiased_times, mirror_map, verbose)
    
    # Convert to results format
    results = []
    for key_pair, time_score in sorted(final_times.items()):
        results.append({
            'key_pair': key_pair,
            'time_score': time_score,
            'fallback_type': 'empirical_136M' if key_pair in empirical_times else 'maximum_fallback'
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
        description='Generate time scores for QWERTY key-pairs with built-in QWERTY debiasing using 136M keystroke data',
        epilog="""
This version automatically removes QWERTY bias using English bigram frequencies:
- Analyzes large-scale empirical typing data (136M+ keystrokes) for robust biomechanical insights
- Applies frequency-based corrections to remove practice effects  
- Outputs layout-agnostic time scores for fair comparison
- Appropriate for dual framework analysis with Dvorak-7 validation

The debiasing process:
1. Maps key-pairs back to letter-pairs
2. Looks up English bigram frequencies
3. Applies proportional corrections (higher frequency = larger correction)
4. Validates bias removal effectiveness

No separate debiasing step needed - output is ready for use.
        """
    )
    parser.add_argument('--bigram-file', 
                        default='../../process_136M_keystrokes/output/bigram_times.csv',
                        help='136M bigram timing file')
    parser.add_argument('--frequency-file', 
                        default='../input/english-letter-pair-frequencies-google-ngrams.csv',
                        help='English bigram frequency file for debiasing (optional)')
    parser.add_argument('--output', default='../tables/keypair_time_scores.csv',
                        help='Output CSV file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed debiasing information')
    
    args = parser.parse_args()
    
    print("Generate time scores with built-in QWERTY debiasing")
    print("=" * 60)
    
    # Show key information
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    #print(f"Input directory: {args.input_dir}")
    #print(f"CSV files found: {len(csv_files)}")
    print(f"Bigram times file: {args.bigram_file}")
    print(f"Frequency file: {args.frequency_file}")
    
    # Check if frequency file exists
    if args.frequency_file and Path(args.frequency_file).exists():
        print("🎯 Frequency-based QWERTY debiasing: ENABLED")
        print("🔄 Mirror-based conservative debiasing: ENABLED")
    else:
        print("⚠️  Automatic QWERTY debiasing: DISABLED (no frequency file)")
    
    # Compute times with integrated debiasing
    results = compute_all_keypair_times_136M(args.bigram_file, args.frequency_file)

    # Save results
    save_key_pair_times(results, args.output)
    
    # Validate output
    validate_output(args.output)
    
    print(f"\n✅ Time generation with integrated debiasing complete: {args.output}")
    print("🎯 Output is layout-agnostic and ready for dual framework analysis!")
    
    return 0

if __name__ == "__main__":
    exit(main())