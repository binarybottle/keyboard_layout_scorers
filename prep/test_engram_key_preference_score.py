#!/usr/bin/env python3
"""
Diagnostic script to debug engram_key_preference scoring for letters IEA → qwerty keys SDF vs FDS.

This script traces through the exact calculation to understand why FDS key order scores
higher than key order SDF for letters IEA, despite 'a' being mapped to a worse key in FDS.

While EA strongly favors SDF (-0.07775904), the combination of IE + EI + II favoring FDS more than compensates:

IE: +0.05808823 (favors FDS)
EI: +0.04004813 (favors FDS)
II: +0.02799537 (favors FDS)
Total FDS advantage: +0.13423173
SDF advantage from EA: -0.07775904
Net result: FDS wins by +0.05647269

Why This Happens
Your intuition about 'a' being more frequent was correct, but you were thinking about single letter frequencies. The scoring uses bigram frequencies, and the bigrams involving 'i' are quite frequent:

EA: 0.43931662 (most frequent)
IE: 0.32818211 (second most frequent)
AI: 0.29757332
IA: 0.28298431
EI: 0.22626061

When you move 'i' from S (0.646) to F (1.000) in the FDS layout:

IE improves from SD (0.758) to FD (0.935) → big gain on a frequent bigram
EI improves from DS (0.758) to DF (0.935) → big gain on a frequent bigram
II improves from SS (0.646) to FF (1.000) → moderate gain

Meanwhile, moving 'a' from F to S only hurts EA, though EA is the most frequent bigram.
The Math

FDS gains more from upgrading 'i' across multiple frequent bigrams (IE, EI, II)
SDF gains more from upgrading 'a' on the single most frequent bigram (EA)
The distributed gains from 'i' outweigh the concentrated loss from 'a'

"""

import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Key preference tier values from the Engram scoring system
TIER_VALUES = {
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

def load_bigram_frequencies(filepath: str) -> Dict[str, float]:
    """Load bigram frequency data from CSV file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Frequency file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Find the bigram and frequency columns
    bigram_col = None
    freq_col = None
    
    for col in ['bigram', 'pair', 'key_pair', 'letter_pair', 'item', 'item_pair']:
        if col in df.columns:
            bigram_col = col
            break
    
    for col in ['normalized_frequency', 'frequency', 'count', 'score']:
        if col in df.columns:
            freq_col = col
            break
    
    if not bigram_col or not freq_col:
        raise ValueError("Could not find required columns in frequency file")
    
    frequencies = {}
    for _, row in df.iterrows():
        bigram = str(row[bigram_col]).strip().upper()
        freq = float(row[freq_col])
        
        if len(bigram) == 2:
            frequencies[bigram] = freq
    
    return frequencies

def compute_key_pair_score(key1: str, key2: str) -> float:
    """Compute key preference score for a key pair."""
    score1 = TIER_VALUES.get(key1.upper(), 0.0)
    score2 = TIER_VALUES.get(key2.upper(), 0.0)
    return (score1 + score2) / 2.0

def create_layout_mapping(letters: str, positions: str) -> Dict[str, str]:
    """Create mapping from letters to keys."""
    mapping = {}
    for letter, position in zip(letters.upper(), positions.upper()):
        mapping[letter] = position
    return mapping

def generate_all_bigrams(letters: str) -> List[str]:
    """Generate all possible bigrams from a set of letters."""
    letters = letters.upper()
    bigrams = []
    for l1 in letters:
        for l2 in letters:
            bigrams.append(l1 + l2)
    return bigrams

def debug_key_preference_scoring(letters: str, positions: str, layout_name: str, 
                                bigram_frequencies: Dict[str, float]) -> Tuple[float, List[Dict]]:
    """Debug the key preference scoring for a specific layout."""
    
    layout_mapping = create_layout_mapping(letters, positions)
    print(f"\n=== {layout_name} ===")
    print(f"Layout mapping: {layout_mapping}")
    
    # Generate all possible bigrams from the letters
    all_bigrams = generate_all_bigrams(letters)
    
    # Filter to only valid bigrams that exist in frequency data
    valid_bigrams = [bg for bg in all_bigrams if bg in bigram_frequencies]
    
    print(f"Valid bigrams found in frequency data: {len(valid_bigrams)} out of {len(all_bigrams)}")
    
    # Calculate score for each bigram
    bigram_details = []
    total_weighted_score = 0.0
    total_frequency = 0.0
    
    print(f"\nBigram breakdown:")
    print(f"{'Bigram':<8} {'Keys':<8} {'Key Score':<10} {'Frequency':<12} {'Weighted':<12}")
    print("-" * 70)
    
    for bigram in sorted(valid_bigrams):
        letter1, letter2 = bigram[0], bigram[1]
        key1 = layout_mapping[letter1]
        key2 = layout_mapping[letter2]
        key_pair = key1 + key2
        
        # Get key preference score for this key pair
        key_score = compute_key_pair_score(key1, key2)
        
        # Get frequency for this bigram
        frequency = bigram_frequencies[bigram]
        
        # Calculate weighted contribution
        weighted_contribution = key_score * frequency
        
        total_weighted_score += weighted_contribution
        total_frequency += frequency
        
        bigram_details.append({
            'bigram': bigram,
            'key_pair': key_pair,
            'key_score': key_score,
            'frequency': frequency,
            'weighted_contribution': weighted_contribution
        })
        
        print(f"{bigram:<8} {key_pair:<8} {key_score:<10.6f} {frequency:<12.8f} {weighted_contribution:<12.8f}")
    
    # Calculate final weighted average
    final_score = total_weighted_score / total_frequency if total_frequency > 0 else 0.0
    
    print(f"\nSummary:")
    print(f"Total weighted score: {total_weighted_score:.8f}")
    print(f"Total frequency: {total_frequency:.8f}")
    print(f"Final weighted average: {final_score:.6f}")
    
    return final_score, bigram_details

def compare_layouts(bigram_frequencies: Dict[str, float]):
    """Compare the two layouts and explain the difference."""
    
    # Layout 1: IEA → SDF (i→S, e→D, a→F)
    sdf_score, sdf_details = debug_key_preference_scoring(
        "IEA", "SDF", "IEA → SDF", bigram_frequencies
    )
    
    # Layout 2: IEA → FDS (i→F, e→D, a→S)  
    fds_score, fds_details = debug_key_preference_scoring(
        "IEA", "FDS", "IEA → FDS", bigram_frequencies
    )
    
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"SDF Score: {sdf_score:.6f}")
    print(f"FDS Score: {fds_score:.6f}")
    print(f"Difference: {fds_score - sdf_score:.6f} (FDS is {'higher' if fds_score > sdf_score else 'lower'})")
    
    # Find the biggest contributors to the difference
    print(f"\nBigram-by-bigram comparison:")
    print(f"{'Bigram':<8} {'SDF Keys':<10} {'SDF Score':<12} {'FDS Keys':<10} {'FDS Score':<12} {'Diff':<12}")
    print("-" * 90)
    
    sdf_dict = {d['bigram']: d for d in sdf_details}
    fds_dict = {d['bigram']: d for d in fds_details}
    
    differences = []
    for bigram in sorted(sdf_dict.keys()):
        sdf_data = sdf_dict[bigram]
        fds_data = fds_dict[bigram]
        
        diff = fds_data['weighted_contribution'] - sdf_data['weighted_contribution']
        differences.append((bigram, diff))
        
        print(f"{bigram:<8} {sdf_data['key_pair']:<10} {sdf_data['weighted_contribution']:<12.8f} "
              f"{fds_data['key_pair']:<10} {fds_data['weighted_contribution']:<12.8f} {diff:<12.8f}")
    
    # Show the biggest contributors to the difference
    differences.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\nBiggest contributors to the difference:")
    for bigram, diff in differences[:5]:
        direction = "favors FDS" if diff > 0 else "favors SDF"
        print(f"  {bigram}: {diff:.8f} ({direction})")

def main():
    """Main entry point."""
    print("Debugging engram_key_preference scoring for IEA → SDF vs FDS")
    print("="*80)
    
    # Load bigram frequencies
    frequency_file = "../optimize_layouts/input/frequency/english-letter-pair-counts-google-ngrams_normalized.csv"
    
    try:
        bigram_frequencies = load_bigram_frequencies(frequency_file)
        print(f"Loaded {len(bigram_frequencies)} bigram frequencies from {frequency_file}")
        
        # Show total frequency mass to verify data
        total_freq = sum(bigram_frequencies.values())
        print(f"Total frequency mass: {total_freq:.6f}")
        
        # Show some sample bigrams for verification
        print(f"\nSample bigram frequencies:")
        sample_bigrams = ['AA', 'EE', 'II', 'AE', 'EA', 'IE', 'EI', 'AI', 'IA']
        for bg in sample_bigrams:
            freq = bigram_frequencies.get(bg, 0.0)
            print(f"  {bg}: {freq:.8f}")
        
        # Run the comparison
        compare_layouts(bigram_frequencies)
        
    except FileNotFoundError:
        print(f"Error: Could not find frequency file at {frequency_file}")
        print("Please check the path and ensure the file exists.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()