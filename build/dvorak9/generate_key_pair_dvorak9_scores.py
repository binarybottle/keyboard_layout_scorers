#!/usr/bin/env python3
"""
Generate precomputed Dvorak-9 scores for all possible QWERTY key-pairs.

This script computes the unweighted average Dvorak-9 score for every possible
combination of QWERTY keys and saves them to input/dvorak9/key_pair_scores.csv.

The output file contains all possible key-pairs (e.g., "QW", "QE", "AS") with
their corresponding Dvorak-9 scores (average of the 9 individual criteria).

This precomputation allows the main scorer to simply look up scores rather
than computing them on-demand, making layout scoring much faster.

Usage:
    python generate_key_pair_scores.py

Output:
    input/dvorak9/key_pair_scores.csv - CSV with columns: key_pair, dvorak9_score
"""

import csv
import os
from pathlib import Path

# Import necessary functions and constants from the scorer script
from dvorak9_scorer import (
    QWERTY_LAYOUT,
    score_bigram_dvorak9
)

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing."""
    # Use the same key set as the original generator (excludes numbers)
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
    """Compute Dvorak-9 scores for all key-pairs."""
    key_pairs = generate_all_key_pairs()
    results = []
    
    print(f"Computing Dvorak-9 scores for {len(key_pairs)} key-pairs...")
    
    for i, key_pair in enumerate(key_pairs):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(key_pairs)} ({i/len(key_pairs)*100:.1f}%)")
        
        # Compute individual Dvorak-9 criteria scores using the scorer's function
        bigram_scores = score_bigram_dvorak9(key_pair)
        
        # Calculate unweighted average (baseline Dvorak-9 score)
        dvorak9_score = sum(bigram_scores.values()) / len(bigram_scores)
        
        results.append({
            'key_pair': key_pair,
            'dvorak9_score': dvorak9_score
        })
    
    return results

def save_key_pair_scores(results, output_file="input/dvorak9/key_pair_scores.csv"):
    """Save key-pair scores to CSV file."""
    
    # Create input directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by key-pair for consistent ordering
    results.sort(key=lambda x: x['key_pair'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'dvorak9_score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Saved {len(results)} key-pair scores to: {output_file}")

def validate_output(output_file="input/dvorak9/key_pair_scores.csv"):
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
    
    # Check score range (should be 0-1)
    scores = [float(row['dvorak9_score']) for row in rows]
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    print(f"   Score range: {min_score:.3f} to {max_score:.3f}")
    print(f"   Average score: {avg_score:.3f}")
    print(f"   Valid range (0-1): {'✅' if 0 <= min_score and max_score <= 1 else '❌'}")
    
    # Show some examples
    print(f"\n📝 Sample key-pairs and scores:")
    for i in range(0, min(10, len(rows)), max(1, len(rows)//10)):
        row = rows[i]
        print(f"   {row['key_pair']}: {float(row['dvorak9_score']):.3f}")
    
    return True

def main():
    """Main entry point."""
    print("Dvorak-9 key-pair Score Generator")
    print("=" * 50)
    
    # Load QWERTY keys to show what we're working with
    keys = get_all_qwerty_keys()
    print(f"QWERTY keys ({len(keys)}): {''.join(sorted(keys))}")
    print(f"Total key-pairs to compute: {len(keys)**2}")
    print()
    
    # Compute scores
    results = compute_key_pair_scores()
    
    # Save results
    output_file = "input/dvorak9/key_pair_scores.csv"
    save_key_pair_scores(results, output_file)
    
    # Validate output
    validate_output(output_file)
    
    print(f"\n✅ key-pair score generation complete!")
    print(f"   Output file: {output_file}")
    print(f"   Ready for use with dvorak9_scorer.py")

if __name__ == "__main__":
    main()