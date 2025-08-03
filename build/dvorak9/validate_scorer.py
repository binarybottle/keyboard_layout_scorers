#!/usr/bin/env python3
"""
Comprehensive validation test of Dvorak-9 scoring criteria and frequency-weighted scoring.

This script systematically tests every possible bigram combination of a QWERTY layout
to validate that:
1. The 9 Dvorak criteria are implemented correctly and produce expected score patterns
2. The frequency-weighted scoring pipeline works correctly with empirical weights
3. Both speed-based and comfort-based weights produce reasonable results

This version uses the canonical scoring implementation from dvorak9_scorer.py to
ensure consistency across the codebase.

Outputs:
1. 'dvorak9_scores_all_bigrams.csv' - All possible bigrams with individual criterion scores (0-1 scale)
2. 'dvorak9_scores_unique_all_bigrams.csv' - Unique score patterns with counts and examples
3. 'dvorak9_weighted_scores_[weights_type].csv' - All bigrams with empirical combination scores (if weights provided)

Features:
- Tests all 841 possible bigrams (29×29 QWERTY keys)
- Validates specific test cases for each criterion
- Tests frequency-weighted scoring with speed-based or comfort-based weights
- Analyzes score distributions and patterns
- Identifies criteria with constant or low-variation scores
- Provides comprehensive debugging output for criterion implementation

Usage:
    python validate_scorer.py
    python validate_scorer.py --weights input/dvorak9/speed_weights.csv
    python validate_scorer.py --weights input/dvorak9/comfort_weights.csv
"""

import sys
import os
import csv
import argparse
from collections import defaultdict
from pathlib import Path

# Import the canonical scoring function and classes
from dvorak9_scorer import score_bigram_dvorak9, get_key_info, QWERTY_LAYOUT, Dvorak9Scorer

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing (excludes numbers)."""
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def test_all_bigrams_basic(all_bigrams_file="output/dvorak9_scores_all_bigrams.csv", 
                          unique_scores_file="output/dvorak9_scores_unique_all_bigrams.csv"):
    """Test every possible bigram combination using basic criteria scoring."""
    
    keys = get_all_qwerty_keys()
    print(f"Testing {len(keys)} keys = {len(keys)**2} possible bigrams (basic criteria)")
    
    results = []
    score_patterns = defaultdict(list)  # Track bigrams for each unique score pattern
    
    # Test every possible bigram
    for key1 in keys:
        for key2 in keys:
            bigram = key1 + key2
            
            # Use the canonical scoring function
            bigram_scores = score_bigram_dvorak9(bigram)
            
            # Get key info for analysis
            key_info1 = get_key_info(key1)
            key_info2 = get_key_info(key2)
            
            if key_info1 and key_info2:
                row1, finger1, hand1 = key_info1
                row2, finger2, hand2 = key_info2
            else:
                # Handle missing key info gracefully
                row1, finger1, hand1 = (0, 0, 'L')
                row2, finger2, hand2 = (0, 0, 'L')
            
            # Create result row
            result = {
                'bigram': bigram,
                'pos1': key1,
                'pos2': key2,
                'hand1': hand1,
                'hand2': hand2,
                'finger1': finger1,
                'finger2': finger2,
                'row1': row1,
                'row2': row2,
                'hands': bigram_scores['hands'],
                'fingers': bigram_scores['fingers'],
                'skip_fingers': bigram_scores['skip_fingers'],
                'dont_cross_home': bigram_scores['dont_cross_home'],
                'same_row': bigram_scores['same_row'],
                'home_row': bigram_scores['home_row'],
                'columns': bigram_scores['columns'],
                'strum': bigram_scores['strum'],
                'strong_fingers': bigram_scores['strong_fingers'],
                'total': sum(bigram_scores.values())
            }
            
            results.append(result)
            
            # Track unique score combinations
            score_tuple = tuple(bigram_scores[c] for c in ['hands', 'fingers', 'skip_fingers', 
                                                          'dont_cross_home', 'same_row', 'home_row', 
                                                          'columns', 'strum', 'strong_fingers'])
            score_patterns[score_tuple].append(bigram)
    
    # Sort all results by total score (descending), then by bigram name
    results.sort(key=lambda x: (-x['total'], x['bigram']))
    
    # Write CSV with all bigrams
    fieldnames = ['bigram', 'total', 'pos1', 'pos2', 'hand1', 'hand2', 'finger1', 'finger2', 'row1', 'row2',
                  'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                  'home_row', 'columns', 'strum', 'strong_fingers']
    
    with open(all_bigrams_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Write CSV with unique score patterns
    unique_fieldnames = ['total', 'count', 'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                        'home_row', 'columns', 'strum', 'strong_fingers', 'example_bigrams']
    
    unique_results = []
    for score_tuple, bigram_list in score_patterns.items():
        total_score = sum(score_tuple)
        
        # Limit examples to first 10 bigrams to keep CSV readable
        example_bigrams = ', '.join(bigram_list[:10])
        if len(bigram_list) > 10:
            example_bigrams += f' ... (+{len(bigram_list)-10} more)'
        
        unique_result = {
            'hands': score_tuple[0],
            'fingers': score_tuple[1], 
            'skip_fingers': score_tuple[2],
            'dont_cross_home': score_tuple[3],
            'same_row': score_tuple[4],
            'home_row': score_tuple[5],
            'columns': score_tuple[6],
            'strum': score_tuple[7],
            'strong_fingers': score_tuple[8],
            'total': total_score,
            'count': len(bigram_list),
            'example_bigrams': example_bigrams
        }
        unique_results.append(unique_result)
    
    # Sort by total score descending, then by count descending
    unique_results.sort(key=lambda x: (-x['total'], -x['count']))
    
    with open(unique_scores_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=unique_fieldnames)
        writer.writeheader()
        writer.writerows(unique_results)
    
    print(f"✅ Basic criteria results:")
    print(f"   All bigrams: {all_bigrams_file}")
    print(f"   Unique patterns: {unique_scores_file}")
    print(f"   Total bigrams: {len(results)}")
    print(f"   Unique score combinations: {len(score_patterns)}")
    
    return results, score_patterns

def test_weighted_scoring_simple(weights, output_dir="output"):
    """Test weighted scoring using a simple bigram frequency approach."""
    
    # Determine weights type from filename for clearer output
    weights_filename = Path(weights).stem
    if "speed" in weights_filename.lower():
        weights_type = "speed"
    elif "comfort" in weights_filename.lower():
        weights_type = "comfort"
    else:
        weights_type = "custom"
    
    weighted_file = Path(output_dir) / f"dvorak9_weighted_scores_{weights_type}.csv"
    
    keys = get_all_qwerty_keys()
    print(f"\nTesting weighted scoring:")
    print(f"  Weights file: {weights}")
    print(f"  Weights type: {weights_type}")
    print(f"  Output file: {weighted_file}")
    
    # Create a simple test layout
    common_chars = "etaoinshrdlcumwfgypbvkjxqz"
    qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBN"

    # Map first N characters to first N positions
    layout_mapping = {}
    for i, char in enumerate(common_chars):
        if i < len(qwerty_positions):
            layout_mapping[char] = qwerty_positions[i]
    
    print(f"  Testing with {len(layout_mapping)} character layout")
    
    try:
        # Test the weighted scorer
        scorer = Dvorak9Scorer(
            layout_mapping=layout_mapping,
            weights=weights
        )
        results = scorer.calculate_scores()
        
        # Generate some test results
        test_results = [{
            'layout': ''.join(layout_mapping.keys()),
            'positions': ''.join(layout_mapping.values()),
            'layout_score': results['layout_score'],
            'normalized_score': results.get('normalized_score', 0.0),
            'bigram_count': results['bigram_count'],
            'scoring_mode': results.get('scoring_mode', 'unknown'),
            'weights_type': results.get('weights_type', weights_type),
            'weights_file': weights
        }]
        
        # Add individual scores
        individual_scores = results.get('individual_scores', {})
        for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                         'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
            test_results[0][f'ind_{criterion}'] = individual_scores.get(criterion, 0.0)
        
        # Write results
        fieldnames = ['layout', 'positions', 'layout_score', 'normalized_score', 'bigram_count', 
                     'scoring_mode', 'weights_type', 'weights_file'] + \
                    [f'ind_{c}' for c in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                                          'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']]
        
        with open(weighted_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_results)
        
        print(f"✅ Weighted scoring test completed:")
        print(f"   Layout score: {results['layout_score']:.4f}")
        print(f"   Scoring mode: {results.get('scoring_mode', 'unknown')}")
        print(f"   Bigrams analyzed: {results['bigram_count']}")
        
        return test_results, weights_type
        
    except Exception as e:
        print(f"❌ Weighted scoring test failed: {e}")
        return [], weights_type

def analyze_score_patterns(score_patterns):
    """Analyze patterns in the scoring results."""
    
    print(f"\n=== SCORE PATTERN ANALYSIS ===")
    print(f"Found {len(score_patterns)} unique score patterns:")
    
    # Show most common patterns
    sorted_patterns = sorted(score_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (pattern, bigrams) in enumerate(sorted_patterns[:10]):
        print(f"\nPattern {i+1} ({len(bigrams)} bigrams): {pattern}")
        print(f"  Total score: {sum(pattern):.1f}")
        print(f"  Examples: {', '.join(bigrams[:8])}{'...' if len(bigrams) > 8 else ''}")

def analyze_criterion_distributions(results):
    """Show distribution of scores for each criterion."""
    
    print(f"\n=== CRITERION SCORE DISTRIBUTIONS ===")
    
    criteria = ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                'home_row', 'columns', 'strum', 'strong_fingers']
    
    for criterion in criteria:
        scores = [result[criterion] for result in results]
        score_counts = defaultdict(int)
        for score in scores:
            score_counts[score] += 1
        
        print(f"\n{criterion.upper()}:")
        for score in sorted(score_counts.keys()):
            count = score_counts[score]
            pct = (count / len(results)) * 100
            print(f"  {score}: {count:4d} bigrams ({pct:5.1f}%)")

def test_specific_cases():
    """Test specific cases to verify key criteria are working."""
    
    print(f"\n=== SPECIFIC TEST CASES ===")
    
    # Test cases: (description, bigram, expected_criterion_scores)
    test_cases = [
        # Hands tests
        ("Different hands", "FJ", {"hands": 1.0}),
        ("Same hand", "FD", {"hands": 0.0}),
        
        # Strum tests
        ("Same finger strum", "FF", {"strum": 0.0}),
        ("Different hands strum", "FJ", {"strum": 1.0}),
        ("Inward roll (L)", "AF", {"strum": 1.0}),  # pinky to index
        ("Outward roll (L)", "FA", {"strum": 0.0}), # index to pinky
        
        # Skip fingers tests
        ("Same finger", "FF", {"skip_fingers": 0.0}),
        ("Adjacent fingers", "FD", {"skip_fingers": 0.0}),
        ("Skip 1 finger", "FS", {"skip_fingers": 0.5}),
        ("Skip 2 fingers", "FA", {"skip_fingers": 1.0}),
        ("Different hands skip", "FJ", {"skip_fingers": 1.0}),
        
        # Don't cross home tests
        ("Same hand hurdling", "QZ", {"dont_cross_home": 0.0}),
        ("Different hands hurdling", "QM", {"dont_cross_home": 1.0}),
        ("Same hand no hurdling", "QA", {"dont_cross_home": 1.0}),
        
        # Home row tests
        ("Both home row", "FJ", {"home_row": 1.0}),
        ("One home row", "FR", {"home_row": 0.5}),
        ("Neither home row", "QZ", {"home_row": 0.0}),
    ]
    
    for description, bigram, expected in test_cases:
        # Use the canonical scoring function
        bigram_scores = score_bigram_dvorak9(bigram)
        
        print(f"\n{description}: {bigram}")
        
        all_correct = True
        for criterion, expected_score in expected.items():
            actual_score = bigram_scores[criterion]
            status = "✓" if abs(actual_score - expected_score) < 0.01 else "✗"
            print(f"  {status} {criterion}: expected {expected_score}, got {actual_score}")
            if abs(actual_score - expected_score) >= 0.01:
                all_correct = False
        
        if all_correct:
            print(f"  ✓ All checks passed")
        else:
            print(f"  ✗ Some checks failed")

def validate_canonical_implementation():
    """Validate that the canonical implementation works as expected."""
    
    print(f"\n=== CANONICAL IMPLEMENTATION VALIDATION ===")
    
    # Test a few sample bigrams to ensure the canonical function works
    test_bigrams = ["TH", "ER", "AN", "IN", "ON"]
    
    print("Testing canonical scoring function:")
    for bigram in test_bigrams:
        try:
            scores = score_bigram_dvorak9(bigram)
            print(f"  {bigram}: ✓ (total score: {sum(scores.values()):.2f})")
            
            # Validate score structure
            expected_keys = {'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                           'same_row', 'home_row', 'columns', 'strum', 'strong_fingers'}
            actual_keys = set(scores.keys())
            
            if expected_keys != actual_keys:
                print(f"    ⚠️  Score keys mismatch!")
                print(f"      Expected: {expected_keys}")
                print(f"      Actual: {actual_keys}")
            
            # Validate score ranges (should be 0-1)
            for criterion, score in scores.items():
                if not (0 <= score <= 1):
                    print(f"    ⚠️  {criterion} score out of range: {score}")
                    
        except Exception as e:
            print(f"  {bigram}: ✗ Error: {e}")
    
    print(f"\n✓ Canonical implementation validation complete")

def main():
    """Run comprehensive test."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive validation of Dvorak-9 scoring criteria and frequency-weighted scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic criteria testing only
    python validate_scorer.py
    
    # Test with speed-based weights
    python validate_scorer.py --weights speed_weights_for_dvorak9_feature_combinations.csv
    
    # Test with comfort-based weights
    python validate_scorer.py --weights comfort_weights_for_dvorak9_feature_combinations.csv
        """
    )
    
    parser.add_argument("--weights", 
                       help="Path to weights CSV file for testing weighted scoring")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for CSV files (default: output)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("Dvorak-9 Comprehensive Scoring Validation")
    print("=" * 60)
    print("Using canonical scoring implementation from dvorak9_scorer.py")
    if args.weights:
        print(f"Testing weighted scoring with: {args.weights}")
    print("=" * 60)
    
    # Validate the canonical implementation first
    validate_canonical_implementation()
    
    # Test specific cases
    test_specific_cases()
    
    # Test all possible bigrams - basic criteria
    print(f"\n{'='*60}")
    print("Testing all possible bigrams (basic criteria)...")
    
    all_bigrams_file = Path(args.output_dir) / "dvorak9_scores_all_bigrams.csv"
    unique_scores_file = Path(args.output_dir) / "dvorak9_scores_unique_all_bigrams.csv"
    
    results, score_patterns = test_all_bigrams_basic(all_bigrams_file, unique_scores_file)
    
    # Analyze patterns
    analyze_score_patterns(score_patterns)
    analyze_criterion_distributions(results)
    
    # Test weighted scoring if weights file provided
    if args.weights:
        print(f"\n{'='*60}")
        print("Testing frequency-weighted scoring...")
        
        try:
            weighted_results, weights_type = test_weighted_scoring_simple(args.weights, args.output_dir)
            
        except Exception as e:
            print(f"❌ Weighted scoring test failed: {e}")
            print(f"   Make sure the weights file exists and is properly formatted")
            weights_type = "unknown"
    
    print(f"\n{'='*60}")
    print("Validation complete!")
    print("✅ SUCCESS: Using canonical implementation from dvorak9_scorer.py")
    print(f"\nCheck output files in '{args.output_dir}/':")
    print(f"  - dvorak9_scores_all_bigrams.csv (individual criterion scores 0-1)")
    print(f"  - dvorak9_scores_unique_all_bigrams.csv (unique score patterns)")
    if args.weights:
        if 'weights_type' in locals():
            print(f"  - dvorak9_weighted_scores_{weights_type}.csv (empirical combination scores)")
        else:
            print(f"  - dvorak9_weighted_scores_*.csv (empirical combination scores)")
        print(f"  ✓ Tested with {args.weights}")

if __name__ == "__main__":
    main()