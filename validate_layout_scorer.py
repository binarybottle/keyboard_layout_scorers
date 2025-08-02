#!/usr/bin/env python3
"""
Comprehensive validation script for the keyboard layout scorer.

This script validates all major functionality including:
- Layout input consistency between execution modes
- Scorer functionality and accuracy
- Cross-hand filtering consistency
- Output format validation
- Configuration and data file validation
- Error handling
- Empirical weighting system validation (NEW)
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def validate_layout_input_consistency() -> bool:
    """Validate that single scorer and compare modes produce identical results."""
    
    print("=" * 60)
    print("VALIDATING LAYOUT INPUT CONSISTENCY")
    print("=" * 60)
    
    try:
        from framework.layout_utils import create_layout_mapping_consistent, parse_layout_compare
        from framework.cli_utils import get_layout_from_args
        import argparse
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test cases with different input patterns
    test_cases = [
        {
            "name": "Mixed letters and punctuation (your original case)",
            "letters": "bfnc'\"liukpsat,.eormvxgd-?hyjwqz",
            "positions": "qwertyuiop[asdfghjkl;'zxcvbnm,./"
        },
        {
            "name": "Standard 26-letter alphabet",
            "letters": "abcdefghijklmnopqrstuvwxyz[';,./",
            "positions": "qwertyuiop[asdfghjkl;'zxcvbnm,./"
        },
        {
            "name": "Dvorak layout (32 chars)",
            "letters": "',.pyfgcrlaoeuidhtns;qjkxbmwvz[/",
            "positions": "qwertyuiop[asdfghjkl;'zxcvbnm,./"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print(f"Letters:    {test_case['letters']}")
        print(f"Positions:  {test_case['positions']}")
        
        # Check length consistency first
        if len(test_case['letters']) != len(test_case['positions']):
            print(f"❌ Length mismatch: {len(test_case['letters'])} vs {len(test_case['positions'])}")
            all_passed = False
            continue
        
        try:
            # Test single scorer mode
            single_mapping = create_layout_mapping_consistent(
                test_case['letters'], test_case['positions']
            )
            
            # Test compare mode
            compare_layouts = parse_layout_compare([f"test:{test_case['letters']}"])
            compare_mapping = compare_layouts["test"]
            
            # Check consistency
            if single_mapping == compare_mapping:
                print(f"✅ Consistent: {len(single_mapping)} characters mapped")
            else:
                print(f"❌ Inconsistent mappings")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            all_passed = False
    
    return all_passed


def validate_scorer_functionality() -> bool:
    """Validate that all scorers work with basic inputs."""
    
    print("\n" + "=" * 60)
    print("VALIDATING SCORER FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from framework.unified_scorer import UnifiedLayoutScorer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test layout
    test_layout = {
        'a': 'Q', 'b': 'W', 'c': 'E', 'd': 'R', 'e': 'T',
        'f': 'Y', 'g': 'U', 'h': 'I', 'i': 'O', 'j': 'P',
        'k': 'A', 'l': 'S', 'm': 'D', 'n': 'F', 'o': 'G',
        'p': 'H', 'q': 'J', 'r': 'K', 's': 'L', 't': ';',
        'u': 'Z', 'v': 'X', 'w': 'C', 'x': 'V', 'y': 'B', 'z': 'N'
    }
    
    unified_scorer = UnifiedLayoutScorer()
    all_passed = True
    
    # Test each scorer
    scorers_to_test = [
        ('engram', {}),
        ('dvorak9', {}),
        ('distance', {'text': 'hello world test'})
    ]
    
    for scorer_name, kwargs in scorers_to_test:
        print(f"\nTesting {scorer_name} scorer...")
        
        try:
            results = unified_scorer.score_layout(test_layout, [scorer_name], **kwargs)
            
            if scorer_name in results and not results[scorer_name].metadata.get('scorer_failed', False):
                score = results[scorer_name].primary_score
                print(f"✅ {scorer_name}: score = {score:.6f}")
            else:
                print(f"❌ {scorer_name}: failed to produce valid score")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {scorer_name}: error = {e}")
            all_passed = False
    
    return all_passed


def validate_cross_hand_filtering() -> bool:
    """Validate that cross-hand filtering works consistently across scorers."""
    
    print("\n" + "=" * 60)
    print("VALIDATING CROSS-HAND FILTERING")
    print("=" * 60)
    
    try:
        from framework.unified_scorer import UnifiedLayoutScorer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Simple test layout
    test_layout = {
        'a': 'Q', 'b': 'W', 'c': 'E', 'd': 'R',  # Left hand
        'e': 'Y', 'f': 'U', 'g': 'I', 'h': 'O'   # Right hand
    }
    
    unified_scorer = UnifiedLayoutScorer()
    all_passed = True
    
    scorers_to_test = ['engram', 'dvorak9']
    
    for scorer_name in scorers_to_test:
        print(f"\nTesting {scorer_name} cross-hand filtering...")
        
        try:
            # Test without filtering
            results_normal = unified_scorer.score_layout(
                test_layout, [scorer_name], ignore_cross_hand=False
            )
            
            # Test with filtering
            results_filtered = unified_scorer.score_layout(
                test_layout, [scorer_name], ignore_cross_hand=True
            )
            
            if (scorer_name in results_normal and scorer_name in results_filtered):
                score_normal = results_normal[scorer_name].primary_score
                score_filtered = results_filtered[scorer_name].primary_score
                
                print(f"✅ {scorer_name}: normal = {score_normal:.6f}, filtered = {score_filtered:.6f}")
                
                # Scores should be different (filtered should remove cross-hand bigrams)
                if abs(score_normal - score_filtered) < 1e-6:
                    print(f"⚠️  Warning: Scores identical - filtering may not be working")
                    
            else:
                print(f"❌ {scorer_name}: failed to produce results")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {scorer_name}: error = {e}")
            all_passed = False
    
    return all_passed


def validate_output_formats() -> bool:
    """Validate that all output formats work correctly."""
    
    print("\n" + "=" * 60)
    print("VALIDATING OUTPUT FORMATS")
    print("=" * 60)
    
    try:
        from framework.base_scorer import ScoreResult
        from framework.output_utils import format_csv_output, format_detailed_output, format_score_only_output
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create test result
    test_result = ScoreResult(
        primary_score=0.123456,
        components={'test_component': 0.789012, 'another_component': 0.345678},
        metadata={'test_meta': 'test_value'},
        validation_info={'test_validation': 'test_info'}
    )
    test_result.scorer_name = "test_scorer"
    
    formats_to_test = [
        ('detailed', format_detailed_output),
        ('csv', format_csv_output),
        ('score_only', format_score_only_output)
    ]
    
    all_passed = True
    
    for format_name, format_function in formats_to_test:
        print(f"\nTesting {format_name} format...")
        
        try:
            output = format_function(test_result)
            
            if output and len(output.strip()) > 0:
                lines = len(output.split('\n'))
                print(f"✅ {format_name}: {lines} lines generated")
            else:
                print(f"❌ {format_name}: empty output")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {format_name}: error = {e}")
            all_passed = False
    
    return all_passed


def validate_data_files() -> bool:
    """Validate that required data files exist and are readable."""
    
    print("\n" + "=" * 60)
    print("VALIDATING DATA FILES")
    print("=" * 60)
    
    try:
        from framework.config_loader import get_config_loader
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    config_loader = get_config_loader()
    all_passed = True
    
    scorers = ['engram_scorer', 'dvorak9_scorer', 'distance_scorer']
    
    for scorer_name in scorers:
        print(f"\nValidating {scorer_name} data files...")
        
        try:
            config = config_loader.get_scorer_config(scorer_name)
            data_files = config.get('data_files', {})
            
            if not data_files:
                print(f"✅ {scorer_name}: no data files required")
                continue
            
            missing_files = []
            for file_key, filepath in data_files.items():
                if filepath and not Path(filepath).exists():
                    missing_files.append(f"{file_key}: {filepath}")
            
            if missing_files:
                print(f"❌ {scorer_name}: missing files:")
                for missing in missing_files:
                    print(f"   - {missing}")
                # Don't fail validation for optional files
                if scorer_name != 'distance_scorer':  # distance scorer has no required files
                    all_passed = False
            else:
                print(f"✅ {scorer_name}: all data files found")
                
        except Exception as e:
            print(f"❌ {scorer_name}: error = {e}")
            all_passed = False
    
    return all_passed


def validate_empirical_weight_loading() -> bool:
    """Validate that empirical weight files load correctly."""
    
    print("\n" + "=" * 60)
    print("VALIDATING EMPIRICAL WEIGHT LOADING")
    print("=" * 60)
    
    try:
        from dvorak9_scorer import load_combination_weights
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test weight file paths (adjust these to match your setup)
    weight_files = [
        ("speed", "input/dvorak9/speed_weights.csv"),
        ("comfort", "input/dvorak9/comfort_weights.csv")
    ]
    
    all_passed = True
    
    for weight_type, filepath in weight_files:
        print(f"\nTesting {weight_type} weights: {filepath}")
        
        # Check file exists
        if not Path(filepath).exists():
            print(f"❌ {weight_type} weights file not found: {filepath}")
            all_passed = False
            continue
        
        try:
            # Test loading weights
            weights = load_combination_weights(filepath, quiet=True)
            
            if len(weights) == 0:
                print(f"⚠️  {weight_type} weights: no combinations loaded (FDR too strict)")
                # This might be expected if FDR correction eliminates all combinations
            else:
                print(f"✅ {weight_type} weights: {len(weights)} combinations loaded")
                
                # Check weight value ranges
                weight_values = list(weights.values())
                weight_range = (min(weight_values), max(weight_values))
                print(f"   Range: {weight_range[0]:.4f} to {weight_range[1]:.4f}")
                
                # Show a few examples
                example_combos = list(weights.items())[:3]
                for combo, weight in example_combos:
                    combo_str = ' + '.join(combo) if combo else 'empty'
                    print(f"   Example: {combo_str}: {weight:.4f}")
                    
        except Exception as e:
            print(f"❌ {weight_type} weights loading error: {e}")
            all_passed = False
    
    return all_passed


def validate_combination_diversity() -> bool:
    """Validate that different bigrams get assigned to different combinations."""
    
    print("\n" + "=" * 60)
    print("VALIDATING COMBINATION DIVERSITY")
    print("=" * 60)
    
    try:
        from dvorak9_scorer import score_bigram_dvorak9, identify_bigram_combination
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test a variety of bigrams to check combination diversity
    test_bigrams = [
        'th', 'he', 'in', 'er', 'an',  # Common bigrams
        'qx', 'zj', 'wq', 'xz',        # Uncommon bigrams
        'aa', 'ss', 'dd', 'ff',        # Same-key bigrams
        'ae', 'fb', 'gc', 'hd'         # Mixed patterns
    ]
    
    combinations = {}
    
    print(f"Testing combination assignment for {len(test_bigrams)} bigrams...")
    
    for bigram in test_bigrams:
        try:
            scores = score_bigram_dvorak9(bigram)
            combination = identify_bigram_combination(scores, threshold=0.1)
            
            if combination not in combinations:
                combinations[combination] = []
            combinations[combination].append(bigram)
            
        except Exception as e:
            print(f"❌ Error processing bigram '{bigram}': {e}")
            return False
    
    print(f"\nFound {len(combinations)} unique combinations:")
    
    # Sort by frequency
    sorted_combos = sorted(combinations.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (combo, bigrams) in enumerate(sorted_combos[:5]):  # Show top 5
        combo_str = ' + '.join(combo) if combo else 'empty'
        bigram_list = ', '.join(bigrams)
        print(f"  {i+1}. {combo_str} ({len(bigrams)} bigrams): {bigram_list}")
    
    # Check for the problematic case: all bigrams assigned to same massive combination
    if len(combinations) == 1:
        combo = list(combinations.keys())[0]
        if len(combo) >= 8:  # 8 or 9 criteria = problematic
            print(f"❌ ALL bigrams assigned to same massive combination!")
            print(f"   This indicates the combination identification bug has returned.")
            print(f"   Combination: {' + '.join(combo)}")
            return False
    
    # Good diversity check
    dominant_combo = sorted_combos[0]
    dominant_pct = len(dominant_combo[1]) / len(test_bigrams) * 100
    
    if dominant_pct > 80:
        print(f"⚠️  Warning: {dominant_pct:.1f}% of bigrams use the same combination")
        print(f"   This may indicate insufficient diversity in combination assignment")
        if dominant_pct > 95:
            return False
    else:
        print(f"✅ Good diversity: Most frequent combination used by {dominant_pct:.1f}% of bigrams")
    
    return True


def validate_empirical_scoring_differences() -> bool:
    """Validate that different layouts get different empirical scores."""
    
    print("\n" + "=" * 60)
    print("VALIDATING EMPIRICAL SCORING DIFFERENCES")
    print("=" * 60)
    
    try:
        from framework.unified_scorer import UnifiedLayoutScorer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create test layouts with different characteristics
    test_layouts = {
        "layout1": {  # Left-hand heavy
            'e': 'Q', 't': 'W', 'a': 'E', 'o': 'R',
            'i': 'A', 'n': 'S', 's': 'D', 'h': 'F'
        },
        "layout2": {  # Right-hand heavy  
            'e': 'U', 't': 'I', 'a': 'O', 'o': 'P',
            'i': 'J', 'n': 'K', 's': 'L', 'h': ';'
        },
        "layout3": {  # Balanced
            'e': 'F', 't': 'J', 'a': 'D', 'o': 'K',
            'i': 'S', 'n': 'L', 's': 'A', 'h': ';'
        }
    }
    
    unified_scorer = UnifiedLayoutScorer()
    all_passed = True
    
    print(f"Testing {len(test_layouts)} different layouts...")
    
    # Score all layouts
    layout_scores = {}
    
    for layout_name, layout_mapping in test_layouts.items():
        try:
            results = unified_scorer.score_layout(
                layout_mapping, ['dvorak9'], ignore_cross_hand=False
            )
            
            if 'dvorak9' in results:
                result = results['dvorak9']
                components = result.components
                
                # Extract the three key scores
                freq_score = components.get('frequency_weighted_score')
                speed_score = components.get('speed_weighted_score') 
                comfort_score = components.get('comfort_weighted_score')
                
                layout_scores[layout_name] = {
                    'frequency': freq_score,
                    'speed': speed_score,
                    'comfort': comfort_score
                }
                
                print(f"✅ {layout_name}: freq={freq_score:.6f}, speed={speed_score:.6f}, comfort={comfort_score:.6f}")
                
            else:
                print(f"❌ {layout_name}: failed to get dvorak9 results")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {layout_name}: error = {e}")
            all_passed = False
    
    if len(layout_scores) < 2:
        print(f"❌ Not enough layouts scored successfully for comparison")
        return False
    
    # Check for differences between layouts
    print(f"\nChecking score differences between layouts...")
    
    score_types = ['frequency', 'speed', 'comfort']
    layouts = list(layout_scores.keys())
    
    for score_type in score_types:
        scores = [layout_scores[layout][score_type] for layout in layouts if layout_scores[layout][score_type] is not None]
        
        if len(scores) < 2:
            print(f"⚠️  {score_type}: insufficient data for comparison")
            continue
        
        score_range = max(scores) - min(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0
        
        print(f"   {score_type}: range = {score_range:.6f}, std = {score_std:.6f}")
        
        # Check for meaningful differences
        if score_range < 1e-6:
            print(f"❌ {score_type}: scores are identical across layouts!")
            all_passed = False
        elif score_range < 1e-4:
            print(f"⚠️  {score_type}: very small differences (may indicate issues)")
        else:
            print(f"✅ {score_type}: meaningful differences found")
    
    return all_passed


def validate_linear_correlation_absence() -> bool:
    """Validate that the three Dvorak scores are NOT perfectly correlated."""
    
    print("\n" + "=" * 60)
    print("VALIDATING ABSENCE OF LINEAR CORRELATION")
    print("=" * 60)
    
    try:
        from framework.unified_scorer import UnifiedLayoutScorer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create multiple diverse test layouts
    test_layouts = {}
    
    # Generate 10 different layouts with varying patterns
    common_chars = "etaoinshrldu"
    base_positions = ["QWERTYUIOPAS", "ASDFGHJKLZXC", "ZXCVBNMQWERT", 
                     "FGJKLASDFGHJ", "YUIOPASDFGHJ"]
    
    for i in range(min(10, len(base_positions))):
        positions = base_positions[i % len(base_positions)]
        layout = {char: pos for char, pos in zip(common_chars, positions)}
        test_layouts[f"layout_{i+1}"] = layout
    
    unified_scorer = UnifiedLayoutScorer()
    
    print(f"Testing correlation across {len(test_layouts)} layouts...")
    
    # Collect scores
    freq_scores = []
    speed_scores = []
    comfort_scores = []
    
    for layout_name, layout_mapping in test_layouts.items():
        try:
            results = unified_scorer.score_layout(
                layout_mapping, ['dvorak9'], ignore_cross_hand=False
            )
            
            if 'dvorak9' in results:
                components = results['dvorak9'].components
                
                freq = components.get('frequency_weighted_score')
                speed = components.get('speed_weighted_score')
                comfort = components.get('comfort_weighted_score')
                
                if freq is not None and speed is not None and comfort is not None:
                    freq_scores.append(freq)
                    speed_scores.append(speed)
                    comfort_scores.append(comfort)
                    
        except Exception as e:
            print(f"⚠️  Error with {layout_name}: {e}")
    
    if len(freq_scores) < 3:
        print(f"❌ Insufficient data for correlation analysis ({len(freq_scores)} layouts)")
        return False
    
    # Calculate correlations
    corr_freq_speed = np.corrcoef(freq_scores, speed_scores)[0, 1]
    corr_freq_comfort = np.corrcoef(freq_scores, comfort_scores)[0, 1] 
    corr_speed_comfort = np.corrcoef(speed_scores, comfort_scores)[0, 1]
    
    print(f"\nCorrelation analysis ({len(freq_scores)} layouts):")
    print(f"   Frequency vs Speed:   r = {corr_freq_speed:.6f}")
    print(f"   Frequency vs Comfort: r = {corr_freq_comfort:.6f}")
    print(f"   Speed vs Comfort:     r = {corr_speed_comfort:.6f}")
    
    # Check for problematic perfect correlations
    threshold = 0.999  # Near-perfect correlation threshold
    
    perfect_correlations = []
    if abs(corr_freq_speed) > threshold:
        perfect_correlations.append("Frequency vs Speed")
    if abs(corr_freq_comfort) > threshold:
        perfect_correlations.append("Frequency vs Comfort")
    if abs(corr_speed_comfort) > threshold:
        perfect_correlations.append("Speed vs Comfort")
    
    if perfect_correlations:
        print(f"❌ Perfect correlations detected: {', '.join(perfect_correlations)}")
        print(f"   This indicates the empirical weighting bug has returned!")
        return False
    else:
        print(f"✅ No perfect correlations - empirical weighting working correctly")
        return True


def validate_empirical_scores_presence() -> bool:
    """Validate that empirical scores are present in results."""
    
    print("\n" + "=" * 60)
    print("VALIDATING EMPIRICAL SCORES PRESENCE")
    print("=" * 60)
    
    try:
        from framework.unified_scorer import UnifiedLayoutScorer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Simple test layout
    test_layout = {
        'e': 'F', 't': 'J', 'a': 'D', 'o': 'K',
        'i': 'S', 'n': 'L', 's': 'A', 'h': ';'
    }
    
    unified_scorer = UnifiedLayoutScorer()
    
    try:
        results = unified_scorer.score_layout(test_layout, ['dvorak9'])
        
        if 'dvorak9' not in results:
            print(f"❌ dvorak9 results not found")
            return False
        
        components = results['dvorak9'].components
        
        # Check for required scores
        required_scores = {
            'pure_dvorak_score': 'Pure Dvorak Score',
            'frequency_weighted_score': 'Frequency Weighted Score'
        }
        
        empirical_scores = {
            'speed_weighted_score': 'Speed Weighted Score',
            'comfort_weighted_score': 'Comfort Weighted Score'
        }
        
        all_passed = True
        
        # Check required scores
        for score_key, score_name in required_scores.items():
            if score_key in components and components[score_key] is not None:
                print(f"✅ {score_name}: {components[score_key]:.6f}")
            else:
                print(f"❌ {score_name}: missing or null")
                all_passed = False
        
        # Check empirical scores
        empirical_present = 0
        for score_key, score_name in empirical_scores.items():
            if score_key in components and components[score_key] is not None:
                print(f"✅ {score_name}: {components[score_key]:.6f}")
                empirical_present += 1
            else:
                print(f"⚠️  {score_name}: missing (no empirical weights loaded)")
        
        if empirical_present == 0:
            print(f"⚠️  No empirical scores found - weights may not be loading")
            # Don't fail validation for this, as it might be expected
        else:
            print(f"✅ {empirical_present}/2 empirical scores present")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing empirical scores: {e}")
        return False


def run_comprehensive_validation(args) -> int:
    """Run all validation tests."""
    
    print("COMPREHENSIVE LAYOUT SCORER VALIDATION")
    print("=" * 80)
    
    validation_tests = [
        ("Layout Input Consistency", validate_layout_input_consistency),
        ("Scorer Functionality", validate_scorer_functionality),
        ("Cross-Hand Filtering", validate_cross_hand_filtering),
        ("Output Formats", validate_output_formats),
        ("Data Files", validate_data_files),
        
        # NEW: Empirical weighting validation tests
        ("Empirical Weight Loading", validate_empirical_weight_loading),
        ("Combination Diversity", validate_combination_diversity), 
        ("Empirical Scoring Differences", validate_empirical_scoring_differences),
        ("Linear Correlation Absence", validate_linear_correlation_absence),
        ("Empirical Scores Presence", validate_empirical_scores_presence),
    ]
    
    if args.test:
        # Run only specific test
        test_name = args.test.replace('_', ' ').title()
        test_function = None
        
        for name, func in validation_tests:
            if name.lower().replace(' ', '_') == args.test.lower():
                test_function = func
                break
        
        if test_function:
            print(f"Running single test: {test_name}")
            success = test_function()
            return 0 if success else 1
        else:
            available_tests = [name.lower().replace(' ', '_') for name, _ in validation_tests]
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(available_tests)}")
            return 1
    
    # Run all tests
    results = []
    
    for test_name, test_function in validation_tests:
        if not args.quiet:
            print(f"\n" + "=" * 80)
            print(f"RUNNING: {test_name.upper()}")
        
        try:
            success = test_function()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error = {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        return 0
    else:
        print(f"💥 {total - passed} validations failed")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive validation for keyboard layout scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available individual tests:
  layout_input_consistency     - Test consistency between execution modes
  scorer_functionality         - Test that all scorers work
  cross_hand_filtering         - Test cross-hand filtering consistency
  output_formats              - Test output format generation
  data_files                  - Test data file availability
  empirical_weight_loading     - Test empirical weight file loading
  combination_diversity        - Test bigram combination assignment diversity
  empirical_scoring_differences - Test that layouts get different empirical scores
  linear_correlation_absence   - Test that Dvorak scores are not perfectly correlated
  empirical_scores_presence    - Test that empirical scores appear in results

Examples:
  python validate_layout_scorer.py                                        # Run all tests
  python validate_layout_scorer.py --test empirical_weight_loading        # Test weight loading
  python validate_layout_scorer.py --test linear_correlation_absence      # Test correlations
  python validate_layout_scorer.py --quiet                               # Minimal output
        """
    )
    
    parser.add_argument(
        '--test',
        help="Run only a specific test (see available tests above)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        return run_comprehensive_validation(args)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())