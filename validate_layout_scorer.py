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
"""

import sys
import os
import argparse
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
  layout_input_consistency  - Test consistency between execution modes
  scorer_functionality      - Test that all scorers work
  cross_hand_filtering      - Test cross-hand filtering consistency
  output_formats           - Test output format generation
  data_files               - Test data file availability

Examples:
  python validate_layout_scorer.py                                    # Run all tests
  python validate_layout_scorer.py --test layout_input_consistency    # Run specific test
  python validate_layout_scorer.py --quiet                           # Minimal output
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