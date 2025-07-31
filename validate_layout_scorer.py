#!/usr/bin/env python3
"""
Comprehensive validation script for layout_scorer.py

Tests all argument combinations, output formats, and error conditions.

# Run all validation tests
python validate_layout_scorer.py

# Test specific script location
python validate_layout_scorer.py --script /path/to/layout_scorer.py

# Save detailed report
python validate_layout_scorer.py --report validation_report.json

# Run quietly (less verbose output)
python validate_layout_scorer.py --quiet
"""

import subprocess
import tempfile
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil


class LayoutScorerValidator:
    """Comprehensive validator for layout_scorer.py functionality."""
    
    def __init__(self, script_path: str = "layout_scorer.py", verbose: bool = True):
        """
        Initialize validator.
        
        Args:
            script_path: Path to layout_scorer.py
            verbose: Whether to print detailed test results
        """
        self.script_path = script_path
        self.verbose = verbose
        self.test_results = []
        self.temp_dir = None
        
        # Test data
        self.test_layouts = {
            'simple': {'letters': 'abc', 'positions': 'ABC'},
            'medium': {'letters': 'etaoinshrlcu', 'positions': 'FDESGJWXRTYZ'},
            'full': {'letters': 'etaoinshrlcudmwfgypbvkjxqz', 'positions': 'FDESGJWXRTYZVBNMQWERTYUIOP'},
        }
        
        self.test_text = "the quick brown fox jumps over the lazy dog"
        
        # Comparison layouts
        self.comparison_layouts = [
            'qwerty:"qwertyuiopasdfghjklzxcvbnm"',
            'dvorak:"\',.pyfgcrlaoeuidhtnsqjkxbmwvz"',
            'colemak:"qwfpgjluy;arstdhneiozxcvbkm"'
        ]
        
    def setup(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="layout_scorer_test_")
        
        # Create test text file
        test_text_file = Path(self.temp_dir) / "test_text.txt"
        with open(test_text_file, 'w') as f:
            f.write(self.test_text)
        
        self.test_text_file = str(test_text_file)
        
        if self.verbose:
            print(f"Test directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def run_command(self, args: List[str], expect_success: bool = True) -> Tuple[bool, str, str]:
        """
        Run layout_scorer.py with given arguments.
        
        Args:
            args: Command line arguments
            expect_success: Whether command should succeed
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [sys.executable, self.script_path] + args
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            success = (result.returncode == 0) == expect_success
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", f"Command failed: {e}"
    
    def test_single_scorer_modes(self):
        """Test all single scorer modes."""
        tests = []
        
        for scorer in ['distance', 'dvorak9', 'engram']:
            for layout_name, layout in self.test_layouts.items():
                args = [
                    '--scorer', scorer,
                    '--letters', layout['letters'],
                    '--positions', layout['positions']
                ]
                
                # Add text for distance scorer
                if scorer == 'distance':
                    args.extend(['--text', self.test_text])
                
                test_name = f"single_{scorer}_{layout_name}"
                tests.append((test_name, args, True))
        
        return self._run_test_batch("Single Scorer Modes", tests)
    
    def test_multiple_scorer_modes(self):
        """Test multiple scorer combinations."""
        tests = []
        
        layout = self.test_layouts['medium']
        base_args = ['--letters', layout['letters'], '--positions', layout['positions']]
        
        # Test specific combinations
        scorer_combos = [
            'distance,dvorak9',
            'dvorak9,engram',
            'distance,engram',
            'all'
        ]
        
        for combo in scorer_combos:
            args = ['--scorers', combo] + base_args + ['--text', self.test_text]
            test_name = f"multi_{combo.replace(',', '_')}"
            tests.append((test_name, args, True))
        
        return self._run_test_batch("Multiple Scorer Modes", tests)
    
    def test_comparison_modes(self):
        """Test layout comparison modes."""
        tests = []
        
        # Basic comparison
        args = ['--compare'] + self.comparison_layouts[:2] + ['--text', self.test_text]
        tests.append(("compare_basic", args, True))
        
        # Three-way comparison
        args = ['--compare'] + self.comparison_layouts + ['--text', self.test_text]
        tests.append(("compare_threeway", args, True))
        
        # Comparison without text (dvorak9 + engram only)
        args = ['--compare'] + self.comparison_layouts[:2]
        tests.append(("compare_no_text", args, True))
        
        return self._run_test_batch("Comparison Modes", tests)
    
    def test_output_formats(self):
        """Test all output formats."""
        tests = []
        
        layout = self.test_layouts['medium']
        base_args = [
            '--scorer', 'dvorak9',
            '--letters', layout['letters'],
            '--positions', layout['positions']
        ]
        
        # Test each format
        for format_type in ['detailed', 'csv', 'score_only']:
            args = base_args + ['--format', format_type]
            tests.append((f"format_{format_type}", args, True))
        
        # Test format shortcuts
        args = base_args + ['--score-only']
        tests.append(("format_shortcut_score_only", args, True))
        
        return self._run_test_batch("Output Formats", tests)
    
    def test_csv_file_output(self):
        """Test CSV file output functionality."""
        tests = []
        
        # Single layout CSV output
        csv_file = Path(self.temp_dir) / "single_output.csv"
        layout = self.test_layouts['medium']
        args = [
            '--scorer', 'dvorak9',
            '--letters', layout['letters'],
            '--positions', layout['positions'],
            '--csv', str(csv_file)
        ]
        tests.append(("csv_single", args, True))
        
        # Comparison CSV output
        csv_file2 = Path(self.temp_dir) / "comparison_output.csv"
        args = [
            '--compare'
        ] + self.comparison_layouts[:2] + [
            '--csv', str(csv_file2),
            '--text', self.test_text
        ]
        tests.append(("csv_comparison", args, True))
        
        results = self._run_test_batch("CSV File Output", tests)
        
        # Validate CSV files were created and have correct structure
        for test_name, _, _ in tests:
            if test_name == "csv_single":
                self._validate_csv_file(csv_file, expected_rows=1)
            elif test_name == "csv_comparison":
                self._validate_csv_file(csv_file2, expected_rows=2)
        
        return results
    
    def test_text_input_methods(self):
        """Test different text input methods."""
        tests = []
        
        layout = self.test_layouts['medium']
        base_args = [
            '--scorer', 'distance',
            '--letters', layout['letters'],
            '--positions', layout['positions']
        ]
        
        # Text from command line
        args = base_args + ['--text', self.test_text]
        tests.append(("text_cmdline", args, True))
        
        # Text from file
        args = base_args + ['--text-file', self.test_text_file]
        tests.append(("text_file", args, True))
        
        return self._run_test_batch("Text Input Methods", tests)
    
    def test_scorer_specific_options(self):
        """Test scorer-specific options."""
        tests = []
        
        layout = self.test_layouts['medium']
        base_args = ['--letters', layout['letters'], '--positions', layout['positions']]
        
        # Engram with ignore-cross-hand
        args = ['--scorer', 'engram'] + base_args + ['--ignore-cross-hand']
        tests.append(("engram_ignore_cross_hand", args, True))
        
        # Dvorak9 with weights (if weights file exists)
        weights_file = "input/dvorak9/speed_weights.csv"
        if Path(weights_file).exists():
            args = ['--scorer', 'dvorak9'] + base_args + ['--weights', weights_file]
            tests.append(("dvorak9_weights", args, True))
        
        return self._run_test_batch("Scorer-Specific Options", tests)
    
    def test_error_conditions(self):
        """Test various error conditions."""
        tests = []
        
        # Missing required arguments
        tests.append(("error_no_args", [], False))
        tests.append(("error_no_positions", ['--scorer', 'dvorak9', '--letters', 'abc'], False))
        tests.append(("error_no_letters", ['--scorer', 'dvorak9', '--positions', 'ABC'], False))
        
        # Invalid scorer
        tests.append(("error_invalid_scorer", [
            '--scorer', 'invalid',
            '--letters', 'abc',
            '--positions', 'ABC'
        ], False))
        
        # Distance scorer without text (should succeed but report error)
        tests.append(("error_distance_no_text", [
            '--scorer', 'distance',
            '--letters', 'abc',
            '--positions', 'ABC'
        ], True))  # Should succeed (True)
        
        # Invalid text file
        tests.append(("error_invalid_text_file", [
            '--scorer', 'distance',
            '--letters', 'abc',
            '--positions', 'ABC',
            '--text-file', '/nonexistent/file.txt'
        ], False))
        
        # Mismatched letters and positions length
        tests.append(("error_length_mismatch", [
            '--scorer', 'dvorak9',
            '--letters', 'abc',
            '--positions', 'ABCD'  # One too many
        ], False))
        
        # Invalid comparison format
        tests.append(("error_invalid_compare", [
            '--compare', 'invalid_format'  # Missing colon
        ], False))
        
        return self._run_test_batch("Error Conditions", tests)
    
    def test_quiet_mode(self):
        """Test quiet mode functionality."""
        tests = []
        
        layout = self.test_layouts['medium']
        base_args = [
            '--scorer', 'dvorak9',
            '--letters', layout['letters'],
            '--positions', layout['positions'],
            '--quiet'
        ]
        
        tests.append(("quiet_mode", base_args, True))
        
        return self._run_test_batch("Quiet Mode", tests)
    
    def _run_test_batch(self, batch_name: str, tests: List[Tuple[str, List[str], bool]]) -> List[Dict[str, Any]]:
        """Run a batch of tests and return results."""
        if self.verbose:
            print(f"\n=== {batch_name} ===")
        
        batch_results = []
        
        for test_name, args, expect_success in tests:
            success, stdout, stderr = self.run_command(args, expect_success)
            
            result = {
                'test_name': test_name,
                'batch': batch_name,
                'args': args,
                'expect_success': expect_success,
                'success': success,
                'stdout': stdout,
                'stderr': stderr
            }
            
            batch_results.append(result)
            self.test_results.append(result)
            
            if self.verbose:
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"  {test_name:<25} {status}")
                
                if not success and expect_success:
                    print(f"    Command: {' '.join([self.script_path] + args)}")
                    if stderr:
                        print(f"    Error: {stderr.strip()}")
        
        return batch_results
    
    def _validate_csv_file(self, csv_file: Path, expected_rows: int = None):
        """Validate that a CSV file exists and has the expected structure."""
        if not csv_file.exists():
            print(f"  ✗ CSV file not created: {csv_file}")
            return False
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 2:  # Header + at least one data row
                    print(f"  ✗ CSV file has insufficient rows: {len(rows)}")
                    return False
                
                if expected_rows and len(rows) - 1 != expected_rows:  # -1 for header
                    print(f"  ✗ CSV file has {len(rows)-1} data rows, expected {expected_rows}")
                    return False
                
                # Check header
                header = rows[0]
                if 'layout' not in header:
                    print(f"  ✗ CSV header missing 'layout' column")
                    return False
                
                if self.verbose:
                    print(f"  ✓ CSV file valid: {len(rows)-1} rows, {len(header)} columns")
                
                return True
                
        except Exception as e:
            print(f"  ✗ Error reading CSV file: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("Starting comprehensive validation of layout_scorer.py...")
        print(f"Script path: {self.script_path}")
        
        # Check if script exists
        if not Path(self.script_path).exists():
            print(f"Error: Script not found at {self.script_path}")
            return {'success': False, 'error': 'Script not found'}
        
        self.setup()
        
        try:
            # Run all test categories
            test_methods = [
                self.test_single_scorer_modes,
                self.test_multiple_scorer_modes,
                self.test_comparison_modes,
                self.test_output_formats,
                self.test_csv_file_output,
                self.test_text_input_methods,
                self.test_scorer_specific_options,
                self.test_error_conditions,
                self.test_quiet_mode,
            ]
            
            for test_method in test_methods:
                test_method()
            
            # Summary
            total_tests = len(self.test_results)
            passed_tests = sum(1 for r in self.test_results if r['success'])
            failed_tests = total_tests - passed_tests
            
            print(f"\n=== VALIDATION SUMMARY ===")
            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
            
            if failed_tests > 0:
                print(f"\nFailed tests:")
                for result in self.test_results:
                    if not result['success']:
                        print(f"  - {result['batch']}: {result['test_name']}")
                        if result['stderr']:
                            print(f"    Error: {result['stderr'].strip()}")
            
            return {
                'success': failed_tests == 0,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'results': self.test_results
            }
        
        finally:
            self.cleanup()
    
    def save_test_report(self, output_file: str):
        """Save detailed test report to file."""
        import json
        
        report_data = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results if r['success']),
                'failed_tests': sum(1 for r in self.test_results if not r['success']),
            },
            'test_results': self.test_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed test report saved to: {output_file}")


def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate layout_scorer.py functionality")
    parser.add_argument(
        '--script',
        default='layout_scorer.py',
        help='Path to layout_scorer.py script'
    )
    parser.add_argument(
        '--report',
        help='Save detailed test report to JSON file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    validator = LayoutScorerValidator(args.script, verbose=not args.quiet)
    results = validator.run_all_tests()
    
    if args.report:
        validator.save_test_report(args.report)
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())