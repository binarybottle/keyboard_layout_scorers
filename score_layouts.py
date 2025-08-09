#!/usr/bin/env python3
"""
Keyboard Layout Scorer using pre-computed score table.

A comprehensive tool for evaluating keyboard layouts using frequency-weighted scoring.
Scoring methods include engram, comfort, distance, time, and dvorak9.

Default behavior:
- Score table: input/score_table.csv
- Frequency data: input/english-letter-pair-counts-google-ngrams.csv  
- Scoring mode: Frequency-weighted (prioritizes common English letter combinations)
- Score mapping: Letter-pair frequencies → Key-pair scores (distance/time inverted)

Usage:
    # Single layout evaluation (all available scoring methods)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
    
    # Force raw (unweighted) scoring
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --raw
    
    # Compare multiple layouts  
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;"
    
    # Verbose output (shows both weighted and raw scores)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --verbose
    
    # Minimal CSV output for scripts/automation
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv-output
    
    # Single scoring method only
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --scorer distance
    
    # Save detailed results to file
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv results.csv
    
    # Custom files (when defaults aren't suitable)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-table custom.csv --frequency-file custom_freq.csv

Key features:
- Automatic frequency weighting using English bigram frequencies
- Smart score inversion (distance/time: higher is worse → lower scores)  
- Letter-pair → Key-pair mapping (e.g., "TH" frequency weights T→H key transition)
- Multiple output formats: detailed, CSV, minimal CSV, score-only
- Graceful fallback to raw scoring if frequency file missing
- Support for all scoring methods in the unified score table
"""

import sys
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class LayoutScorer:
    """Unified layout scorer using pre-computed score table."""
    
    def __init__(self, score_table_path: str, frequency_file: Optional[str] = None, use_raw: bool = False, verbose: bool = False):
        """Initialize scorer with score table and optional frequency data."""
        self.verbose = verbose
        self.use_raw = use_raw
        self.score_table = self._load_score_table(score_table_path)
        self.available_scorers = self._detect_available_scorers()
        
        # Load frequency data unless raw scoring is requested
        if frequency_file and not use_raw:
            self.bigram_frequencies = self._load_frequency_data(frequency_file)
        else:
            self.bigram_frequencies = None
        
        if self.verbose:
            print(f"Score table: {score_table_path}")
            if frequency_file:
                print(f"Frequency file: {frequency_file}")
            print(f"Loaded score table with {len(self.score_table)} key pairs")
            print(f"Available scorers: {', '.join(self.available_scorers)}")
            if self.use_raw:
                print("Using raw (unweighted) scoring only")
            elif self.bigram_frequencies is not None:
                print(f"Loaded frequency data for {len(self.bigram_frequencies)} bigrams")
                total_freq = sum(self.bigram_frequencies.values())
                print(f"Total frequency count: {total_freq:,}")
                print("Using frequency-weighted scoring by default")
            else:
                print("No frequency file found - using raw scoring")
    
    def _load_score_table(self, filepath: str) -> pd.DataFrame:
        """Load the unified score table."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Score table not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Error reading score table: {e}")
        
        if 'key_pair' not in df.columns:
            raise ValueError("Score table must have 'key_pair' column")
        
        # Set key_pair as index for fast lookup
        df = df.set_index('key_pair')
        return df
    
    def _load_frequency_data(self, filepath: str) -> Dict[str, float]:
        """Load bigram frequency data from CSV file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Frequency file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Error reading frequency file: {e}")
        
        # Try to detect column names
        possible_bigram_cols = ['bigram', 'pair', 'key_pair', 'letter_pair']
        possible_freq_cols = ['count', 'frequency', 'freq', 'occurrences']
        
        bigram_col = None
        freq_col = None
        
        for col in possible_bigram_cols:
            if col in df.columns:
                bigram_col = col
                break
        
        for col in possible_freq_cols:
            if col in df.columns:
                freq_col = col
                break
        
        if bigram_col is None:
            raise ValueError(f"Could not find bigram column. Expected one of: {possible_bigram_cols}")
        
        if freq_col is None:
            raise ValueError(f"Could not find frequency column. Expected one of: {possible_freq_cols}")
        
        if self.verbose:
            print(f"Using bigram column: '{bigram_col}', frequency column: '{freq_col}'")
        
        # Load frequencies
        frequencies = {}
        total_count = 0
        
        for _, row in df.iterrows():
            bigram = str(row[bigram_col]).strip().upper()
            freq = float(row[freq_col])
            
            if len(bigram) == 2:
                frequencies[bigram] = freq
                total_count += freq
        
        if self.verbose:
            print(f"Loaded {len(frequencies)} bigram frequencies")
            print(f"Total frequency count: {total_count:,}")
        
        return frequencies
    
    def _detect_available_scorers(self) -> List[str]:
        """Detect available scoring methods from table columns."""
        scorers = []
        
        # Look for normalized score columns
        for col in self.score_table.columns:
            if col.endswith('_normalized'):
                scorer_name = col.replace('_normalized', '').replace('_score', '')
                scorers.append(scorer_name)
        
        return sorted(list(set(scorers)))
    
    def _generate_letter_pairs(self, layout_mapping: Dict[str, str]) -> List[str]:
        """Generate all possible letter-pairs from layout letters."""
        letters = list(layout_mapping.keys())
        letter_pairs = []
        
        for letter1 in letters:
            for letter2 in letters:
                letter_pairs.append(letter1 + letter2)
        
        return letter_pairs
    
    def _score_layout_with_method(self, layout_mapping: Dict[str, str], scorer: str) -> Dict[str, float]:
        """Score a layout using a specific scoring method."""
        
        # Get column name for this scorer
        score_col = f"{scorer}_score_normalized"
        if score_col not in self.score_table.columns:
            # Try without _score suffix
            score_col = f"{scorer}_normalized"
            if score_col not in self.score_table.columns:
                raise ValueError(f"Scorer '{scorer}' not found in score table")
        
        # Determine if this scorer should be inverted (higher is worse)
        invert_scores = scorer in ['distance', 'time']
        
        # Generate all letter-pairs for this layout
        letter_pairs = self._generate_letter_pairs(layout_mapping)
        
        # Initialize scoring variables
        raw_total_score = 0.0
        raw_count = 0
        missing_pairs = []
        
        # Frequency-weighted scoring (if not using raw mode)
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        
        for letter_pair in letter_pairs:
            # Convert letter-pair to key-pair via layout mapping
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    key1 = layout_mapping[letter1]
                    key2 = layout_mapping[letter2]
                    key_pair = key1 + key2
                    
                    # Look up key-pair score in table
                    if key_pair in self.score_table.index:
                        raw_score = self.score_table.loc[key_pair, score_col]
                        
                        # Invert score if needed (distance, time: higher is worse)
                        if invert_scores:
                            score = 1.0 - raw_score
                        else:
                            score = raw_score
                        
                        # Raw scoring (treat all letter-pairs equally)
                        raw_total_score += score
                        raw_count += 1
                        
                        # Frequency-weighted scoring (if enabled)
                        if use_frequency:
                            frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                            weighted_total_score += score * frequency
                            total_frequency += frequency
                            if frequency > 0:
                                frequency_coverage += frequency
                    else:
                        missing_pairs.append(f"{letter_pair}→{key_pair}")
                else:
                    # Letter not in layout mapping
                    missing_pairs.append(f"{letter_pair}→??")
        
        # Calculate results
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        
        # Determine primary score based on mode
        if use_frequency:
            # Frequency weighting is primary
            primary_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
            results = {
                'average_score': primary_score,  # Primary score
                'total_score': weighted_total_score,
                'raw_average_score': raw_average,  # Secondary 
                'raw_total_score': raw_total_score,
                'pair_count': raw_count,
                'missing_pairs': len(missing_pairs),
                'coverage': raw_count / len(letter_pairs) if letter_pairs else 0.0,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            }
        else:
            # Raw scoring is primary
            results = {
                'average_score': raw_average,  # Primary score
                'total_score': raw_total_score,
                'pair_count': raw_count,
                'missing_pairs': len(missing_pairs),
                'coverage': raw_count / len(letter_pairs) if letter_pairs else 0.0
            }
        
        if self.verbose and missing_pairs:
            print(f"  Warning: {len(missing_pairs)} missing pairs for {scorer}")
            if len(missing_pairs) <= 5:
                print(f"    Missing: {missing_pairs}")
        
        return results
    
    def score_layout(self, layout_mapping: Dict[str, str], scorers: List[str]) -> Dict[str, Dict[str, float]]:
        """Score a layout using specified scoring methods."""
        
        results = {}
        
        for scorer in scorers:
            if scorer not in self.available_scorers:
                print(f"Warning: Scorer '{scorer}' not available. Available: {self.available_scorers}")
                continue
            
            try:
                results[scorer] = self._score_layout_with_method(layout_mapping, scorer)
            except Exception as e:
                print(f"Error scoring with {scorer}: {e}")
        
        return results
    
    def compare_layouts(self, layouts: Dict[str, Dict[str, str]], scorers: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare multiple layouts across scoring methods."""
        
        results = {}
        
        for layout_name, layout_mapping in layouts.items():
            if self.verbose:
                print(f"Scoring layout: {layout_name}")
            
            results[layout_name] = self.score_layout(layout_mapping, scorers)
        
        return results


def parse_layout_string(layout_str: str) -> Dict[str, str]:
    """Parse layout string into character -> position mapping."""
    # Handle format like "name:layout_string"
    if ':' in layout_str:
        name, layout = layout_str.split(':', 1)
        return layout.strip()
    else:
        return layout_str.strip()


def create_layout_mapping(letters: str, positions: str) -> Dict[str, str]:
    """Create mapping from letters to QWERTY positions."""
    if len(letters) != len(positions):
        raise ValueError(f"Letters ({len(letters)}) and positions ({len(positions)}) must have same length")
    
    mapping = {}
    for letter, position in zip(letters, positions):
        mapping[letter.upper()] = position.upper()
    
    return mapping


def parse_layout_compare(compare_args: List[str]) -> Dict[str, Dict[str, str]]:
    """Parse layout comparison arguments."""
    layouts = {}
    
    for arg in compare_args:
        if ':' not in arg:
            raise ValueError(f"Layout comparison format should be 'name:layout'. Got: {arg}")
        
        name, layout_str = arg.split(':', 1)
        name = name.strip()
        layout_str = layout_str.strip()
        
        # Create mapping from layout string to QWERTY positions
        # Assume layout string maps to standard QWERTY positions in order
        qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
        
        if len(layout_str) > len(qwerty_positions):
            raise ValueError(f"Layout '{name}' too long. Max {len(qwerty_positions)} characters.")
        
        mapping = {}
        for i, char in enumerate(layout_str):
            mapping[char.upper()] = qwerty_positions[i]
        
        layouts[name] = mapping
    
    return layouts


def print_results(results: Dict[str, float], format_type: str = 'detailed', scorer_name: str = '', use_raw: bool = False, verbose: bool = False):
    """Print scoring results."""
    
    if format_type == 'csv_output':
        # Minimal CSV output for programmatic use
        if use_raw or 'raw_average_score' not in results:
            print(f"{results['average_score']:.6f}")
        else:
            print(f"{results['average_score']:.6f},{results['raw_average_score']:.6f}")
        return
    
    if format_type == 'score_only':
        print(f"{results['average_score']:.6f}")
        return
    
    if format_type == 'csv':
        # Print CSV header if this is the first call
        if scorer_name:
            if use_raw:
                print("scorer,average_score,total_score,pair_count,coverage")
            else:
                print("scorer,average_score,total_score,raw_average_score,raw_total_score,pair_count,coverage,frequency_coverage")
        
        if use_raw:
            print(f"{scorer_name},{results['average_score']:.6f},{results['total_score']:.6f},"
                  f"{results['pair_count']},{results['coverage']:.6f}")
        else:
            print(f"{scorer_name},{results['average_score']:.6f},{results['total_score']:.6f},"
                  f"{results['raw_average_score']:.6f},{results['raw_total_score']:.6f},"
                  f"{results['pair_count']},{results['coverage']:.6f},{results['frequency_coverage']:.6f}")
        return
    
    # Detailed format
    if use_raw:
        print(f"Average bigram score: {results['average_score']:.6f}")
        print(f"Total score: {results['total_score']:.6f}")
    else:
        print(f"Frequency-weighted average bigram score: {results['average_score']:.6f}")
        #print(f"Frequency-weighted total score: {results['total_score']:.6f}")
        
        # Show raw scores if verbose or if they're significantly different
        if verbose or 'raw_average_score' in results:
            print(f"Raw average bigram score: {results['raw_average_score']:.6f}")
            #print(f"Raw total score: {results['raw_total_score']:.6f}")
    
    print(f"Pair count: {results['pair_count']}")
    print(f"Coverage (% letter-pairs with precomputed scores): {results['coverage']:.1%}")
    
    if not use_raw and 'frequency_coverage' in results:
        print(f"Frequency coverage (% English frequency that layout covers): {results['frequency_coverage']:.1%}")
    
    if results['missing_pairs'] > 0:
        print(f"Missing pairs: {results['missing_pairs']}")


def print_comparison_summary(comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
                           format_type: str = 'detailed', quiet: bool = False, use_raw: bool = False, verbose: bool = False):
    """Print summary of layout comparison."""
    
    if format_type == 'csv_output':
        # Minimal CSV output for programmatic use (no headers)
        for layout_name, layout_results in comparison_results.items():
            for scorer, results in layout_results.items():
                if use_raw or 'raw_average_score' not in results:
                    print(f"{layout_name},{scorer},{results['average_score']:.6f}")
                else:
                    print(f"{layout_name},{scorer},{results['average_score']:.6f},{results['raw_average_score']:.6f}")
        return
    
    if format_type == 'score_only':
        for layout_name, layout_results in comparison_results.items():
            for scorer, results in layout_results.items():
                print(f"{layout_name},{scorer},{results['average_score']:.6f}")
        return
    
    if format_type == 'csv':
        if use_raw:
            print("layout,scorer,average_score,total_score,pair_count,coverage")
        else:
            print("layout,scorer,average_score,total_score,raw_average_score,raw_total_score,pair_count,coverage,frequency_coverage")
        
        for layout_name, layout_results in comparison_results.items():
            for scorer, results in layout_results.items():
                if use_raw:
                    print(f"{layout_name},{scorer},{results['average_score']:.6f},{results['total_score']:.6f},"
                          f"{results['pair_count']},{results['coverage']:.6f}")
                else:
                    freq_cov = results.get('frequency_coverage', 0.0)
                    raw_avg = results.get('raw_average_score', results['average_score'])
                    raw_total = results.get('raw_total_score', results['total_score'])
                    print(f"{layout_name},{scorer},{results['average_score']:.6f},{results['total_score']:.6f},"
                          f"{raw_avg:.6f},{raw_total:.6f},"
                          f"{results['pair_count']},{results['coverage']:.6f},{freq_cov:.6f}")
        return
    
    # Detailed format
    if not quiet:
        print("\nComparison Summary:")
        print("=" * 70)
    
    # Group by scorer for easier comparison
    scorers = set()
    for layout_results in comparison_results.values():
        scorers.update(layout_results.keys())
    
    for scorer in sorted(scorers):
        if not quiet:
            score_type = "unweighted scores" if use_raw else "frequency-weighted scores"
            print(f"\n{scorer.upper()} {score_type}:")
            print("-" * 50)
        
        scorer_results = []
        for layout_name, layout_results in comparison_results.items():
            if scorer in layout_results:
                score = layout_results[scorer]['average_score']
                scorer_results.append((layout_name, score))
        
        # Sort by score (descending for better is higher)
        scorer_results.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (layout_name, score) in enumerate(scorer_results, 1):
            print(f"{rank:2d}. {layout_name:20s} {score:.6f}")
        
        # Show raw scores as secondary if using weighted and verbose
        if not use_raw and verbose and not quiet:
            raw_results = []
            for layout_name, layout_results in comparison_results.items():
                if scorer in layout_results and 'raw_average_score' in layout_results[scorer]:
                    raw_score = layout_results[scorer]['raw_average_score']
                    raw_results.append((layout_name, raw_score))
            
            if raw_results:
                print(f"\n{scorer.upper()} unweighted scores (for reference):")
                print("-" * 40)
                raw_results.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (layout_name, score) in enumerate(raw_results, 1):
                    print(f"{rank:2d}. {layout_name:20s} {score:.6f}")


def save_detailed_comparison_csv(comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
                               filename: str, layout_mappings: Dict[str, Dict[str, str]] = None, 
                               use_raw: bool = False):
    """Save detailed comparison results to CSV."""
    
    rows = []
    for layout_name, layout_results in comparison_results.items():
        for scorer, results in layout_results.items():
            row = {
                'layout_name': layout_name,
                'scorer': scorer,
                'average_score': results['average_score'],
                'total_score': results['total_score'],
                'pair_count': results['pair_count'],
                'coverage': results['coverage'],
                'missing_pairs': results.get('missing_pairs', 0)
            }
            
            # Add frequency-weighted vs raw details
            if not use_raw and 'raw_average_score' in results:
                row.update({
                    'raw_average_score': results['raw_average_score'],
                    'raw_total_score': results['raw_total_score'],
                    'total_frequency': results.get('total_frequency', 0),
                    'frequency_coverage': results.get('frequency_coverage', 0.0)
                })
            
            # Add layout mapping if available
            if layout_mappings and layout_name in layout_mappings:
                mapping = layout_mappings[layout_name]
                row['layout_letters'] = ''.join(sorted(mapping.keys()))
                row['layout_positions'] = ''.join(mapping[c] for c in sorted(mapping.keys()))
            
            rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Format floating point numbers
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted_row[key] = f"{value:.6f}"
                    else:
                        formatted_row[key] = value
                writer.writerow(formatted_row)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Unified keyboard layout scorer with automatic defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage (uses default files: input/score_table.csv and input/english-letter-pair-counts-google-ngrams.csv)
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
  
  # Raw (unweighted) scoring only
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --raw
  
  # Compare layouts with default frequency weighting
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl"
  
  # Use custom score table and frequency file
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-table custom_scores.csv --frequency-file custom_freqs.csv
  
  # Minimal CSV output for programmatic use (no headers, scores only)
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output
  
  # Verbose output (shows both weighted and raw scores)
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --verbose
  
  # Save to CSV file with frequency weighting
  python score_layouts.py --compare qwerty:"qwerty" dvorak:"',.py" --csv results.csv

Default behavior:
- Uses input/score_table.csv for scoring data
- Uses input/english-letter-pair-counts-google-ngrams.csv for frequency weighting (if it exists)
- Falls back to raw scoring if frequency file is not found
- With --raw: Ignores frequencies and uses raw (unweighted) scoring
- With --verbose: Shows both weighted and raw scores for comparison
- With --csv-output: Minimal CSV format for programmatic use (layout,scorer,weighted_score,raw_score)

Available scoring methods depend on the score table contents (e.g., distance, comfort, dvorak9, time).
Distance and time scores are automatically inverted (1-score) since higher values are worse.
        """
    )
    
    # Required arguments (now optional with defaults)
    parser.add_argument(
        '--score-table',
        default="input/score_table.csv",
        help="Path to unified score table CSV file (default: input/score_table.csv)"
    )
    
    # Optional frequency weighting (with default)
    parser.add_argument(
        '--frequency-file',
        default="input/english-letter-pair-counts-google-ngrams.csv",
        help="Path to bigram frequency CSV file (default: input/english-letter-pair-counts-google-ngrams.csv)"
    )
    
    # Raw scoring option
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Use raw (unweighted) scoring only, ignore frequency weighting"
    )
    
    # Minimal CSV output option
    parser.add_argument(
        '--csv-output',
        action='store_true',
        help="Output minimal CSV format to stdout (scores only, no headers, for programmatic use)"
    )
    
    # Scorer selection (mutually exclusive)
    scorer_group = parser.add_mutually_exclusive_group()
    scorer_group.add_argument(
        '--scorer',
        help="Run a single scorer (available scorers depend on table contents)"
    )
    scorer_group.add_argument(
        '--scorers',
        help="Run multiple scorers (comma-separated or 'all')"
    )
    scorer_group.add_argument(
        '--compare',
        nargs='+',
        help="Compare layouts (e.g., qwerty:qwertyuiop dvorak:',.pyfgcrl)"
    )
    
    # Layout definition (for single layout scoring)
    layout_group = parser.add_argument_group('Layout Definition')
    layout_group.add_argument(
        '--letters',
        help="String of characters in the layout"
    )
    layout_group.add_argument(
        '--positions', 
        help="String of corresponding QWERTY positions"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--format',
        choices=['detailed', 'csv', 'score_only'],
        default='detailed',
        help="Output format (default: detailed)"
    )
    output_group.add_argument(
        '--csv',
        help="Save detailed comparison to CSV file"
    )
    output_group.add_argument(
        '--score-only',
        action='store_true',
        help="Output only primary scores (minimal output)"
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress verbose output"
    )
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed processing information"
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Handle output format shortcuts
    if args.score_only:
        args.format = 'score_only'
    elif args.csv_output:
        args.format = 'csv_output'
    
    try:
        # Check if frequency file exists (only required if not using --raw)
        frequency_file = None
        if not args.raw:
            if Path(args.frequency_file).exists():
                frequency_file = args.frequency_file
            elif args.frequency_file != "input/english-letter-pair-counts-google-ngrams.csv":
                # User explicitly provided a frequency file that doesn't exist
                print(f"Error: Frequency file not found: {args.frequency_file}")
                return 1
            # If using default frequency file and it doesn't exist, just use raw scoring
        
        # Initialize scorer (suppress verbose output if using csv-output)
        suppress_verbose = args.format == 'csv_output'
        scorer = LayoutScorer(args.score_table, frequency_file, args.raw, 
                             verbose=args.verbose and not suppress_verbose)
        
        if args.compare:
            # Layout comparison mode
            layouts = parse_layout_compare(args.compare)
            
            # Determine which scorers to run
            if args.scorer:
                scorers = [args.scorer] if args.scorer in scorer.available_scorers else []
            elif args.scorers:
                if args.scorers.lower() == 'all':
                    scorers = scorer.available_scorers
                else:
                    requested = [s.strip() for s in args.scorers.split(',')]
                    scorers = [s for s in requested if s in scorer.available_scorers]
                    invalid = [s for s in requested if s not in scorer.available_scorers]
                    if invalid:
                        print(f"Warning: Unknown scorers ignored: {invalid}")
            else:
                scorers = scorer.available_scorers  # Default to all
            
            if not scorers:
                print("Error: No valid scorers specified")
                return 1
            
            if not args.quiet and args.format != 'csv_output':
                print(f"Comparing {len(layouts)} layouts using {len(scorers)} scoring methods...")
                print(f"Layouts: {', '.join(layouts.keys())}")
                print(f"Scorers: {', '.join(scorers)}")
                if args.raw:
                    print("Using raw (unweighted) scoring")
                elif scorer.bigram_frequencies:
                    print("Using frequency-weighted scoring")
                else:
                    print("Using raw scoring (frequency file not found)")
            
            # Run comparison
            results = scorer.compare_layouts(layouts, scorers)
            
            # Handle output
            if args.csv:
                save_detailed_comparison_csv(results, args.csv, layouts, args.raw)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Detailed comparison saved to: {args.csv}")
            else:
                if not args.quiet and args.format != 'csv_output':
                    print(f"\n=== RESULTS ===")
                print_comparison_summary(results, args.format, args.quiet, args.raw, args.verbose)

        else:
            # Single layout mode
            if not args.letters or not args.positions:
                print("Error: Must specify --letters and --positions for single layout scoring")
                return 1
            
            # Create layout mapping
            try:
                layout_mapping = create_layout_mapping(args.letters, args.positions)
            except Exception as e:
                print(f"Error creating layout mapping: {e}")
                return 1
            
            # Determine which scorers to run
            if args.scorer:
                scorers = [args.scorer] if args.scorer in scorer.available_scorers else []
            elif args.scorers:
                if args.scorers.lower() == 'all':
                    scorers = scorer.available_scorers
                else:
                    requested = [s.strip() for s in args.scorers.split(',')]
                    scorers = [s for s in requested if s in scorer.available_scorers]
                    invalid = [s for s in requested if s not in scorer.available_scorers]
                    if invalid:
                        print(f"Warning: Unknown scorers ignored: {invalid}")
            else:
                scorers = scorer.available_scorers  # Default to all
            
            if not scorers:
                print("Error: No valid scorers specified")
                return 1
            
            # Create layout name for display
            layout_name = f"{args.letters} → {args.positions}"
            
            # Run scoring
            results = scorer.score_layout(layout_mapping, scorers)
            
            # Handle output
            if args.csv:
                # Convert to comparison format
                comparison_results = {layout_name: results}
                layout_mappings_for_csv = {layout_name: layout_mapping}
                save_detailed_comparison_csv(comparison_results, args.csv, layout_mappings_for_csv, args.raw)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Results saved to: {args.csv}")
            else:
                # Print to stdout
                if len(scorers) == 1:
                    # Single scorer
                    scorer_name = scorers[0]
                    
                    if not args.quiet and args.format != 'csv_output':
                        print(f"\nLayout: {layout_name}")
                        print(f"{scorer_name.upper()} results:")
                        print("=" * 50)
                    print_results(results[scorer_name], args.format, scorer_name, args.raw, args.verbose)
                else:
                    # Multiple scorers
                    comparison_results = {layout_name: results}
                    print_comparison_summary(comparison_results, args.format, args.quiet, args.raw, args.verbose)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if not Path(args.score_table).exists():
            print(f"Score table file not found: {args.score_table}")
            print("Make sure the file exists or specify a different path with --score-table")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())