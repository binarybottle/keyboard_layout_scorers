#!/usr/bin/env python3
"""
Unified Keyboard Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

A comprehensive tool for evaluating keyboard layouts using multiple scoring methods.
This unified manager can run individual scorers or compare layouts across all methods.

Features:
- Run individual scorers (distance, dvorak9, engram)
- Compare multiple scoring methods on the same layout
- Compare multiple layouts across scoring methods
- Consistent CLI interface across all scoring methods

# Exactly like distance_scorer.py
python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

# Exactly like dvorak9_scorer.py  
python layout_scorer.py --scorer dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Exactly like engram_scorer.py
python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Run all scorers and get ranking table
python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

# Run specific subset
python layout_scorer.py --scorers distance,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"

# Compare QWERTY vs Dvorak across all methods
python layout_scorer.py --compare qwerty:"qwertyuiopasdfghjklzxcvbnm" dvorak:"',.pyfgcrlaoeuidhtnsqjkxbmwvz" --text "hello"

"""

import sys
import argparse
from typing import Dict, Any, List, Optional

# Import framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config, get_config_loader
from framework.layout_utils import create_layout_mapping, filter_to_letters_only, parse_layout_compare
from framework.output_utils import print_results, save_detailed_comparison_csv, print_comparison_summary
from framework.cli_utils import handle_common_errors
from framework.scorer_factory import ScorerFactory
from framework.unified_scorer import UnifiedLayoutScorer


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for the unified scorer."""
    
    parser = argparse.ArgumentParser(
        description="Unified keyboard layout scorer - evaluate layouts using multiple methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single scorer (same output as individual script)
  python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"
  
  # Multiple scorers with comparison
  python layout_scorer.py --scorers distance,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"
  
  # All scorers
  python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"
  
  # Compare layouts (layout string maps to QWERTY positions)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --text "hello"
  
  # Save detailed comparison to CSV file
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --csv results.csv --text "hello"

  # With specific options
  python layout_scorer.py --scorer engram --letters "abc" --positions "ABC" --ignore-cross-hand
  python layout_scorer.py --scorer dvorak9 --letters "abc" --positions "ABC" --weights "input/dvorak9/speed_weights.csv"
        """
    )
    
    # Scorer selection (mutually exclusive)
    scorer_group = parser.add_mutually_exclusive_group(required=True)
    scorer_group.add_argument(
        '--scorer',
        choices=['distance', 'dvorak9', 'engram'],
        help="Run a single scorer"
    )
    scorer_group.add_argument(
        '--scorers',
        help="Run multiple scorers (comma-separated or 'all')"
    )
    scorer_group.add_argument(
        '--compare',
        nargs='+',
        help="Compare layouts where layout string maps to QWERTY positions (e.g., qwerty:qwertyuiop dvorak:',.pyfgcrl)"
    )
    
    # Layout definition (for single layout scoring)
    layout_group = parser.add_argument_group('Layout Definition')
    layout_group.add_argument(
        '--letters',
        help="String of characters in the layout (e.g., 'etaoinshrlcu')"
    )
    layout_group.add_argument(
        '--positions', 
        help="String of corresponding QWERTY positions (e.g., 'FDESGJWXRTYZ')"
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--text',
        help="Text to analyze (for distance scorer)"
    )
    input_group.add_argument(
        '--text-file',
        help="Path to text file to analyze (for distance scorer)"
    )
    input_group.add_argument(
        '--config',
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    # Scorer-specific options
    options_group = parser.add_argument_group('Scorer Options')
    options_group.add_argument(
        '--weights',
        help="Path to empirical weights CSV file (for dvorak9 scorer)"
    )
    options_group.add_argument(
        '--ignore-cross-hand',
        action='store_true',
        help="Ignore bigrams that cross hands (for engram scorer)"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--format', '--output-format',
        choices=['detailed', 'csv', 'score_only'],
        default='detailed',
        help="Output format (default: detailed)"
    )
    output_group.add_argument(
        '--csv',
        help="Save detailed comparison to CSV file (provide filename)"
    )
    output_group.add_argument(
        '--score-only',
        action='store_true',
        help="Output only scores"
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress verbose output"
    )
    
    return parser


@handle_common_errors
def main() -> int:
    """Main entry point for the unified layout scorer."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Handle output format shortcuts
    if args.score_only:
        args.format = 'score_only'
    
    # Validate text input for distance scorer
    text = None
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: Text file not found: {args.text_file}")
            return 1
    elif args.text:
        text = args.text
    
    # Create unified scorer
    try:
        unified_scorer = UnifiedLayoutScorer(args.config)
    except Exception as e:
        print(f"Error initializing scorer: {e}")
        return 1
    
    try:
        if args.compare:
            # Layout comparison mode
            layouts = parse_layout_compare(args.compare)
            
            # Determine which scorers to run
            if 'distance' in [s for s in unified_scorer.factory.get_available_scorers() if text]:
                scorers = ['distance', 'dvorak9', 'engram'] if text else ['dvorak9', 'engram']
            else:
                scorers = ['dvorak9', 'engram']
            
            # Run comparison
            results = unified_scorer.compare_layouts(
                layouts, scorers,
                text=text,
                weights=args.weights,
                ignore_cross_hand=args.ignore_cross_hand,
                quiet=args.quiet
            )
            
            # Handle CSV output
            if args.csv:
                save_detailed_comparison_csv(results, args.csv)
                if not args.quiet:
                    print(f"Detailed comparison saved to: {args.csv}")
            else:
                # Print comparison results to stdout
                print_comparison_summary(results, args.format, args.quiet)

        else:
            # Single layout mode
            if not args.letters or not args.positions:
                print("Error: Must specify --letters and --positions for single layout scoring")
                return 1
            
            # Create layout mapping
            layout_mapping = create_layout_mapping(args.letters, args.positions)
            layout_mapping = filter_to_letters_only(layout_mapping)
            
            if not layout_mapping:
                print("Error: No letters found in layout")
                return 1
            
            # Determine which scorers to run
            if args.scorer:
                scorers = [args.scorer]
            elif args.scorers:
                if args.scorers.lower() == 'all':
                    scorers = unified_scorer.factory.get_available_scorers()
                else:
                    scorers = [s.strip() for s in args.scorers.split(',')]
                    
                    # Validate scorer names
                    available = unified_scorer.factory.get_available_scorers()
                    invalid = [s for s in scorers if s not in available]
                    if invalid:
                        print(f"Error: Unknown scorers: {invalid}. Available: {available}")
                        return 1
            
            # Run scoring
            results = unified_scorer.score_layout(
                layout_mapping, scorers,
                text=text,
                weights=args.weights,
                ignore_cross_hand=args.ignore_cross_hand,
                quiet=args.quiet
            )
            
            # Handle CSV output for single layout
            if args.csv:
                # Convert single layout results to comparison format
                layout_name = f"{args.letters} → {args.positions}"
                comparison_results = {layout_name: results}
                save_detailed_comparison_csv(comparison_results, args.csv)
                if not args.quiet:
                    print(f"Detailed results saved to: {args.csv}")
            else:
                # Print results to stdout
                if len(results) == 1:
                    # Single scorer output
                    result = list(results.values())[0]
                    print_results(result, args.format)
                else:
                    # Multiple scorer output
                    layout_name = f"{args.letters} → {args.positions}"
                    comparison_results = {layout_name: results}
                    print_comparison_summary(comparison_results, args.format, args.quiet)
                            
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())