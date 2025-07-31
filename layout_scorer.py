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
- Cross-hand filtering support for all scorers
- Common-keys-only comparison for layout comparisons

Scoring methods available:
- Distance scorer: Physical finger travel distance analysis
- Dvorak9 scorer: Four scoring approaches (pure, frequency-weighted, speed-weighted, comfort-weighted)
- Engram scorer: Two scoring modes (32-key full layout, 24-key home block only)

# Exactly like distance_scorer.py
python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

# Exactly like dvorak9_scorer.py (all 4 approaches)
python layout_scorer.py --scorer dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Exactly like engram_scorer.py (both 32-key and 24-key results)
python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# With cross-hand filtering (available for all scorers)
python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello" --ignore-cross-hand

# Run all scorers and get ranking table
python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

# Run specific subset
python layout_scorer.py --scorers distance,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"

# Compare QWERTY vs Dvorak across all methods (shows both full and common-keys-only results)
python layout_scorer.py --compare qwerty:"qwertyuiopasdfghjklzxcvbnm" dvorak:"',.pyfgcrlaoeuidhtnsqjkxbmwvz" --text "hello"

"""

import sys
import argparse
from typing import Dict, Any, List, Optional, Set

# Import framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config, get_config_loader
from framework.layout_utils import create_layout_mapping, filter_to_letters_only, parse_layout_compare
from framework.output_utils import print_results, save_detailed_comparison_csv, print_comparison_summary
from framework.cli_utils import handle_common_errors
from framework.scorer_factory import ScorerFactory
from framework.unified_scorer import UnifiedLayoutScorer


def find_common_keys(layouts: Dict[str, Dict[str, str]]) -> Set[str]:
    """
    Find keys that are common across all layouts.
    
    Args:
        layouts: Dict mapping layout names to character mappings
        
    Returns:
        Set of characters present in all layouts
    """
    if not layouts:
        return set()
    
    # Start with keys from first layout
    common_keys = set(list(layouts.values())[0].keys())
    
    # Intersect with keys from all other layouts
    for layout_mapping in layouts.values():
        common_keys &= set(layout_mapping.keys())
    
    return common_keys


def filter_layouts_to_common_keys(layouts: Dict[str, Dict[str, str]], 
                                common_keys: Set[str]) -> Dict[str, Dict[str, str]]:
    """
    Filter all layouts to only include common keys.
    
    Args:
        layouts: Dict mapping layout names to character mappings
        common_keys: Set of keys to keep
        
    Returns:
        Dict with filtered layout mappings
    """
    filtered_layouts = {}
    
    for name, layout_mapping in layouts.items():
        filtered_mapping = {char: pos for char, pos in layout_mapping.items() 
                          if char in common_keys}
        filtered_layouts[name] = filtered_mapping
    
    return filtered_layouts


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
  
  # With cross-hand filtering (available for all scorers)
  python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --ignore-cross-hand
  
  # Compare layouts (layout string maps to QWERTY positions)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --text "hello"
  
  # Save detailed comparison to CSV file (includes both full and common-keys-only results)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --csv results.csv --text "hello"

Scoring methods:
  - distance: Physical finger travel distance analysis
  - dvorak9: Four approaches (pure, frequency-weighted, speed-weighted, comfort-weighted)  
  - engram: Two modes (32-key full layout, 24-key home block only)

Note: Layout comparison automatically shows both full layout results and common-keys-only results.
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
    
    # Scorer options
    options_group = parser.add_argument_group('Scorer Options')
    options_group.add_argument(
        '--ignore-cross-hand',
        action='store_true',
        help="Ignore bigrams that cross hands (available for all scorers)"
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
            
            if not args.quiet:
                print(f"Comparing {len(layouts)} layouts using {len(scorers)} scoring methods...")
                layout_names = list(layouts.keys())
                print(f"Layouts: {', '.join(layout_names)}")
                print(f"Scorers: {', '.join(scorers)}")
            
            # Run comparison on full layouts
            if not args.quiet:
                print(f"\n=== FULL LAYOUT COMPARISON ===")
            
            full_results = unified_scorer.compare_layouts(
                layouts, scorers,
                text=text,
                ignore_cross_hand=args.ignore_cross_hand,
                quiet=args.quiet
            )
            
            # Find common keys and run comparison on common keys only
            common_keys = find_common_keys(layouts)
            
            if not args.quiet:
                print(f"\n=== COMMON KEYS ONLY COMPARISON ===")
                print(f"Common keys ({len(common_keys)}): {''.join(sorted(common_keys))}")
            
            if len(common_keys) > 0:
                filtered_layouts = filter_layouts_to_common_keys(layouts, common_keys)
                common_results = unified_scorer.compare_layouts(
                    filtered_layouts, scorers,
                    text=text,
                    ignore_cross_hand=args.ignore_cross_hand,
                    quiet=args.quiet
                )
                
                # Rename results to indicate common-keys-only
                common_results_renamed = {}
                for layout_name, layout_results in common_results.items():
                    common_results_renamed[f"{layout_name}_common"] = layout_results
            else:
                common_results_renamed = {}
                if not args.quiet:
                    print("No common keys found across all layouts.")
            
            # Combine results for output
            combined_results = {**full_results, **common_results_renamed}
            
            # Handle CSV output
            if args.csv:
                save_detailed_comparison_csv(combined_results, args.csv)
                if not args.quiet:
                    print(f"Detailed comparison saved to: {args.csv}")
            else:
                # Print comparison results to stdout
                if not args.quiet:
                    print(f"\n=== RESULTS SUMMARY ===")
                print_comparison_summary(combined_results, args.format, args.quiet)

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