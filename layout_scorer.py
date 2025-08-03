"""
Unified Keyboard Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

A comprehensive tool for evaluating keyboard layouts using multiple scoring methods.
This unified manager can run individual scorers or compare layouts across all methods.

Features:
- Run individual scorers (distance, dvorak9, engram)
- Compare multiple scoring methods on the same layout
- Compare multiple layouts across scoring methods
- Automatic cross-hand filtering comparison (shows both filtered and unfiltered results)

Scoring methods available:
- Distance scorer: Physical finger travel distance analysis
- Dvorak9 scorer: Four scoring approaches (pure, frequency-weighted, speed-weighted, comfort-weighted)
- Engram scorer: Two scoring modes (32-key full layout, 24-key home block only)

# Basic scoring (shows both full and cross-hand filtered results)
python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"
python layout_scorer.py --scorer dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Multiple methods with automatic cross-hand filtering comparison
python layout_scorer.py --scorers engram,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"

# Compare layouts (shows full and filtered results)
python layout_scorer.py --compare qwerty:"qwertyuiopasdfghjkl;zxcvbnm,./['" dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz['" --text "hello"

"""

import sys
import argparse
from typing import Dict, Any, List, Optional, Set

# Import framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config, get_config_loader
from framework.layout_utils import create_layout_mapping, filter_to_letters_only, parse_layout_compare
from framework.output_utils import print_results, save_detailed_comparison_csv, print_comparison_summary
from framework.cli_utils import handle_common_errors, get_layout_from_args
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
  
  # With cross-hand filtering (available for all scorers)
  python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --ignore-cross-hand
  
  # Compare layouts (layout string maps to QWERTY positions)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --text "hello"
  
  # Save detailed comparison to CSV file
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --csv results.csv --text "hello"

Scoring methods:
  - distance: Physical finger travel distance analysis
  - dvorak9: Four approaches (pure, frequency-weighted, speed-weighted, comfort-weighted)  
  - engram: Two modes (32-key full layout, 24-key home block only)

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
            
            # Run comparison WITHOUT cross-hand filtering
            if not args.quiet:
                print(f"\n=== FULL LAYOUT COMPARISON ===")
            
            full_results = unified_scorer.compare_layouts(
                layouts, scorers,
                text=text,
                ignore_cross_hand=False,
                quiet=args.quiet
            )
            
            # Run comparison WITH cross-hand filtering
            if not args.quiet:
                print(f"\n=== CROSS-HAND FILTERED COMPARISON ===")
            
            filtered_results = unified_scorer.compare_layouts(
                layouts, scorers,
                text=text,
                ignore_cross_hand=True,
                quiet=args.quiet
            )
            
            # Rename results to distinguish them
            full_results_renamed = {}
            filtered_results_renamed = {}
            
            # IMPORTANT: Create a mapping for CSV output
            layout_mappings_for_csv = {}
            
            for layout_name, layout_results in full_results.items():
                full_results_renamed[f"full_{layout_name}"] = layout_results
                # Store the original layout mapping for CSV
                if layout_name in layouts:
                    layout_mappings_for_csv[f"full_{layout_name}"] = layouts[layout_name]
                
            for layout_name, layout_results in filtered_results.items():
                filtered_results_renamed[f"no_crosshand_{layout_name}"] = layout_results
                # Store the original layout mapping for CSV
                if layout_name in layouts:
                    layout_mappings_for_csv[f"no_crosshand_{layout_name}"] = layouts[layout_name]
            
            # Combine all results for output
            combined_results = {**full_results_renamed, **filtered_results_renamed}
            
            # Handle CSV output - NOW WITH LAYOUT MAPPINGS
            if args.csv:
                save_detailed_comparison_csv(combined_results, args.csv, layout_mappings_for_csv)
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
            
            # Get layout mapping from arguments
            letters, positions, layout_mapping = get_layout_from_args(args)
            
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
            
            # Create filtered layout name for display
            filtered_chars = ''.join(sorted(layout_mapping.keys()))
            filtered_positions = ''.join(layout_mapping[c] for c in sorted(layout_mapping.keys()))
            layout_name_base = f"{filtered_chars} → {filtered_positions}"
            
            # Run scoring WITHOUT cross-hand filtering
            results_full = unified_scorer.score_layout(
                layout_mapping, scorers,
                text=text,
                ignore_cross_hand=False,
                quiet=args.quiet
            )
            
            # Run scoring WITH cross-hand filtering
            results_filtered = unified_scorer.score_layout(
                layout_mapping, scorers,
                text=text,
                ignore_cross_hand=True,
                quiet=args.quiet
            )
            
            # Handle CSV output for single layout
            if args.csv:
                # Convert single layout results to comparison format with both versions
                filtered_chars = ''.join(sorted(layout_mapping.keys()))
                filtered_positions = ''.join(layout_mapping[c] for c in sorted(layout_mapping.keys()))
                layout_name_base = f"{filtered_chars} → {filtered_positions}"
                
                comparison_results = {
                    f"full_{layout_name_base}": results_full,
                    f"no_crosshand_{layout_name_base}": results_filtered
                }
                
                # Create layout mappings for CSV
                layout_mappings_for_csv = {
                    f"full_{layout_name_base}": layout_mapping,
                    f"no_crosshand_{layout_name_base}": layout_mapping
                }
                
                save_detailed_comparison_csv(comparison_results, args.csv, layout_mappings_for_csv)
                if not args.quiet:
                    print(f"Detailed results saved to: {args.csv}")
            else:
                # Print results to stdout
                if len(results_full) == 1:
                    # Single scorer output - show both versions
                    scorer_name = list(results_full.keys())[0]
                    
                    print(f"full_{layout_name_base}")
                    if not args.quiet:
                        print(f"\n{scorer_name.replace('_', ' ').capitalize()} results")
                        print("=" * 70)
                    print_results(results_full[scorer_name], args.format)
                    
                    print(f"\nfiltered_{layout_name_base}")
                    if not args.quiet:
                        print(f"\n{scorer_name.replace('_', ' ').capitalize()} results (cross-hand filtered)")
                        print("=" * 70)
                    print_results(results_filtered[scorer_name], args.format)
                else:
                    # Multiple scorer output
                    comparison_results = {
                        f"full_{layout_name_base}": results_full,
                        f"filtered_{layout_name_base}": results_filtered
                    }
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