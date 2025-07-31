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
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# Import framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config, get_config_loader
from framework.layout_utils import create_layout_mapping, filter_to_letters_only, validate_layout_mapping
from framework.output_utils import print_results, format_comparison_output, create_summary_table
from framework.cli_utils import handle_common_errors

# Import individual scorers
from distance_scorer import DistanceScorer
from dvorak9_scorer import Dvorak9Scorer  
from engram_scorer import EngramScorer


class ScorerFactory:
    """Factory for creating scorer instances."""
    
    SCORERS = {
        'distance': DistanceScorer,
        'dvorak9': Dvorak9Scorer,
        'engram': EngramScorer,
    }
    
    @classmethod
    def create_scorer(cls, scorer_name: str, layout_mapping: Dict[str, str], 
                     config: Dict[str, Any]) -> BaseLayoutScorer:
        """
        Create a scorer instance.
        
        Args:
            scorer_name: Name of scorer ('distance', 'dvorak9', 'engram')
            layout_mapping: Character to position mapping
            config: Configuration dictionary
            
        Returns:
            Configured scorer instance
            
        Raises:
            ValueError: If scorer_name is not recognized
        """
        if scorer_name not in cls.SCORERS:
            available = list(cls.SCORERS.keys())
            raise ValueError(f"Unknown scorer '{scorer_name}'. Available: {available}")
        
        scorer_class = cls.SCORERS[scorer_name]
        return scorer_class(layout_mapping, config)
    
    @classmethod
    def get_available_scorers(cls) -> List[str]:
        """Get list of available scorer names."""
        return list(cls.SCORERS.keys())


class UnifiedLayoutScorer:
    """
    Unified manager for running and comparing keyboard layout scorers.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the unified scorer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_loader = get_config_loader(config_path)
        self.factory = ScorerFactory()
    
    def score_layout(self, layout_mapping: Dict[str, str], 
                    scorers: List[str],
                    **kwargs) -> Dict[str, ScoreResult]:
        """
        Score a layout using specified scorers.
        
        Args:
            layout_mapping: Character to position mapping
            scorers: List of scorer names to run
            **kwargs: Additional arguments (text, weights, etc.)
            
        Returns:
            Dict mapping scorer names to their results
        """
        results = {}
        
        for scorer_name in scorers:
            try:
                # Load scorer-specific configuration
                config = load_scorer_config(f'{scorer_name}_scorer')
                
                # Add scorer-specific arguments
                if scorer_name == 'distance' and 'text' in kwargs:
                    config['text'] = kwargs['text']
                elif scorer_name == 'dvorak9' and 'weights' in kwargs:
                    config['weights_file'] = kwargs['weights']
                elif scorer_name == 'engram' and 'ignore_cross_hand' in kwargs:
                    if 'scoring_options' not in config:
                        config['scoring_options'] = {}
                    config['scoring_options']['ignore_cross_hand'] = kwargs['ignore_cross_hand']
                
                # Add quiet mode
                config['quiet_mode'] = kwargs.get('quiet', False)
                
                # Create and run scorer
                scorer = self.factory.create_scorer(scorer_name, layout_mapping, config)
                result = scorer.score_layout()
                results[scorer_name] = result
                
            except Exception as e:
                if not kwargs.get('quiet', False):
                    print(f"Warning: {scorer_name} scorer failed: {e}")
                # Create empty result for failed scorer
                results[scorer_name] = ScoreResult(
                    primary_score=0.0,
                    metadata={'error': str(e), 'scorer_failed': True}
                )
        
        return results
    
    def compare_layouts(self, layouts: Dict[str, Dict[str, str]], 
                       scorers: List[str],
                       **kwargs) -> Dict[str, Dict[str, ScoreResult]]:
        """
        Compare multiple layouts across specified scorers.
        
        Args:
            layouts: Dict mapping layout names to their character mappings
            scorers: List of scorer names to run
            **kwargs: Additional arguments
            
        Returns:
            Nested dict: {layout_name: {scorer_name: ScoreResult}}
        """
        results = {}
        
        for layout_name, layout_mapping in layouts.items():
            if not kwargs.get('quiet', False):
                print(f"Scoring layout: {layout_name}")
            
            results[layout_name] = self.score_layout(layout_mapping, scorers, **kwargs)
        
        return results


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
  
  # CSV comparison table
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --csv-compare --text "hello"
  
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
        choices=['detailed', 'csv', 'score_only', 'comparison'],
        default='detailed',
        help="Output format (default: detailed)"
    )
    output_group.add_argument(
        '--csv',
        action='store_true',
        help="Output in CSV format"
    )
    output_group.add_argument(
        '--csv-compare',
        action='store_true',
        help="Output comparison in CSV table format (layout vs scorer matrix)"
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


def parse_layout_compare(compare_args: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse layout comparison arguments.
    
    Args:
        compare_args: List of "name:layout" strings where layout is the actual layout
        
    Returns:
        Dict mapping layout names to character mappings
        
    Example:
        --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl"
        Maps: q→Q, w→W, e→E, etc. and '→Q, ,→W, .→E, etc.
    """
    layouts = {}
    
    # Standard QWERTY positions for mapping
    standard_positions = "QWERTYUIOP[ASDFGHJKL;'ZXCVBNM,./"
    
    for arg in compare_args:
        if ':' not in arg:
            raise ValueError(f"Invalid compare format: '{arg}'. Expected 'name:layout'")
        
        name, layout_str = arg.split(':', 1)
        
        # The layout string IS the layout - map each character to corresponding QWERTY position
        if len(layout_str) > len(standard_positions):
            # Truncate if too long
            layout_str = layout_str[:len(standard_positions)]
        
        # Create mapping: layout_char → QWERTY_position
        layout_mapping = {}
        for i, char in enumerate(layout_str):
            if i < len(standard_positions):
                layout_mapping[char.lower()] = standard_positions[i]
        
        # Filter to letters only
        layout_mapping = filter_to_letters_only(layout_mapping)
        
        layouts[name] = layout_mapping
    
    return layouts


def print_single_result(result: ScoreResult, output_format: str, quiet: bool = False) -> None:
    """Print result from a single scorer."""
    if output_format == 'comparison':
        output_format = 'detailed'  # Fall back for single results
    
    print_results(result, output_format)


def print_multiple_results(results: Dict[str, ScoreResult], 
                         layout_name: str,
                         output_format: str,
                         quiet: bool = False) -> None:
    """Print results from multiple scorers."""
    if output_format == 'csv':
        # CSV output for multiple scorers - comparison table
        print("scorer,primary_score,description")
        for scorer_name, result in results.items():
            if not result.metadata.get('scorer_failed', False):
                description = {
                    'distance': 'normalized distance score (higher=better)',
                    'dvorak9': 'frequency-weighted dvorak principles (0-1)',
                    'engram': 'frequency-comfort combination score'
                }.get(scorer_name, 'layout score')
                print(f"{scorer_name},{result.primary_score:.6f},{description}")
    
    elif output_format == 'score_only':
        # Score-only output
        for scorer_name, result in results.items():
            if not result.metadata.get('scorer_failed', False):
                print(f"{scorer_name}: {result.primary_score:.6f}")
    
    elif output_format in ['detailed', 'comparison']:
        # Detailed comparison output
        if not quiet:
            print(f"\nLayout Scoring Results")
            if layout_name:
                print(f"layout: {layout_name}")
            print("=" * 60)
        
        # Summary table (no ranking since scores aren't comparable)
        print(f"\n{'scorer':<15} {'primary score':<15} {'description':<30}")
        print("-" * 65)
        
        for scorer_name, result in results.items():
            if result.metadata.get('scorer_failed', False):
                print(f"{scorer_name:<15} {'failed':<15} {'error':<30}")
            else:
                description = {
                    'distance': 'distance (higher=better)',
                    'dvorak9': 'dvorak principles (0-1)',
                    'engram': 'frequency-comfort combo'
                }.get(scorer_name, 'layout score')
                print(f"{scorer_name:<15} {result.primary_score:<15.6f} {description:<30}")
        
        # Detailed breakdown if requested
        if output_format == 'detailed' and not quiet:
            for scorer_name, result in results.items():
                if not result.metadata.get('scorer_failed', False):
                    print(f"\n{'-' * 20} {scorer_name} details {'-' * 20}")
                    print_results(result, 'detailed')


@handle_common_errors
def main() -> int:
    """Main entry point for the unified layout scorer."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Handle output format shortcuts
    if args.csv:
        args.format = 'csv'
    elif args.csv_compare:
        args.format = 'csv_compare'
    elif args.score_only:
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
            
            # Print comparison results
            if args.format == 'csv':
                print("layout,scorer,primary_score")
                for layout_name, layout_results in results.items():
                    for scorer_name, result in layout_results.items():
                        if not result.metadata.get('scorer_failed', False):
                            print(f"{layout_name},{scorer_name},{result.primary_score:.6f}")
            elif args.format == 'csv_compare':
                # Create CSV comparison table (layout vs scorer matrix)
                layout_names = list(results.keys())
                scorer_names = list(unified_scorer.factory.get_available_scorers())
                
                # Header
                header = "layout," + ",".join(scorer_names)
                print(header)
                
                # Data rows
                for layout_name in layout_names:
                    row = [layout_name]
                    layout_results = results[layout_name]
                    for scorer_name in scorer_names:
                        if scorer_name in layout_results and not layout_results[scorer_name].metadata.get('scorer_failed', False):
                            row.append(f"{layout_results[scorer_name].primary_score:.6f}")
                        else:
                            row.append("N/A")
                    print(",".join(row))
            else:
                for layout_name, layout_results in results.items():
                    print_multiple_results(layout_results, layout_name, args.format, args.quiet)
        
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
            
            # Print results
            if len(results) == 1:
                # Single scorer output (same as individual scripts)
                result = list(results.values())[0]
                print_single_result(result, args.format, args.quiet)
            else:
                # Multiple scorer output with comparison
                layout_name = f"{args.letters} → {args.positions}"
                print_multiple_results(results, layout_name, args.format, args.quiet)
        
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())