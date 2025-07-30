#!/usr/bin/env python3
"""
CLI utilities for keyboard layout scoring.

Common functions for command-line argument parsing, help text generation,
and standardized CLI interfaces across all scorers.
"""

import argparse
import sys
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from framework.config_loader import get_config_loader
from framework.layout_utils import create_layout_mapping, validate_layout_mapping


class StandardCLIParser:
    """
    Standardized command-line argument parser for all keyboard layout scorers.
    
    Provides consistent argument handling and help text across different scoring methods.
    """
    
    def __init__(self, scorer_name: str, config_path: str = "config.yaml"):
        """
        Initialize the CLI parser for a specific scorer.
        
        Args:
            scorer_name: Name of the scorer (e.g., 'distance_scorer')
            config_path: Path to configuration file
        """
        self.scorer_name = scorer_name
        self.config_loader = get_config_loader(config_path)
        
        # Load scorer configuration
        try:
            self.scorer_config = self.config_loader.get_scorer_config(scorer_name)
            self.cli_config = self.config_loader.get_cli_config()
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self.scorer_config = {}
            self.cli_config = {}
        
        # Create argument parser
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with standard and scorer-specific arguments."""
        
        # Get description from config
        description = self.scorer_config.get('description', f'{self.scorer_name} for keyboard layouts')
        method = self.scorer_config.get('method', 'Keyboard layout scoring method')
        
        epilog = self._generate_epilog()
        
        parser = argparse.ArgumentParser(
            description=f"{description}\n\nMethod: {method}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog
        )
        
        # Add standard layout arguments
        self._add_layout_arguments(parser)
        
        # Add input/output arguments
        self._add_input_output_arguments(parser)
        
        # Add scorer-specific arguments
        self._add_scorer_specific_arguments(parser)
        
        return parser
    
    def _add_layout_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add standard layout definition arguments."""
        layout_group = parser.add_argument_group('Layout Definition')
        
        # Use standard argument names directly (not from config for now)
        layout_group.add_argument(
            '--letters', '--layout-letters',
            dest='letters',
            required=True,
            help="String of characters in the layout (e.g., 'etaoinshrlcu')"
        )
        
        layout_group.add_argument(
            '--positions', '--layout-positions', '--qwerty-keys',
            dest='positions',
            required=True,
            help="String of corresponding QWERTY positions (e.g., 'FDESGJWXRTYZ')"
        )
    
    def _add_input_output_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add standard input and output arguments."""
        
        # Input arguments
        input_group = parser.add_argument_group('Input Options')
        
        input_group.add_argument(
            '--data-dir',
            dest='data_dir',
            help="Directory containing data files (overrides config)"
        )
        
        input_group.add_argument(
            '--config',
            dest='config',
            default="config.yaml",
            help="Path to configuration file (default: config.yaml)"
        )
        
        # Add text input for distance scorer
        if 'distance' in self.scorer_name:
            input_group.add_argument(
                '--text',
                dest='text',
                help="Text to analyze (alternative to --text-file)"
            )
            input_group.add_argument(
                '--text-file',
                dest='text_file',
                help="Path to text file to analyze (alternative to --text)"
            )
        
        # Output arguments
        output_group = parser.add_argument_group('Output Options')
        
        output_group.add_argument(
            '--output-format',
            dest='output_format',
            choices=['detailed', 'csv', 'score_only'],
            default='detailed',
            help="Output format (default: detailed)"
        )
        
        output_group.add_argument(
            '--csv',
            dest='csv',
            action='store_true',
            help="Output in CSV format (same as --output-format csv)"
        )
        
        output_group.add_argument(
            '--detailed',
            dest='detailed',
            action='store_true',
            help="Show detailed breakdown (same as --output-format detailed)"
        )
        
        output_group.add_argument(
            '--score-only',
            dest='score_only',
            action='store_true',
            help="Output only scores (same as --output-format score_only)"
        )
        
        output_group.add_argument(
            '--quiet',
            dest='quiet',
            action='store_true',
            help="Suppress verbose output"
        )
    
    def _add_scorer_specific_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add scorer-specific arguments based on configuration."""
        
        scorer_group = parser.add_argument_group(f'{self.scorer_name.title()} Options')
        
        # Weights argument (for dvorak9)
        if 'dvorak9' in self.scorer_name:
            scorer_group.add_argument(
                '--weights',
                dest='weights',
                help="Path to empirical weights CSV file (optional)"
            )
        
        # Cross-hand filtering (for engram)
        if 'engram' in self.scorer_name:
            scorer_group.add_argument(
                '--ignore-cross-hand',
                dest='ignore_cross_hand',
                action='store_true',
                help="Ignore bigrams that cross hands"
            )
        
        # Add data file overrides
        data_files = self.scorer_config.get('data_files', {})
        if data_files:
            for file_key, default_path in data_files.items():
                if default_path:  # Only add arguments for files that have default paths
                    arg_name = f"--{file_key.replace('_', '-')}-file"
                    scorer_group.add_argument(
                        arg_name,
                        help=f"Path to {file_key} file (default: {default_path})"
                    )
    
    def _generate_epilog(self) -> str:
        """Generate epilog text with examples and usage information."""
        
        lines = ["Examples:"]
        
        # Basic example
        basic_cmd = f"python {self.scorer_name}.py --letters 'etaoinshrlcu' --positions 'FDESGJWXRTYZ'"
        lines.append(f"  # Basic scoring")
        lines.append(f"  {basic_cmd}")
        lines.append("")
        
        # CSV output example
        csv_cmd = f"{basic_cmd} --csv"
        lines.append(f"  # CSV output")
        lines.append(f"  {csv_cmd}")
        lines.append("")
        
        # Scorer-specific examples
        if 'dvorak9' in self.scorer_name:
            weights_cmd = f"{basic_cmd} --weights input/dvorak9/speed_weights.csv"
            lines.append(f"  # With empirical weights")
            lines.append(f"  {weights_cmd}")
            lines.append("")
        
        if 'distance' in self.scorer_name:
            text_cmd = f"{basic_cmd} --text 'the quick brown fox'"
            lines.append(f"  # With text input")
            lines.append(f"  {text_cmd}")
            lines.append("")
        
        if 'engram' in self.scorer_name:
            cross_hand_cmd = f"{basic_cmd} --ignore-cross-hand"
            lines.append(f"  # Ignore cross-hand bigrams")
            lines.append(f"  {cross_hand_cmd}")
            lines.append("")
        
        # Data files info
        data_files = self.scorer_config.get('data_files', {})
        if data_files:
            lines.append("Required data files:")
            for file_key, filepath in data_files.items():
                if filepath:
                    lines.append(f"  {file_key}: {filepath}")
            lines.append("")
        
        return "\n".join(lines)
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments with validation.
        
        Args:
            args: List of arguments (uses sys.argv if None)
            
        Returns:
            Parsed arguments namespace
        """
        parsed_args = self.parser.parse_args(args)
        
        # Handle output format shortcuts
        if parsed_args.csv:
            parsed_args.output_format = 'csv'
        elif parsed_args.detailed:
            parsed_args.output_format = 'detailed'
        elif parsed_args.score_only:
            parsed_args.output_format = 'score_only'
        
        # Validate layout arguments
        try:
            layout_mapping = create_layout_mapping(parsed_args.letters, parsed_args.positions)
            validation_issues = validate_layout_mapping(layout_mapping)
            
            if validation_issues and not parsed_args.quiet:
                print("Layout validation warnings:")
                for issue in validation_issues:
                    print(f"  {issue}")
                print()
                
        except ValueError as e:
            self.parser.error(f"Layout validation error: {e}")
        
        # Validate text input for distance scorer
        if 'distance' in self.scorer_name:
            if not parsed_args.text and not parsed_args.text_file:
                self.parser.error("Must provide either --text or --text-file for distance scoring")
            
            if parsed_args.text and parsed_args.text_file:
                self.parser.error("Cannot specify both --text and --text-file")
        
        return parsed_args


def create_standard_parser(scorer_name: str, 
                         config_path: str = "config.yaml") -> StandardCLIParser:
    """
    Create a standardized CLI parser for a scorer.
    
    Args:
        scorer_name: Name of the scorer
        config_path: Path to configuration file
        
    Returns:
        Configured StandardCLIParser instance
    """
    return StandardCLIParser(scorer_name, config_path)


def handle_common_errors(func):
    """
    Decorator to handle common CLI errors gracefully.
    
    Args:
        func: Function to wrap (typically main())
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.", file=sys.stderr)
            return 130
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except PermissionError as e:
            print(f"Permission error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    return wrapper


def validate_file_access(filepath: str, mode: str = 'r') -> bool:
    """
    Validate that a file can be accessed with the specified mode.
    
    Args:
        filepath: Path to file to check
        mode: Access mode ('r', 'w', 'a')
        
    Returns:
        True if file can be accessed, False otherwise
    """
    try:
        if mode == 'r':
            return Path(filepath).exists() and Path(filepath).is_file()
        elif mode in ['w', 'a']:
            # Check if parent directory exists and is writable
            parent = Path(filepath).parent
            return parent.exists() and parent.is_dir()
        else:
            return False
    except (OSError, PermissionError):
        return False


def print_configuration_summary(config: Dict[str, Any], 
                              scorer_name: str,
                              quiet: bool = False) -> None:
    """
    Print a summary of the current configuration.
    
    Args:
        config: Configuration dictionary
        scorer_name: Name of the scorer
        quiet: If True, suppress output
    """
    if quiet:
        return
    
    print(f"{scorer_name.replace('_', ' ').title()}")
    print("=" * 50)
    
    # Basic info
    description = config.get('description', 'No description available')
    method = config.get('method', 'No method description available')
    
    print(f"Description: {description}")
    print(f"Method: {method}")
    
    # Data files
    data_files = config.get('data_files', {})
    if data_files:
        print(f"\nData files:")
        for file_key, filepath in data_files.items():
            if filepath:
                status = "✓" if Path(filepath).exists() else "✗"
                print(f"  {status} {file_key}: {filepath}")
    
    # Scoring options
    scoring_options = config.get('scoring_options', {})
    if scoring_options:
        print(f"\nScoring options:")
        for option, value in scoring_options.items():
            print(f"  {option}: {value}")
    
    print()


def get_layout_from_args(args: argparse.Namespace) -> Tuple[str, str, Dict[str, str]]:
    """
    Extract layout information from parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (letters, positions, layout_mapping)
    """
    letters = getattr(args, 'letters', None)
    positions = getattr(args, 'positions', None)
    
    if not letters or not positions:
        raise ValueError("Layout letters and positions must be specified")
    
    layout_mapping = create_layout_mapping(letters, positions)
    
    return letters, positions, layout_mapping


def determine_output_mode(args: argparse.Namespace) -> str:
    """
    Determine the output mode from parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Output mode string ('csv', 'detailed', 'score_only')
    """
    # Check for explicit format specification
    if hasattr(args, 'output_format'):
        return args.output_format
    
    # Check for boolean flags
    if getattr(args, 'csv', False):
        return 'csv'
    elif getattr(args, 'score_only', False):
        return 'score_only'
    elif getattr(args, 'detailed', False):
        return 'detailed'
    
    # Default to detailed
    return 'detailed'


def validate_input_files(args: argparse.Namespace, 
                        required_files: List[str],
                        optional_files: Optional[List[str]] = None) -> List[str]:
    """
    Validate that required input files exist and are accessible.
    
    Args:
        args: Parsed command-line arguments
        required_files: List of required file attribute names
        optional_files: List of optional file attribute names
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check required files
    for file_attr in required_files:
        filepath = getattr(args, file_attr, None)
        if filepath is None:
            issues.append(f"Required file not specified: {file_attr}")
        elif not validate_file_access(filepath, 'r'):
            issues.append(f"Cannot access required file: {filepath}")
    
    # Check optional files
    if optional_files:
        for file_attr in optional_files:
            filepath = getattr(args, file_attr, None)
            if filepath is not None and not validate_file_access(filepath, 'r'):
                issues.append(f"Cannot access optional file: {filepath}")
    
    return issues