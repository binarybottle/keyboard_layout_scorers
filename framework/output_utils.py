#!/usr/bin/env python3
"""
Output utilities for keyboard layout scoring.

Common functions for formatting and displaying scoring results in various formats.
"""

from typing import Dict, Any, List, Optional
import sys
from framework.base_scorer import ScoreResult


def format_csv_output(result: ScoreResult, 
                     config: Optional[Dict[str, Any]] = None,
                     include_metadata: bool = True) -> str:
    """
    Format scoring results as CSV output.
    
    Args:
        result: ScoreResult object to format
        config: Output format configuration
        include_metadata: Whether to include metadata fields
        
    Returns:
        CSV formatted string
    """
    if config is None:
        config = {}
    
    delimiter = config.get('delimiter', ',')
    precision = config.get('precision', 6)
    include_headers = config.get('include_headers', True)
    
    lines = []
    
    # Headers
    if include_headers:
        headers = ['primary_score']
        
        # Add component headers
        for component in sorted(result.components.keys()):
            headers.append(f'component_{component}')
        
        # Add metadata headers if requested
        if include_metadata:
            headers.extend(['scorer_name', 'execution_time'])
            
            # Add validation headers
            for key in sorted(result.validation_info.keys()):
                if isinstance(result.validation_info[key], (str, int, float, bool)):
                    headers.append(f'validation_{key}')
            
            # Add metadata headers
            for key in sorted(result.metadata.keys()):
                if isinstance(result.metadata[key], (str, int, float, bool)):
                    headers.append(f'meta_{key}')
        
        lines.append(delimiter.join(headers))
    
    # Data row
    values = [f"{result.primary_score:.{precision}f}"]
    
    # Add component values
    for component in sorted(result.components.keys()):
        values.append(f"{result.components[component]:.{precision}f}")
    
    # Add metadata values if requested
    if include_metadata:
        values.append(result.scorer_name)
        values.append(f"{result.execution_time:.3f}")
        
        # Add validation values
        for key in sorted(result.validation_info.keys()):
            value = result.validation_info[key]
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, float):
                    values.append(f"{value:.{precision}f}")
                else:
                    values.append(str(value))
        
        # Add metadata values
        for key in sorted(result.metadata.keys()):
            value = result.metadata[key]
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, float):
                    values.append(f"{value:.{precision}f}")
                else:
                    values.append(str(value))
    
    lines.append(delimiter.join(values))
    
    return '\n'.join(lines)


def format_score_only_output(result: ScoreResult,
                            config: Optional[Dict[str, Any]] = None,
                            include_components: bool = False) -> str:
    """
    Format scoring results as score-only output (compact format).
    
    Args:
        result: ScoreResult object to format
        config: Output format configuration
        include_components: Whether to include component scores
        
    Returns:
        Space-separated scores string
    """
    if config is None:
        config = {}
    
    precision = config.get('precision', 6)
    separator = config.get('separator', ' ')
    
    scores = [f"{result.primary_score:.{precision}f}"]
    
    if include_components:
        # Add component scores in consistent order
        for component in sorted(result.components.keys()):
            scores.append(f"{result.components[component]:.{precision}f}")
    
    return separator.join(scores)


def format_detailed_output(result: ScoreResult,
                         config: Optional[Dict[str, Any]] = None) -> str:
    """
    Format scoring results as detailed human-readable output.
    
    Args:
        result: ScoreResult object to format
        config: Output format configuration
        
    Returns:
        Formatted detailed output string
    """
    if config is None:
        config = {}
    
    show_breakdown = config.get('show_breakdown', True)
    show_validation = config.get('show_validation_info', True)
    show_file_sources = config.get('show_file_sources', False)
    
    lines = []
    
    # Header
    lines.append(f"\n{result.scorer_name.replace('_', ' ').title()} Results")
    lines.append("=" * 70)
    
    # Primary score
    lines.append(f"Primary score: {result.primary_score:12.6f}")
    
    # Component scores
    if result.components:
        lines.append(f"\nComponent scores:")
        for component, score in result.components.items():
            component_name = component.replace('_', ' ').title()
            lines.append(f"  {component_name:<25}: {score:10.6f}")
    
    # Execution info
    if result.execution_time > 0:
        lines.append(f"\nExecution time: {result.execution_time:.3f} seconds")
    
    # Layout info
    if result.layout_mapping:
        chars = ''.join(sorted(result.layout_mapping.keys()))
        positions = ''.join(result.layout_mapping[c] for c in sorted(result.layout_mapping.keys()))
        lines.append(f"Layout: {chars} → {positions}")
    
    # Validation information
    if show_validation and result.validation_info:
        lines.append(f"\nValidation information:")
        for key, value in sorted(result.validation_info.items()):
            # SPECIAL HANDLING for empirical_coverage
            if key == 'empirical_coverage' and isinstance(value, dict):
                lines.append(f"\nEmpirical Weight Coverage:")
                lines.append(f"  Exact matches: {value.get('exact_matches_count', 0):,} bigrams "
                           f"({value.get('exact_matches_percentage', 0):.1f}% of evaluated bigrams)")
                lines.append(f"  Frequency-weighted coverage: {value.get('exact_matches_frequency_weight', 0):.1f}%")
                lines.append(f"  No exact matches: {value.get('no_matches_count', 0):,} bigrams "
                           f"({value.get('no_matches_percentage', 0):.1f}% of evaluated bigrams)")
                continue
            
            # Regular validation info formatting
            key_name = key.replace('_', ' ').title()
            if isinstance(value, float):
                lines.append(f"  {key_name:<25}: {value:10.6f}")
            elif isinstance(value, (int, bool)):
                lines.append(f"  {key_name:<25}: {value:10}")
            else:
                lines.append(f"  {key_name:<25}: {str(value)}")
    
    # Metadata
    if result.metadata:
        lines.append(f"\nAdditional information:")
        for key, value in sorted(result.metadata.items()):
            if isinstance(value, (str, int, float, bool)):
                key_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    lines.append(f"  {key_name:<25}: {value:10.6f}")
                else:
                    lines.append(f"  {key_name:<25}: {str(value)}")
    
    # Detailed breakdown (scorer-specific)
    if show_breakdown and result.detailed_breakdown:
        lines.append(f"\nDetailed breakdown:")
        _format_detailed_breakdown(result.detailed_breakdown, lines, indent="  ")
    
    # Configuration used
    if show_file_sources and result.config_used:
        data_files = result.config_used.get('data_files', {})
        if data_files:
            lines.append(f"\nData files used:")
            for file_key, filepath in sorted(data_files.items()):
                if filepath:
                    lines.append(f"  {file_key}: {filepath}")
    
    return '\n'.join(lines)


def _format_detailed_breakdown(breakdown: Dict[str, Any], 
                             lines: List[str], 
                             indent: str = "") -> None:
    """
    Recursively format detailed breakdown information.
    
    Args:
        breakdown: Dictionary of breakdown information
        lines: List to append formatted lines to
        indent: Current indentation string
    """
    for key, value in sorted(breakdown.items()):
        key_name = key.replace('_', ' ').title()
        
        if isinstance(value, dict):
            lines.append(f"{indent}{key_name}:")
            _format_detailed_breakdown(value, lines, indent + "  ")
        elif isinstance(value, list):
            lines.append(f"{indent}{key_name}:")
            for i, item in enumerate(value[:10]):  # Limit to first 10 items
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    # Format as key-value pairs
                    lines.append(f"{indent}  {item[0]}: {item[1]}")
                else:
                    lines.append(f"{indent}  {item}")
            if len(value) > 10:
                lines.append(f"{indent}  ... and {len(value) - 10} more")
        elif isinstance(value, float):
            lines.append(f"{indent}{key_name}: {value:.6f}")
        else:
            lines.append(f"{indent}{key_name}: {value}")


def print_results(result: ScoreResult,
                 output_format: str = "detailed",
                 config: Optional[Dict[str, Any]] = None,
                 file=None) -> None:
    """
    Print scoring results in the specified format.
    
    Args:
        result: ScoreResult object to print
        output_format: Format type ('detailed', 'csv', 'score_only')
        config: Output format configuration
        file: File object to write to (defaults to stdout)
    """
    if file is None:
        file = sys.stdout
    
    if output_format == "csv":
        output = format_csv_output(result, config)
    elif output_format == "score_only":
        output = format_score_only_output(result, config)
    elif output_format == "detailed":
        output = format_detailed_output(result, config)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    print(output, file=file)


def format_comparison_output(results: List[ScoreResult],
                           layout_names: Optional[List[str]] = None,
                           output_format: str = "detailed") -> str:
    """
    Format comparison of multiple scoring results.
    
    Args:
        results: List of ScoreResult objects to compare
        layout_names: Optional names for each layout
        output_format: Format type ('detailed', 'csv', 'score_only')
        
    Returns:
        Formatted comparison string
    """
    if not results:
        return "No results to compare"
    
    if layout_names is None:
        layout_names = [f"Layout {i+1}" for i in range(len(results))]
    
    if len(layout_names) != len(results):
        raise ValueError("Number of layout names must match number of results")
    
    lines = []
    
    if output_format == "csv":
        # CSV comparison format
        if results:
            # Headers
            headers = ['layout_name', 'primary_score']
            component_names = sorted(results[0].components.keys())
            headers.extend(component_names)
            lines.append(','.join(headers))
            
            # Data rows
            for name, result in zip(layout_names, results):
                row = [name, f"{result.primary_score:.6f}"]
                for component in component_names:
                    score = result.components.get(component, 0.0)
                    row.append(f"{score:.6f}")
                lines.append(','.join(row))
    
    elif output_format == "score_only":
        # Score-only comparison
        for name, result in zip(layout_names, results):
            scores = [f"{result.primary_score:.6f}"]
            for component in sorted(result.components.keys()):
                scores.append(f"{result.components[component]:.6f}")
            lines.append(f"{name}: {' '.join(scores)}")
    
    else:  # detailed
        # Detailed comparison format
        scorer_name = results[0].scorer_name if results else "Unknown"
        lines.append(f"\n{scorer_name.replace('_', ' ').title()} Comparison")
        lines.append("=" * 70)
        
        # Primary scores comparison
        lines.append("Primary scores:")
        sorted_results = sorted(zip(layout_names, results), 
                              key=lambda x: x[1].primary_score, reverse=True)
        
        for i, (name, result) in enumerate(sorted_results):
            rank = f"#{i+1:2d}"
            lines.append(f"  {rank} {name:<20}: {result.primary_score:10.6f}")
        
        # Component comparison
        if results[0].components:
            lines.append(f"\nComponent score breakdown:")
            component_names = sorted(results[0].components.keys())
            
            # Header
            header = f"  {'Layout':<20}"
            for component in component_names:
                header += f" {component[:8]:>8}"
            lines.append(header)
            lines.append("  " + "-" * (20 + 9 * len(component_names)))
            
            # Data rows
            for name, result in zip(layout_names, results):
                row = f"  {name:<20}"
                for component in component_names:
                    score = result.components.get(component, 0.0)
                    row += f" {score:8.3f}"
                lines.append(row)
    
    return '\n'.join(lines)


def save_results_to_file(result: ScoreResult,
                        filepath: str,
                        output_format: str = "csv",
                        config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save scoring results to a file.
    
    Args:
        result: ScoreResult object to save
        filepath: Path to output file
        output_format: Format type ('detailed', 'csv', 'score_only')
        config: Output format configuration
    """
    if output_format == "csv":
        content = format_csv_output(result, config)
    elif output_format == "score_only":
        content = format_score_only_output(result, config)
    elif output_format == "detailed":
        content = format_detailed_output(result, config)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write('\n')  # Ensure file ends with newline


def create_summary_table(results: Dict[str, ScoreResult],
                        title: str = "Layout Scoring Summary") -> str:
    """
    Create a summary table of multiple layout results.
    
    Args:
        results: Dictionary mapping layout names to ScoreResult objects
        title: Title for the summary table
        
    Returns:
        Formatted summary table string
    """
    if not results:
        return "No results to summarize"
    
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * len(title))
    
    # Sort by primary score (descending)
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1].primary_score, 
                          reverse=True)
    
    # Create table
    lines.append(f"{'Rank':<6} {'Layout':<20} {'Primary score':<15} {'Scorer':<15}")
    lines.append("-" * 60)
    
    for i, (name, result) in enumerate(sorted_results):
        rank = f"#{i+1}"
        scorer = result.scorer_name.replace('_scorer', '').title()
        lines.append(f"{rank:<6} {name:<20} {result.primary_score:<15.6f} {scorer:<15}")
    
    return '\n'.join(lines)


def format_empirical_coverage(empirical_coverage: Dict[str, float]) -> str:
    """Format empirical weight coverage information for display."""
    if not empirical_coverage:
        return ""
    
    lines = [
        "\n=== Empirical Weight Coverage ===",
        f"Exact matches: {empirical_coverage['exact_matches_count']:,} bigrams "
        f"({empirical_coverage['exact_matches_percentage']:.1f}% of evaluated bigrams)",
        f"Frequency-weighted coverage: {empirical_coverage['exact_matches_frequency_weight']:.1f}%",
        f"No exact matches: {empirical_coverage['no_matches_count']:,} bigrams "
        f"({empirical_coverage['no_matches_percentage']:.1f}% of evaluated bigrams)",
    ]
    return "\n".join(lines)