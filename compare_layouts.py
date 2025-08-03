#!/usr/bin/env python3
"""
compare_layouts.py - Parallel Coordinates Layout Comparison

Creates parallel coordinates plots comparing keyboard layouts across performance metrics.
Each layout is represented as a line connecting its normalized scores across available metrics.

This version is more robust and handles missing columns gracefully.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional

# Define the metrics in the specified order (matching your actual CSV columns)
IDEAL_METRICS = [
    'engram_item_component_32key', 
    'engram_item_pair_component_32key',
    'distance_primary',
    'dvorak9_pure_dvorak_score',
    'dvorak9_frequency_weighted_score',
    'dvorak9_speed_weighted_score',
    'dvorak9_comfort_weighted_score',
    'dvorak9_hands',
    'dvorak9_fingers',
    'dvorak9_skip_fingers',
    'dvorak9_dont_cross_home',
    'dvorak9_same_row',
    'dvorak9_home_row',
    'dvorak9_columns',
    'dvorak9_strum',
    'dvorak9_strong_fingers'
]

# Short names for display
METRIC_LABELS = {
    'engram_item_component_32key': 'Engram\nitem',
    'engram_item_pair_component_32key': 'Engram\npair',
    'distance_primary': 'Distance',
    'dvorak9_pure_dvorak_score': 'Dvorak-9',
    'dvorak9_frequency_weighted_score': 'Dvorak-9\nfrequency',
    'dvorak9_speed_weighted_score': 'Dvorak-9\nspeed',
    'dvorak9_comfort_weighted_score': 'Dvorak-9\ncomfort',
    'dvorak9_hands': '1. different\nhands',
    'dvorak9_fingers': '2. different\nfingers',
    'dvorak9_skip_fingers': '3. skip\nfingers',
    'dvorak9_dont_cross_home': '4. don\'t cross\nhome row',
    'dvorak9_same_row': '5. same\nrow',
    'dvorak9_home_row': '6. home\nrow',
    'dvorak9_columns': '7. within\ncolumns',
    'dvorak9_strum': '8. inward\nroll',
    'dvorak9_strong_fingers': '9. strong\nfingers'
}

def find_available_metrics(dfs: List[pd.DataFrame], verbose: bool = False) -> List[str]:
    """Find which metrics from IDEAL_METRICS are actually available in the data."""
    # Get all columns that appear to be numeric metrics
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    # Find numeric columns that might be metrics
    numeric_metrics = []
    for df in dfs:
        for col in df.columns:
            if col != 'layout' and pd.api.types.is_numeric_dtype(df[col]):
                if col not in numeric_metrics:
                    numeric_metrics.append(col)
    
    # Only use metrics from IDEAL_METRICS that are available in the data
    available_metrics = []
    for metric in IDEAL_METRICS:
        if metric in numeric_metrics:
            available_metrics.append(metric)
    
    if verbose:
        print(f"\nFound {len(available_metrics)} metrics from IDEAL_METRICS to plot:")
        for i, metric in enumerate(available_metrics):
            print(f"  {i+1:2d}. {metric}")
        
        missing_ideal = [m for m in IDEAL_METRICS if m not in available_metrics]
        if missing_ideal:
            print(f"\nMissing ideal metrics ({len(missing_ideal)}):")
            for metric in missing_ideal:
                print(f"     {metric}")
    
    return available_metrics

def load_and_filter_data(file_path: str, variant: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
    """Load CSV data and optionally filter by variant."""
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"\nLoaded {len(df)} rows from {file_path}")
            print(f"Columns: {list(df.columns)}")
        
        if variant:
            # Filter by variant 
            if variant == 'no_crosshand':
                filtered_df = df[df['layout'].str.contains('no_crosshand_', na=False)]
            elif variant == 'full':
                filtered_df = df[df['layout'].str.contains('full_', na=False)]
            else:
                print(f"Warning: Unknown variant '{variant}', using all data")
                filtered_df = df
                
            if verbose:
                print(f"Filtered to {len(filtered_df)} rows for variant '{variant}'")
            return filtered_df
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def normalize_data(dfs: List[pd.DataFrame], metrics: List[str]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison."""
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
    normalized_dfs = []
    
    for df in dfs:
        normalized_df = df.copy()
        
        for metric in metrics:
            if metric in df.columns:
                global_min = all_data[metric].min()
                global_max = all_data[metric].max()
                
                if pd.notna(global_min) and pd.notna(global_max) and global_max != global_min:
                    normalized_df[metric] = (df[metric] - global_min) / (global_max - global_min)
                else:
                    normalized_df[metric] = 0.5  # Default to middle if no variation
            else:
                normalized_df[metric] = 0.0  # Default to 0 if metric is missing
        
        normalized_dfs.append(normalized_df)
    
    return normalized_dfs

def get_colors(num_tables: int) -> List[str]:
    """Get color scheme based on number of tables."""
    if num_tables == 1:
        return ['gray']
    elif num_tables == 2:
        return ['#2196F3', '#F44336']  # Blue, Red
    elif num_tables == 3:
        return ['#2196F3', '#F44336', '#4CAF50']  # Blue, Red, Green
    elif num_tables == 4:
        return ['#2196F3', '#F44336', '#4CAF50', '#FF9800']  # Blue, Red, Green, Orange
    else:
        # Use a colormap for many tables
        cmap = plt.cm.Set3
        return [cmap(i / num_tables) for i in range(num_tables)]

def create_parallel_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                        metrics: List[str], variant: str, output_path: Optional[str] = None) -> None:
    """Create parallel coordinates plot."""
    
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs, metrics)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.2), 10))
    
    # Get colors
    colors = get_colors(len(dfs))
    
    # Plot parameters
    x_positions = range(len(metrics))
    
    # Plot each table's data
    for i, (df, table_name, color) in enumerate(zip(normalized_dfs, table_names, colors)):
        
        if len(dfs) == 1:
            # Single table: use grayscale with line numbers
            num_layouts = len(df)
            gray_values = np.linspace(0.2, 0.8, num_layouts)
            
            for j, (_, row) in enumerate(df.iterrows()):
                y_values = [row.get(metric, 0) for metric in metrics]
                
                # Skip rows with too much missing data
                valid_values = [val for val in y_values if pd.notna(val)]
                if len(valid_values) < len(metrics) * 0.5:  # Need at least 50% valid data
                    continue
                
                # Replace NaN values with 0
                y_values = [val if pd.notna(val) else 0 for val in y_values]
                
                gray_color = str(gray_values[j])
                ax.plot(x_positions, y_values, color=gray_color, alpha=0.7, linewidth=1.5,
                       label=f"Layout {j+1}" if j < 10 else "")  # Only label first 10
        else:
            # Multiple tables: use color groups
            valid_layout_count = 0
            for _, row in df.iterrows():
                y_values = [row.get(metric, 0) for metric in metrics]
                
                # Skip rows with too much missing data
                valid_values = [val for val in y_values if pd.notna(val)]
                if len(valid_values) < len(metrics) * 0.5:  # Need at least 50% valid data
                    continue
                
                # Replace NaN values with 0
                y_values = [val if pd.notna(val) else 0 for val in y_values]
                
                ax.plot(x_positions, y_values, color=color, alpha=0.6, linewidth=1.5)
                valid_layout_count += 1
            
            # Add a single legend entry for this table
            ax.plot([], [], color=color, linewidth=3, label=f"{table_name} ({valid_layout_count} layouts)")
    
    # Customize the plot
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis labels
    metric_display_names = []
    for metric in metrics:
        if metric in METRIC_LABELS:
            metric_display_names.append(METRIC_LABELS[metric])
        else:
            # Create a reasonable display name from the column name
            display_name = metric.replace('_', ' ').title()
            if len(display_name) > 15:
                # Split long names
                words = display_name.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    display_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            metric_display_names.append(display_name)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=10)
    
    # Add vertical grid lines for each metric
    for x in x_positions:
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Set y-axis
    ax.set_ylabel('Normalized Score (0 = worst, 1 = best)', fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    if variant and variant != "all":
        ax.set_title(f'Keyboard Layout Comparison ({variant})\n'
                    f'Parallel coordinates across {len(metrics)} metrics', 
                    fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'Keyboard Layout Comparison\n'
                    f'Parallel coordinates across {len(metrics)} metrics', 
                    fontsize=16, fontweight='bold', pad=20)

    if len(dfs) > 1 or len(dfs[0]) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def print_summary_stats(dfs: List[pd.DataFrame], table_names: List[str], metrics: List[str]) -> None:
    """Print summary statistics for the loaded data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for df, name in zip(dfs, table_names):
        print(f"\n{name}:")
        print(f"  Layouts: {len(df)}")
        
        # Count missing metrics for available metrics only
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            missing_counts = df[available_metrics].isnull().sum()
            if missing_counts.sum() > 0:
                print(f"  Missing data points: {missing_counts.sum()}")
        
        # Sample layout names
        if 'layout' in df.columns:
            sample_layouts = df['layout'].head(3).tolist()
            print(f"  Sample layouts: {', '.join(sample_layouts)}")
        else:
            print("  Warning: No 'layout' column found")
            
def main():
    parser = argparse.ArgumentParser(
        description='Create parallel coordinates plots comparing keyboard layouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_layouts.py --tables layouts.csv
  python compare_layouts.py --tables standard.csv experimental.csv
  python compare_layouts.py --tables *.csv --variant full
  python compare_layouts.py --tables data1.csv data2.csv --output comparison.png
  python compare_layouts.py --tables layouts.csv --variant filtered --verbose
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files containing layout data')
    parser.add_argument('--variant', choices=['full', 'no_crosshand'], 
                       help='Filter layouts by variant (full/no_crosshand)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (if not specified, plot is shown)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Validate files exist
    for table_path in args.tables:
        if not Path(table_path).exists():
            print(f"Error: File '{table_path}' not found")
            sys.exit(1)
    
    # Load data
    dfs = []
    table_names = []
    
    for table_path in args.tables:
        df = load_and_filter_data(table_path, args.variant, args.verbose)
        
        if len(df) == 0:
            if args.verbose:
                print(f"Warning: No data found in {table_path} for variant '{args.variant}'")
            continue
            
        dfs.append(df)
        table_names.append(Path(table_path).stem)
    
    if not dfs:
        print("Error: No valid data found in any table")
        sys.exit(1)
    
    # Find available metrics
    metrics = find_available_metrics(dfs, args.verbose)
    
    if not metrics:
        print("Error: No numeric metrics found in the data")
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print_summary_stats(dfs, table_names, metrics)
    
    # Create plot
    if args.verbose:
        print(f"\nCreating parallel coordinates plot...")
        print(f"Tables: {len(dfs)}")
        print(f"Total layouts: {sum(len(df) for df in dfs)}")
        print(f"Metrics to plot: {len(metrics)}")
    
    create_parallel_plot(dfs, table_names, metrics, args.variant or "all", args.output)

if __name__ == "__main__":
    main()