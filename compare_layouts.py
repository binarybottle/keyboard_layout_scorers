#!/usr/bin/env python3
"""
compare_layouts.py - Parallel Coordinates Layout Comparison

Creates parallel coordinates plots comparing keyboard layouts across performance metrics.
Each layout is represented as a line connecting its normalized scores across 18 metrics.

ARGUMENTS:
    --tables <file1.csv> [file2.csv ...]
        One or more CSV files containing keyboard layout performance data.
        
        Single table behavior:
            - Each layout rendered as a grayscale line
            - Line darkness varies by CSV row number (lighter = earlier rows)
            - Useful for exploring variation within a single dataset
            
        Multiple tables behavior:
            - Each table gets a distinct color (blue, red, green, orange, etc.)
            - All layouts from the same table share the same color
            - Useful for comparing different layout groups (e.g., standard vs experimental)
            - Legend shows table names and layout counts
    
    --variant {common_filtered,common,full,filtered}
        Optional filter to select specific layout variants from the data.
        
        full:          Include only layouts with '_full' in their name  
                      (all letter patterns, all letter-pair analyses included)
        filtered:       Include only layouts with '_filtered' in their name
                       (all letter patterns, cross-hand bigrams removed)
        common:         Include only layouts with '_common_full' in their name
                       (common letter patterns, all analyses included)  
        common_filtered: Include only layouts with '_common_filtered' in their name
                        (common letter patterns, cross-hand bigrams removed)
        
        If not specified, all layouts in the CSV files are included.
        
    --output <filepath>
        Save the plot to a file instead of displaying it interactively.
        
        Supported formats: PNG, PDF, SVG, JPG (determined by file extension)
        Recommended: Use .png for high-quality raster images
        Example: --output keyboard_comparison.png
        
        If not specified, plot is displayed in a window using matplotlib's default backend.
        
    --verbose
        Print detailed information during processing including:
        - Number of layouts loaded from each file
        - Filtering results when using --variant
        - Missing metrics warnings
        - Data normalization details
        
        Useful for debugging data issues or understanding what's being plotted.

DATA REQUIREMENTS:
    CSV files must contain:
    - A 'layout' column with layout identifiers
    - 18 performance metric columns (distance_scorer_primary, engram_scorer_*, dvorak9_scorer_*)
    - Numeric values for all metrics (missing values are handled gracefully)
    
METRICS PLOTTED (in order):
    1. Distance Scorer:    distance_scorer_primary
    2. Engram Scorer:      engram_scorer_item_component_24key/32key
                          engram_scorer_item_pair_component_24key/32key  
    3. Dvorak9 Main:       dvorak9_scorer_pure_dvorak_score
                          dvorak9_scorer_frequency/speed/comfort_weighted_score
    4. Dvorak9 Subscores:  dvorak9_scorer_columns/dont_cross_home/fingers/hands
                          dvorak9_scorer_home_row/same_row/skip_fingers
                          dvorak9_scorer_strong_fingers/strum

NORMALIZATION:
    All metrics are normalized to 0-1 scale using global min/max across ALL input tables.
    This ensures fair comparison when plotting multiple datasets together.
    0 = worst performance across all layouts, 1 = best performance across all layouts.

Usage Examples:
    # Explore single dataset with grayscale visualization
    python compare_layouts.py --tables my_layouts.csv
    
    # Compare standard vs experimental layouts (blue vs red)
    python compare_layouts.py --tables standard_layouts.csv experimental_layouts.csv
    
    # Filter for only common filtered data and save result
    python compare_layouts.py --tables *.csv --variant common_filtered --output comparison.png
    
    # Compare multiple layout groups with detailed output
    python compare_layouts.py --tables group1.csv group2.csv group3.csv --verbose
    
    # Analyze only filtered (cross-hand bigrams removed) data
    python compare_layouts.py --tables layouts.csv --variant filtered --output filtered_analysis.pdf
    
    # Compare common letter patterns only
    python compare_layouts.py --tables standard.csv experimental.csv --variant common
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional

# Define the metrics in the specified order
METRICS = [
    'distance_scorer_primary',
    'engram_scorer_item_component_24key',
    'engram_scorer_item_component_32key', 
    'engram_scorer_item_pair_component_24key',
    'engram_scorer_item_pair_component_32key',
    'dvorak9_scorer_pure_dvorak_score',
    'dvorak9_scorer_frequency_weighted_score',
    'dvorak9_scorer_speed_weighted_score',
    'dvorak9_scorer_comfort_weighted_score',
    'dvorak9_scorer_columns',
    'dvorak9_scorer_dont_cross_home',
    'dvorak9_scorer_fingers',
    'dvorak9_scorer_hands',
    'dvorak9_scorer_home_row',
    'dvorak9_scorer_same_row',
    'dvorak9_scorer_skip_fingers',
    'dvorak9_scorer_strong_fingers',
    'dvorak9_scorer_strum'
]

# Short names for display
METRIC_LABELS = {
    'distance_scorer_primary': 'Distance\nPrimary',
    'engram_scorer_item_component_24key': 'Engram\nItem 24k',
    'engram_scorer_item_component_32key': 'Engram\nItem 32k',
    'engram_scorer_item_pair_component_24key': 'Engram\nPair 24k',
    'engram_scorer_item_pair_component_32key': 'Engram\nPair 32k',
    'dvorak9_scorer_pure_dvorak_score': 'Pure\nDvorak',
    'dvorak9_scorer_frequency_weighted_score': 'Frequency\nScore',
    'dvorak9_scorer_speed_weighted_score': 'Speed\nScore',
    'dvorak9_scorer_comfort_weighted_score': 'Comfort\nScore',
    'dvorak9_scorer_columns': 'Columns',
    'dvorak9_scorer_dont_cross_home': 'Dont Cross\nHome',
    'dvorak9_scorer_fingers': 'Fingers',
    'dvorak9_scorer_hands': 'Hands',
    'dvorak9_scorer_home_row': 'Home\nRow',
    'dvorak9_scorer_same_row': 'Same\nRow',
    'dvorak9_scorer_skip_fingers': 'Skip\nFingers',
    'dvorak9_scorer_strong_fingers': 'Strong\nFingers',
    'dvorak9_scorer_strum': 'Strum'
}

def load_and_filter_data(file_path: str, variant: Optional[str] = None) -> pd.DataFrame:
    """Load CSV data and optionally filter by variant."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        
        if variant:
            # Filter by variant 
            if variant == 'common_filtered':
                filtered_df = df[df['layout'].str.contains('_common_filtered', na=False)]
            elif variant == 'common':
                filtered_df = df[df['layout'].str.contains('_common_full', na=False)]
            elif variant == 'filtered':
                filtered_df = df[df['layout'].str.contains('_filtered', na=False)]
            elif variant == 'full':
                filtered_df = df[df['layout'].str.contains('_full', na=False)]
            else:
                print(f"Warning: Unknown variant '{variant}', using all data")
                filtered_df = df
                
            print(f"Filtered to {len(filtered_df)} rows for variant '{variant}'")
            return filtered_df
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def normalize_data(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison."""
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
    normalized_dfs = []
    
    for df in dfs:
        normalized_df = df.copy()
        
        for metric in METRICS:
            if metric in df.columns:
                global_min = all_data[metric].min()
                global_max = all_data[metric].max()
                
                if global_max != global_min:
                    normalized_df[metric] = (df[metric] - global_min) / (global_max - global_min)
                else:
                    normalized_df[metric] = 0.5  # If no variation, put in middle
            else:
                print(f"Warning: Metric '{metric}' not found in data")
                normalized_df[metric] = 0.5
        
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
                        variant: str, output_path: Optional[str] = None) -> None:
    """Create parallel coordinates plot."""
    
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Get colors
    colors = get_colors(len(dfs))
    
    # Plot parameters
    x_positions = range(len(METRICS))
    
    # Plot each table's data
    for i, (df, table_name, color) in enumerate(zip(normalized_dfs, table_names, colors)):
        
        if len(dfs) == 1:
            # Single table: use grayscale with line numbers
            num_layouts = len(df)
            gray_values = np.linspace(0.2, 0.8, num_layouts)
            
            for j, (_, row) in enumerate(df.iterrows()):
                y_values = [row[metric] for metric in METRICS]
                
                # Skip rows with missing data
                if any(pd.isna(val) for val in y_values):
                    continue
                
                gray_color = str(gray_values[j])
                ax.plot(x_positions, y_values, color=gray_color, alpha=0.7, linewidth=1.5,
                       label=f"Layout {j+1}" if j < 10 else "")  # Only label first 10
        else:
            # Multiple tables: use color groups
            for _, row in df.iterrows():
                y_values = [row[metric] for metric in METRICS]
                
                # Skip rows with missing data
                if any(pd.isna(val) for val in y_values):
                    continue
                
                ax.plot(x_positions, y_values, color=color, alpha=0.6, linewidth=1.5)
            
            # Add a single legend entry for this table
            ax.plot([], [], color=color, linewidth=3, label=f"{table_name} ({len(df)} layouts)")
    
    # Customize the plot
    ax.set_xlim(-0.5, len(METRICS) - 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in METRICS], 
                       rotation=45, ha='right', fontsize=10)
    
    # Add vertical grid lines for each metric
    for x in x_positions:
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Set y-axis
    ax.set_ylabel('Normalized Score (0 = worst, 1 = best)', fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    variant_title = variant.upper() if variant else "ALL VARIANTS"
    ax.set_title(f'Keyboard Layout Comparison - {variant_title}\n'
                f'Parallel Coordinates Across {len(METRICS)} Performance Metrics', 
                fontsize=16, fontweight='bold', pad=20)
    
    if len(dfs) > 1 or len(dfs[0]) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add metric category annotations
    category_positions = {
        'Distance': 0,
        'Engram': 2.5,
        'Dvorak9 Main': 7.5,
        'Dvorak9 Subscores': 12.5
    }
    
    for category, pos in category_positions.items():
        ax.text(pos, 1.08, category, transform=ax.get_xaxis_transform(), 
               ha='center', va='bottom', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def print_summary_stats(dfs: List[pd.DataFrame], table_names: List[str]) -> None:
    """Print summary statistics for the loaded data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for df, name in zip(dfs, table_names):
        print(f"\n{name}:")
        print(f"  Layouts: {len(df)}")
        
        # Count missing metrics
        missing_counts = df[METRICS].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  Missing data points: {missing_counts.sum()}")
        
        # Sample layout names
        if 'layout' in df.columns:
            sample_layouts = df['layout'].head(3).tolist()
            print(f"  Sample layouts: {', '.join(sample_layouts)}")

def main():
    parser = argparse.ArgumentParser(
        description='Create parallel coordinates plots comparing keyboard layouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_layouts.py --tables layouts.csv
  python compare_layouts.py --tables standard.csv experimental.csv
  python compare_layouts.py --tables *.csv --variant common_filtered
  python compare_layouts.py --tables data1.csv data2.csv --output comparison.png
  python compare_layouts.py --tables layouts.csv --variant filtered --verbose
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files containing layout data')
    parser.add_argument('--variant', choices=['common_filtered', 'common', 'filtered', 'full'], 
                       help='Filter layouts by variant (common_filtered/common/filtered/full)')
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
        df = load_and_filter_data(table_path, args.variant)
        
        if len(df) == 0:
            print(f"Warning: No data found in {table_path} for variant '{args.variant}'")
            continue
            
        dfs.append(df)
        table_names.append(Path(table_path).stem)
    
    if not dfs:
        print("Error: No valid data found in any table")
        sys.exit(1)
    
    # Print summary
    print_summary_stats(dfs, table_names)
    
    # Verify metrics exist
    all_metrics_present = True
    for df in dfs:
        missing_metrics = [metric for metric in METRICS if metric not in df.columns]
        if missing_metrics:
            print(f"Warning: Missing metrics in {table_names[dfs.index(df)]}: {missing_metrics}")
            all_metrics_present = False
    
    if not all_metrics_present:
        print("\nContinuing with available metrics...")
    
    # Create plot
    print(f"\nCreating parallel coordinates plot...")
    print(f"Tables: {len(dfs)}")
    print(f"Total layouts: {sum(len(df) for df in dfs)}")
    
    create_parallel_plot(dfs, table_names, args.variant or "all", args.output)

if __name__ == "__main__":
    main()