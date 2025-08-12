#!/usr/bin/env python3
"""
Keyboard layout comparison

Creates parallel coordinates and heatmap plots comparing keyboard layouts across performance metrics.

This version expects CSV files in the format:
layout_name,scorer,weighted_score,raw_score

Usage:
# Single table - sorts all layouts by performance
python compare_layouts.py --tables layout_scores.csv

# Multiple tables - sorts within each table, maintains grouping  
python compare_layouts.py --tables scores1.csv scores2.csv

# With output file
python compare_layouts.py --tables scores.csv# This creates a CSV file with layout_name,scorer,weighted_score,raw_score
poetry run python3 score_layouts.py --compare moo_layout_1:"xpou'\"dsmwgiae,.tnrckjyh-?lfbvqz" moo_layout_2:"xpou'\"dsmbgiae,.tnrckjyh-?lfvwqz" moo_layout_3:"kpou'\"dsmwgiae,.tnrcxjyh-?lfvbqz" moo_layout_4:"kpou'\"lscwgiae,.tnrfxjyh-?dmvbqz" moo_layout_5:"xpou'\"lscbgiae,.tnrmkjyh-?dfvwqz" moo_layout_6:"kpou'\"lscbgiae,.tnrmxjyh-?dfvwqz" moo_layout_7:"kpou'\"dscwgiae,.tnrmxjyh-?lfvbqz" moo_layout_8:"kuod'\"csmwpiae,.tnrgxjyh-?lfvbqz" moo_layout_9:"juod'\"lsmbpiae,.tnrfkxyh-?cgvwqz" moo_layout_10:"juod'\"csmbpiae,.tnrfkxyh-?lgvwqz" moo_layout_11:"kuod'\"csmbpiae,.tnrfxjyh-?lgvwqz" moo_layout_12:"xpou'\"lnfghiae,.tsrckjyd-?mbvwqz" moo_layout_13:"yuod'\"csmbpiae,.tnrfkjgh-?lwvxqz" moo_layout_14:"kpou'\"mrgbhiae,.tnscxjyd-?lfvwqz" moo_layout_15:"kpou'\"mrcbhiae,.tnsgxjyd-?lfvwqz" moo_layout_16:"ypou'\"mrgbdiae,.tnscxjkh-?lfvwqz" moo_layout_17:"xpou'\"mrgbdiae,.tnsckjyh-?lfvwqz" moo_layout_18:"ypou'\"lsmbgiae,.tnrcxjkh-?dfvwqz" moo_layout_19:"xpou'\"dscwgiae,.tnrmkjyh-?lfbvqz" moo_layout_20:"kmil'\"cnfwuroe,.taspxjyh-?dgvbqz" moo_layout_21:"kuil'\"cnmbproe,.tasfxjyh-?dgvwqz" moo_layout_22:"kuil'\"cnmwproe,.tasfxjyh-?dgvbqz" moo_layout_23:"kuil'\"dnfwmroe,.taspxjyh-?cgvbqz" moo_layout_24:"kmil'\"dnfguroe,.taspxjyh-?cwvbqz" moo_layout_25:"yuil'\"cnfbmroe,.taspkjwh-?dgxvqz" moo_layout_26:"kuil'\"cnfwmroe,.taspxjyh-?dgvbqz" moo_layout_27:"kuil'\"dngwmroe,.taspxjyh-?cfvbqz" moo_layout_28:"kmil'\"dngwuroe,.taspxjyh-?cfvbqz" moo_layout_29:"kuid'\"cnmbfsae,.torpwjyh-?lgxvqz" moo_layout_30:"kgil'\"dnfwuroe,.tasmxjyh-?cpbvqz" moo_layout_31:"wuid'\"cnmbfsae,.torpkjyh-?lgxvqz" moo_layout_32:"ymil'\"dngwuroe,.taspxjkh-?cfvbqz" moo_layout_33:"kuid'\"cnmbfsae,.torpyjwh-?lgxvqz" moo_layout_34:"kpio'\"dnlwghae,.tsrcxjyu-?mfvbqz" moo_layout_35:"xpou'\"mrcbhiae,.tnsgkjyd-?lfvwqz" moo_layout_36:"ymil'\"cnfburoe,.taspkjwh-?dgxvqz" moo_layout_37:"yuid'\"cnmbfsae,.torpkjwh-?lgxvqz" moo_layout_38:"kgio'\"dnlwphae,.tsrcxjyu-?mfvbqz" moo_layout_39:"kpou'\"lscbhiae,.tnrgxjyd-?mfvwqz" moo_layout_40:"ypou'\"lsmbgiae,.tnrcjxkh-?dfvwqz" moo_layout_41:"kuil'\"cnfbmroe,.taspwjyh-?dgxvqz" moo_layout_42:"kpou'\"lngfhiae,.tsrcxjyd-?mwvbqz" moo_layout_43:"klio'\"dncwghae,.tsrpxjyu-?mfvbqz" moo_layout_44:"xpou'\"mrgbhiae,.tnsckjyd-?lfvwqz" moo_layout_45:"xpou'\"lnfghiae,.tsrckjyd-?mwvbqz" moo_layout_46:"xpou'\"lngfhiae,.tsrckjyd-?mbvwqz" moo_layout_47:"yuil'\"cnmbproe,.tasfkjwh-?dgxvqz" moo_layout_48:"xpou'\"lngfhiae,.tsrckjyd-?mwvbqz" moo_layout_49:"kpou'\"lngwhiae,.tsrcxjyd-?mfvbqz" moo_layout_50:"xpou'\"lngwhiae,.tsrckjyd-?mfvbqz" moo_layout_51:"xpou'\"lscbhiae,.tnrgkjyd-?mfvwqz" moo_layout_52:"yuod'\"csmbpiae,.tnrfkjgh-?lwxvqz" moo_layout_53:"xpou'\"csmbdiae,.tnrgkjyh-?lfvwqz" moo_layout_54:"xpou'\"dscwgiae,.tnrmkjyh-?lfvbqz" moo_layout_55:"juod'\"csmbpiae,.tnrgkxyh-?lfvwqz" moo_layout_56:"kuod'\"csmbpiae,.tnrgxjyh-?lfvwqz" moo_layout_57:"ywil'\"dngfuroe,.tasmxjkh-?cpbvqz" moo_layout_58:"kuod'\"csmwpiae,.tnrfxjyh-?lgvbqz" moo_layout_59:"whif'\"lsmbpnae,.torckjyu-?dgxvqz" moo_layout_60:"kpou'\"csmbdiae,.tnrgxjyh-?lfvwqz" moo_layout_61:"whif'\"lsmbunae,.torckjyp-?dgxvqz" moo_layout_62:"xpou'\"lsmbgiae,.tnrckjyh-?dfvwqz" moo_layout_63:"kpou'\"lsmbgiae,.tnrcxjyh-?dfvwqz" moo_layout_64:"xpou'\"dsmwgiae,.tnrckjyh-?lfvbqz" moo_layout_65:"kmil'\"dnfwuroe,.taspxjyh-?cgvbqz" moo_layout_66:"xpou'\"lscbgiae,.tnrfkjyh-?dmvwqz" moo_layout_67:"kuil'\"dnfgmroe,.taspxjyh-?cwvbqz" moo_layout_68:"kpou'\"lscbgiae,.tnrfxjyh-?dmvwqz" moo_layout_69:"kgio'\"dncwphae,.tsrlxjyu-?mfvbqz" --csv-output > moo_layout_scores.csv


# Use raw scores instead of weighted scores
python compare_layouts.py --tables scores.csv --use-raw

# Verbose mode
python compare_layouts.py --tables scores.csv --verbose

Example input format (from score_layouts.py --csv-output):
layout_name,scorer,weighted_score,raw_score
qwerty,distance,0.756234,0.742156
qwerty,comfort,0.623451,0.618923
dvorak,distance,0.834567,0.821234
dvorak,comfort,0.712345,0.708912
...

For reference:
- Halmak 2.2	     wlrbz;qudjshnt,.aeoifmvc/gpxky['
- Hieamtsrn	       byou'kdclphiea,mtsrnx-".?wgfjzqv
- Colemak-DH	     qwfpbjluy;arstgmneiozxcdvkh,./['
- Norman	         qwdfkjurl;asetgyniohzxcvbpm,./['
- Workman	         qdrwbjfup;ashtgyneoizxmcvkl,./['
- MTGAP 2.0	       ,fhdkjcul.oantgmseriqxbpzyw'v;['
- QGMLWB	         qgmlwbyuv;dstnriaeohzxcfjkp,./['
- Colemak	         qwfpgjluy;arstdhneiozxcvbkm,./['
- Asset	           qwfgjypul;asetdhniorzxcvbkm,./['
- Capewell-Dvorak	 ',.pyqfgrkoaeiudhtnszxcvjlmwb;['
- Klausler	       k,uypwlmfcoaeidrnthsq.';zxvgbj['
- Dvorak	         ',.pyfgcrlaoeuidhtns;qjkxbmwvz['
- QWERTY	         qwertyuiopasdfghjkl;zxcvbnm,./['

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional

def load_and_pivot_data(file_path: str, use_raw: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Load CSV data from score_layouts.py output and pivot to layout x scorer format."""
    try:
        df = pd.read_csv(file_path)
        if verbose:
            print(f"\nLoaded {len(df)} rows from {file_path}")
            print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['layout_name', 'scorer']
        
        # Handle different score column formats from score_layouts.py
        if use_raw:
            # For raw scoring, accept: 'raw_score', 'score', or any other numeric column
            possible_cols = ['raw_score', 'score']
            score_col = None
            for col in possible_cols:
                if col in df.columns:
                    score_col = col
                    break
            
            if score_col is None:
                # Fallback: find any numeric column that's not layout_name/scorer
                numeric_cols = [col for col in df.columns 
                              if col not in ['layout_name', 'scorer'] and pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    score_col = numeric_cols[0]
                    if verbose:
                        print(f"Warning: Using '{score_col}' column for raw scores")
                else:
                    raise ValueError("No suitable score column found for raw scoring")
        else:
            # For weighted scoring, try in order of preference
            possible_cols = ['weighted_score', 'average_score', 'raw_score', 'score']
            score_col = None
            for col in possible_cols:
                if col in df.columns:
                    score_col = col
                    break
            
            if score_col is None:
                # Fallback: find any numeric column
                numeric_cols = [col for col in df.columns 
                              if col not in ['layout_name', 'scorer'] and pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    score_col = numeric_cols[0]
                    if verbose:
                        print(f"Warning: Using '{score_col}' column for scoring")
                else:
                    raise ValueError("No suitable score column found")
            
            if score_col in ['raw_score', 'score'] and verbose:
                print("Warning: No weighted_score column found, using raw scores")        

        required_cols.append(score_col)
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if verbose:
            print(f"Using score column: '{score_col}'")
            print(f"Found {len(df['layout_name'].unique())} unique layouts")
            print(f"Found {len(df['scorer'].unique())} unique scorers")
            print(f"Scorers: {', '.join(sorted(df['scorer'].unique()))}")
        
        # Pivot the data: layouts as rows, scorers as columns
        try:
            pivoted = df.pivot(index='layout_name', columns='scorer', values=score_col)
            pivoted = pivoted.reset_index()
            pivoted = pivoted.rename(columns={'layout_name': 'layout'})
            
            if verbose:
                print(f"Pivoted data shape: {pivoted.shape}")
                print(f"Layouts: {len(pivoted)}")
                print(f"Metrics (scorers): {len(pivoted.columns) - 1}")
            
            return pivoted
            
        except Exception as e:
            raise ValueError(f"Error pivoting data - possible duplicate layout+scorer combinations: {e}")
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def find_available_metrics(dfs: List[pd.DataFrame], verbose: bool = False) -> List[str]:
    """Find which scorer metrics are available in the data."""
    # Get all columns that appear to be numeric metrics (excluding 'layout')
    all_metrics = set()
    for df in dfs:
        for col in df.columns:
            if col != 'layout' and pd.api.types.is_numeric_dtype(df[col]):
                all_metrics.add(col)
    
    # Convert to sorted list
    available_metrics = sorted(list(all_metrics))
    
    if verbose:
        print(f"\nFound {len(available_metrics)} scorer metrics to plot:")
        for i, metric in enumerate(available_metrics):
            print(f"  {i+1:2d}. {metric}")
    
    return available_metrics

def normalize_data(dfs: List[pd.DataFrame], metrics: List[str]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison."""
    """Return data as-is since scores are already normalized."""
    #return dfs  # No normalization needed!

    
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
    # For scorer outputs, higher scores are generally better
    # No need to invert since score_layouts.py already handles this
    
    normalized_dfs = []
    
    for table_idx, df in enumerate(dfs):
        normalized_df = df.copy()
        
        for metric in metrics:
            if metric in df.columns:
                global_min = all_data[metric].min()
                global_max = all_data[metric].max()
                
                if pd.notna(global_min) and pd.notna(global_max) and global_max != global_min:
                    # Standard normalization: higher score = better performance
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

def create_heatmap_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                       metrics: List[str], use_raw: str, output_path: Optional[str] = None) -> None:
    """Create heatmap visualization with layouts on y-axis and metrics on x-axis."""
    
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs, metrics)
    
    # Prepare data with sorting within each table
    all_data = []
    layout_names = []
    
    for i, (df, table_name) in enumerate(zip(normalized_dfs, table_names)):
        # Collect data for this table
        table_data = []
        table_layout_names = []
        
        for _, row in df.iterrows():
            # Get metric values for this layout
            metric_values = []
            valid_count = 0
            
            for metric in metrics:
                if metric in row and pd.notna(row[metric]):
                    metric_values.append(row[metric])
                    valid_count += 1
                else:
                    metric_values.append(0.0)  # Default for missing data
            
            # Skip layouts with too much missing data
            if valid_count < len(metrics) * 0.5:
                continue
            
            table_data.append(metric_values)
            layout_name = row.get('layout', f'Layout_{len(table_layout_names)+1}')
            table_layout_names.append(layout_name)
        
        if not table_data:
            continue
        
        # Convert to numpy array for sorting
        table_matrix = np.array(table_data)
        
        # Sort this table's layouts by average performance (descending)
        table_averages = np.mean(table_matrix, axis=1)
        sort_indices = np.argsort(table_averages)[::-1]  # Descending order
        
        # Apply sorting
        sorted_table_data = table_matrix[sort_indices]
        sorted_table_names = [table_layout_names[idx] for idx in sort_indices]
        
        # Add to combined data
        all_data.extend(sorted_table_data.tolist())
        layout_names.extend(sorted_table_names)
    
    if not all_data:
        print("No valid data found for heatmap")
        return
    
    # Convert to numpy array
    data_matrix = np.array(all_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.8), max(8, len(layout_names) * 0.3)))
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(layout_names)))
    
    # Format metric labels
    metric_display_names = []
    for metric in metrics:
        # Clean up scorer names for display
        display_name = metric.replace('_', ' ').title()
        metric_display_names.append(display_name)
    
    ax.set_xticklabels(metric_display_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(layout_names, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score (0 = worst, 1 = best)', rotation=270, labelpad=20)
    
    # Add value annotations for smaller matrices
    if len(layout_names) <= 20 and len(metrics) <= 15:
        for i in range(len(layout_names)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                # Use white text for dark cells, black for light cells
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=7, weight='bold')
    
    # Title with sorting info
    sort_info = " (sorted by avg. performance)"
    if len(dfs) > 1:
        sort_info = " (sorted within each table)"
    
    score_type = "raw scores" if use_raw else "weighted scores"
    title = f'Keyboard Layout Comparison Heatmap ({score_type}){sort_info}\n{len(layout_names)} layouts across {len(metrics)} metrics'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Labels
    ax.set_xlabel('Scoring Methods', fontsize=12)
    ax.set_ylabel('Keyboard Layouts', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Modify output path for heatmap
        if output_path.endswith('.png'):
            heatmap_path = output_path.replace('.png', '_heatmap.png')
        else:
            heatmap_path = output_path + '_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Heatmap saved to {heatmap_path}")
    else:
        plt.show()

def create_parallel_plot(dfs: List[pd.DataFrame], table_names: List[str], 
                        metrics: List[str], use_raw: str, output_path: Optional[str] = None) -> None:
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
        # Clean up scorer names for display
        display_name = metric.replace('_', ' ').title()
        # Split long names
        if len(display_name) > 12:
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
    score_type = "raw scores" if use_raw else "weighted scores"
    ax.set_title(f'Keyboard Layout Comparison ({score_type})\n'
                f'Parallel coordinates across {len(metrics)} scoring methods', 
                fontsize=16, fontweight='bold', pad=20)

    if len(dfs) > 1 or len(dfs[0]) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Modify output path for parallel plot
        if output_path.endswith('.png'):
            parallel_path = output_path.replace('.png', '_parallel.png')
        else:
            parallel_path = output_path + '_parallel.png'
        plt.savefig(parallel_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        print(f"Parallel plot saved to {parallel_path}")
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
        description='Create parallel coordinates plots and heatmaps comparing keyboard layouts from score_layouts.py output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_layouts.py --tables layout_scores.csv
  python compare_layouts.py --tables scores1.csv scores2.csv
  python compare_layouts.py --tables scores.csv --use-raw
  python compare_layouts.py --tables scores.csv --output comparison.png
  python compare_layouts.py --tables scores.csv --verbose

Input format:
  CSV files should be output from: score_layouts.py --csv-output
  Expected columns: layout_name,scorer,weighted_score,raw_score
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files containing layout scoring data from score_layouts.py')
    parser.add_argument('--use-raw', action='store_true',
                       help='Use raw scores instead of weighted scores (if available)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (if not specified, plots are shown)')
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
        df = load_and_pivot_data(table_path, args.use_raw, args.verbose)
        
        if len(df) == 0:
            if args.verbose:
                print(f"Warning: No data found in {table_path}")
            continue
            
        dfs.append(df)
        table_names.append(Path(table_path).stem)
    
    if not dfs:
        print("Error: No valid data found in any table")
        sys.exit(1)
    
    # Find available metrics (scorers)
    metrics = find_available_metrics(dfs, args.verbose)
    
    if not metrics:
        print("Error: No scorer metrics found in the data")
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print_summary_stats(dfs, table_names, metrics)
    
    # Create plots
    if args.verbose:
        print(f"\nCreating visualization plots...")
        print(f"Tables: {len(dfs)}")
        print(f"Total layouts: {sum(len(df) for df in dfs)}")
        print(f"Scorer metrics to plot: {len(metrics)}")
        score_type = "raw" if args.use_raw else "weighted"
        print(f"Using {score_type} scores")
    
    # Generate parallel coordinates plot
    create_parallel_plot(dfs, table_names, metrics, args.use_raw, args.output)

    # Generate heatmap plot  
    create_heatmap_plot(dfs, table_names, metrics, args.use_raw, args.output)

if __name__ == "__main__":
    main()