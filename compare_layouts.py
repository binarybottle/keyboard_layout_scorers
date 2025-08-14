#!/usr/bin/env python3
"""
Keyboard layout comparison with metric filtering

Creates parallel coordinates and heatmap plots comparing keyboard layouts 
across performance metrics, and allows filtering to specific metrics in a specified order.

Examples:
    # All available metrics (alphabetical order)
    poetry run python3 compare_layouts.py --tables layout_scores.csv

    # Specific metrics in custom order (1 table)
    poetry run python3 compare_layouts.py --metrics engram comfort comfort-key dvorak7 time_total distance_total --tables layout_scores.csv
    poetry run python3 compare_layouts.py --metrics engram comfort comfort-key dvorak7 dvorak7_repetition dvorak7_movement dvorak7_vertical dvorak7_horizontal dvorak7_adjacent dvorak7_weak dvorak7_outward time_total time_setup time_interval time_return distance_total distance_setup distance_interval distance_return --tables layout_scores.csv

    # Create both plots and rankings with rank-based coloring (1 table)
    poetry run python3 compare_layouts.py --metrics engram comfort comfort-key dvorak7 distance_total time_total --output comparison.png --rankings rankings.csv --tables layout_scores.csv 

    # Compare multiple tables with filtered metrics and output file
    poetry run python3 compare_layouts.py --metrics engram comfort comfort-key dvorak7 time_total distance_total --output output/layout_comparison.png --tables layout_scores.csv moo_layout_scores.csv
    poetry run python3 compare_layouts.py --metrics engram comfort comfort-key dvorak7 dvorak7_repetition dvorak7_movement dvorak7_vertical dvorak7_horizontal dvorak7_adjacent dvorak7_weak dvorak7_outward time_total time_setup time_interval time_return distance_total distance_setup distance_interval distance_return --output output/layout_comparison_detailed.png --tables layout_scores.csv moo_layout_scores.csv

    # Get layout rankings and plots for the metrics applied to the layouts (1 table)
    poetry run python3 compare_layouts.py --metrics comfort comfort-key dvorak7 time_total distance_total --rankings output/layout_rankings.csv --output output/layout_comparison.csv --tables moo_layout_scores.csv
    poetry run python3 compare_layouts.py --metrics comfort comfort-key dvorak7 dvorak7_repetition dvorak7_movement dvorak7_vertical dvorak7_horizontal dvorak7_adjacent dvorak7_weak dvorak7_outward time_total time_setup time_interval time_return distance_total distance_setup distance_interval distance_return --rankings output/layout_rankings_detailed.csv --output output/layout_comparison_detailed.csv --tables moo_layout_scores.csv

Input format:
  CSV files should be output from: score_layouts.py --csv-output
  Expected columns: layout_name,scorer,weighted_score,raw_score
  Optional: layout_string (enables additional layout analysis columns)

Rankings output:
  CSV with columns: layout, [letters, positions, qwerty_letters, qwerty_positions], [metric_ranks], total_rank_sum, [metric_values]
  - letters: layout letters in keyboard order
  - positions: corresponding QWERTY positions for layout letters  
  - qwerty_letters: what letter is at each QWERTY position
  - qwerty_positions: QWERTY reference positions
  Layouts ordered by total rank sum (lower = better overall performance)
  
Rank-based coloring:
  When --rankings is used, parallel plot lines are colored from dark red (best) to light red (worst) based on total rank sum.

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib.cm as cm

def parse_layout_string(layout_string: str) -> tuple:
    """Parse layout string to extract letters and their positions."""
    if pd.isna(layout_string) or not layout_string:
        return "", ""
    
    # QWERTY reference positions
    qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    qwerty_letters = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    # Extract letters from layout string (should be same length as QWERTY)
    layout_letters = layout_string.strip('"').replace('\\', '')  # Clean up quotes and escapes
    
    if len(layout_letters) != len(qwerty_positions):
        # Try to pad or truncate to match QWERTY length
        if len(layout_letters) < len(qwerty_positions):
            layout_letters = layout_letters + ' ' * (len(qwerty_positions) - len(layout_letters))
        else:
            layout_letters = layout_letters[:len(qwerty_positions)]
    
    # Create mapping from layout position to letter
    position_to_letter = {}
    letter_to_position = {}
    
    for i, (pos, letter) in enumerate(zip(qwerty_positions, layout_letters)):
        position_to_letter[pos] = letter
        letter_to_position[letter] = pos
    
    # Create strings for the 4 columns
    letters = layout_letters  # Letters in layout order
    positions = qwerty_positions  # Corresponding QWERTY positions
    
    # Letters in QWERTY order (what letter is at each QWERTY position)
    qwerty_order_letters = ''.join(position_to_letter.get(pos, ' ') for pos in qwerty_positions)
    
    # QWERTY positions (reference)
    qwerty_order_positions = qwerty_positions
    
    return letters, positions, qwerty_order_letters, qwerty_order_positions

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
            
            # If layout_string column exists, preserve it
            if 'layout_string' in df.columns:
                # Get unique layout strings for each layout name
                layout_strings = df.groupby('layout_name')['layout_string'].first()
                pivoted = pivoted.merge(layout_strings.reset_index().rename(columns={'layout_name': 'layout'}), 
                                      on='layout', how='left')
            
            if verbose:
                print(f"Pivoted data shape: {pivoted.shape}")
                print(f"Layouts: {len(pivoted)}")
                print(f"Metrics (scorers): {len(pivoted.columns) - 1}")
                if 'layout_string' in pivoted.columns:
                    print("Layout strings preserved in data")
            
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
    
    # Convert to sorted list (alphabetical order)
    available_metrics = sorted(list(all_metrics))
    
    if verbose:
        print(f"\nFound {len(available_metrics)} scorer metrics available:")
        for i, metric in enumerate(available_metrics):
            print(f"  {i+1:2d}. {metric}")
    
    return available_metrics

def filter_and_order_metrics(dfs: List[pd.DataFrame], requested_metrics: Optional[List[str]] = None, 
                           verbose: bool = False) -> List[str]:
    """Filter and order metrics based on user specification."""
    # Get all available metrics
    available_metrics = find_available_metrics(dfs, verbose)
    
    if not requested_metrics:
        # Return all available metrics in alphabetical order
        return available_metrics
    
    # Filter to only requested metrics that are available
    filtered_metrics = []
    missing_metrics = []
    
    for metric in requested_metrics:
        if metric in available_metrics:
            filtered_metrics.append(metric)
        else:
            missing_metrics.append(metric)
    
    if missing_metrics and verbose:
        print(f"\nWarning: Requested metrics not found in data: {', '.join(missing_metrics)}")
    
    if not filtered_metrics:
        print("Error: None of the requested metrics were found in the data")
        print(f"Available metrics: {', '.join(available_metrics)}")
        sys.exit(1)
    
    if verbose:
        print(f"\nUsing {len(filtered_metrics)} metrics in specified order:")
        for i, metric in enumerate(filtered_metrics):
            print(f"  {i+1:2d}. {metric}")
    
    return filtered_metrics

def normalize_data(dfs: List[pd.DataFrame], metrics: List[str]) -> List[pd.DataFrame]:
    """Normalize all data across tables for fair comparison."""
    # Combine all data to get global min/max for each metric
    all_data = pd.concat(dfs, ignore_index=True)
    
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
                        metrics: List[str], use_raw: str, output_path: Optional[str] = None,
                        rankings_df: Optional[pd.DataFrame] = None) -> None:
    """Create parallel coordinates plot with optional rank-based coloring."""
    # Normalize data across all tables
    normalized_dfs = normalize_data(dfs, metrics)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 1.2), 10))
    
    # Determine coloring scheme
    use_ranking_colors = rankings_df is not None and len(rankings_df) > 0
    
    if use_ranking_colors:
        # Create rank-based color mapping
        layout_to_rank = {}
        for idx, (_, row) in enumerate(rankings_df.iterrows()):
            layout_name = row['layout']
            # Use the index (0-based) as the rank position for coloring
            layout_to_rank[layout_name] = idx
        
        total_layouts = len(layout_to_rank)
        
        # Create red color gradient: dark red (best) to light red (worst)
        red_colormap = cm.Reds
        
        # Define color range (avoid pure white, use darker range of reds)
        min_color_val = 0.3  # Light red
        max_color_val = 1.0  # Dark red
        
        print(f"Using ranking-based coloring for {total_layouts} layouts")
    else:
        # Use original table-based coloring
        colors = get_colors(len(dfs))
    
    # Plot parameters
    x_positions = range(len(metrics))
    
    # Plot each table's data
    for i, (df, table_name) in enumerate(zip(normalized_dfs, table_names)):
        valid_layout_count = 0
        
        if not use_ranking_colors:
            color = colors[i] if i < len(colors) else colors[-1]
        
        for _, row in df.iterrows():
            y_values = [row.get(metric, 0) for metric in metrics]
            
            # Skip rows with too much missing data
            valid_values = [val for val in y_values if pd.notna(val)]
            if len(valid_values) < len(metrics) * 0.5:  # Need at least 50% valid data
                continue
            
            # Replace NaN values with 0
            y_values = [val if pd.notna(val) else 0 for val in y_values]
            
            # Determine line color
            if use_ranking_colors:
                layout_name = row.get('layout', '')
                if layout_name in layout_to_rank:
                    # Calculate color based on rank position
                    rank_position = layout_to_rank[layout_name]
                    # Invert so best rank (0) gets darkest color
                    color_intensity = max_color_val - (rank_position / (total_layouts - 1)) * (max_color_val - min_color_val)
                    color = red_colormap(color_intensity)
                else:
                    # Fallback color for layouts not in rankings
                    color = 'gray'
            # else: color already set above for table-based coloring
            
            ax.plot(x_positions, y_values, color=color, alpha=0.7, linewidth=1.5)
            valid_layout_count += 1
        
        # Add legend entry (only for table-based coloring or first table)
        if not use_ranking_colors:
            ax.plot([], [], color=color, linewidth=3, label=f"{table_name} ({valid_layout_count} layouts)")
        elif i == 0:  # Only add one legend entry for ranking-based coloring
            # Create legend showing color gradient
            ax.plot([], [], color=red_colormap(max_color_val), linewidth=3, label=f"Best ranked layouts")
            ax.plot([], [], color=red_colormap(min_color_val), linewidth=3, label=f"Worst ranked layouts")

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
    
    if use_ranking_colors:
        title_suffix = f'\nRank-ordered visualization: dark red = best, light red = worst'
    else:
        title_suffix = f'\nParallel coordinates across {len(metrics)} scoring methods'
    
    ax.set_title(f'Keyboard Layout Comparison ({score_type}){title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)

    # Show legend
    if len(dfs) > 1 or len(dfs[0]) <= 10 or use_ranking_colors:
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

def create_rankings_table(dfs: List[pd.DataFrame], table_names: List[str], 
                         metrics: List[str], use_raw: bool, rankings_output: str) -> pd.DataFrame:
    """Create a rankings table, save to CSV, and return the rankings DataFrame."""
    
    all_rankings = []
    
    for df, table_name in zip(dfs, table_names):
        # Create a copy for ranking
        ranking_df = df.copy()
        
        # Check if we have the layout column
        if 'layout' not in ranking_df.columns:
            print(f"Warning: No 'layout' column found in {table_name}, skipping rankings")
            continue
        
        # Parse layout strings if available
        has_layout_strings = 'layout_string' in ranking_df.columns
        if has_layout_strings:
            # Parse layout strings to create the 4 new columns
            parsed_data = ranking_df['layout_string'].apply(parse_layout_string)
            ranking_df['letters'] = [x[0] for x in parsed_data]
            ranking_df['positions'] = [x[1] for x in parsed_data]
            ranking_df['qwerty_letters'] = [x[2] for x in parsed_data]
            ranking_df['qwerty_positions'] = [x[3] for x in parsed_data]
        
        # Filter to only include layouts with sufficient data
        valid_layouts = []
        for _, row in ranking_df.iterrows():
            valid_count = sum(1 for metric in metrics if metric in row and pd.notna(row[metric]))
            if valid_count >= len(metrics) * 0.5:  # Need at least 50% valid data
                valid_layouts.append(row.name)
        
        if not valid_layouts:
            print(f"Warning: No layouts with sufficient data in {table_name}")
            continue
        
        ranking_df = ranking_df.loc[valid_layouts].copy()
        
        # Rank each metric (lower rank = better performance)
        # Since higher scores are better, we rank in descending order
        rank_columns = []
        for metric in metrics:
            if metric in ranking_df.columns:
                rank_col = f"{metric}_rank"
                # rank(method='min', ascending=False) gives rank 1 to highest score
                ranking_df[rank_col] = ranking_df[metric].rank(method='min', ascending=False)
                rank_columns.append(rank_col)
            else:
                # If metric is missing, assign worst possible rank
                rank_col = f"{metric}_rank"
                ranking_df[rank_col] = len(ranking_df) + 1
                rank_columns.append(rank_col)
        
        # Calculate total rank sum (lower is better)
        ranking_df['total_rank_sum'] = ranking_df[rank_columns].sum(axis=1)
        
        # Sort by total rank sum (best layouts first)
        ranking_df = ranking_df.sort_values('total_rank_sum')
        
        # Prepare output columns in requested order:
        # layout_name, letters, positions, qwerty_letters, qwerty_positions, ranks, then original metrics
        output_columns = ['layout']
        
        # Add layout string columns if available
        if has_layout_strings:
            output_columns.extend(['letters', 'positions', 'qwerty_letters', 'qwerty_positions'])
        
        # Add rank columns
        output_columns.extend(rank_columns)
        output_columns.append('total_rank_sum')
        
        # Add original metric columns
        output_columns.extend(metrics)
        
        # Add table identifier if multiple tables
        if len(dfs) > 1:
            ranking_df['table'] = table_name
            output_columns.insert(1, 'table')  # Insert after layout name
        
        # Select available columns only
        available_columns = [col for col in output_columns if col in ranking_df.columns]
        
        # Select and store rankings
        table_rankings = ranking_df[available_columns].copy()
        all_rankings.append(table_rankings)
    
    if not all_rankings:
        print("No valid rankings data found")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Combine all tables
    combined_rankings = pd.concat(all_rankings, ignore_index=True)
    
    # If multiple tables, we might want to rank across all tables or keep separate
    if len(dfs) == 1:
        # Single table - rankings are already calculated
        final_rankings = combined_rankings
    else:
        # Multiple tables - provide both per-table and global rankings
        # For now, keep the per-table rankings but sort globally by total_rank_sum
        final_rankings = combined_rankings.sort_values('total_rank_sum')
    
    # Round rank sums to remove floating point precision issues
    if 'total_rank_sum' in final_rankings.columns:
        final_rankings['total_rank_sum'] = final_rankings['total_rank_sum'].round(1)
    
    # Save to CSV
    final_rankings.to_csv(rankings_output, index=False)
    
    print(f"\nRankings saved to {rankings_output}")
    print(f"Best performing layouts (by rank sum):")
    
    # Show top 10 layouts
    display_cols = ['layout']
    if 'table' in final_rankings.columns:
        display_cols.append('table')
    display_cols.extend(['total_rank_sum'])
    
    top_layouts = final_rankings[display_cols].head(10)
    for i, (_, row) in enumerate(top_layouts.iterrows(), 1):
        if 'table' in row:
            print(f"  {i:2d}. {row['layout']} ({row['table']}) - Rank Sum: {row['total_rank_sum']}")
        else:
            print(f"  {i:2d}. {row['layout']} - Rank Sum: {row['total_rank_sum']}")
    
    return final_rankings  # NEW: Return the rankings DataFrame

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
        description='Create parallel coordinates plots and heatmaps comparing keyboard layouts with metric filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All available metrics (alphabetical order)
  python compare_layouts.py --tables layout_scores.csv
  
  # Specific metrics in custom order
  python compare_layouts.py --tables layout_scores.csv --metrics engram comfort comfort-key dvorak7 distance_total time_total
  
  # Create rankings table only (no plots)
  python compare_layouts.py --tables layout_scores.csv --metrics engram comfort dvorak7 --rankings layout_rankings.csv
  
  # Create both plots and rankings with rank-based coloring
  python compare_layouts.py --tables layout_scores.csv --metrics engram comfort comfort-key dvorak7 distance_total time_total --output comparison.png --rankings rankings.csv
  
  # Multiple tables with filtered metrics and rankings
  python compare_layouts.py --tables scores1.csv scores2.csv --metrics comfort distance_total --rankings combined_rankings.csv

Input format:
  CSV files should be output from: score_layouts.py --csv-output
  Expected columns: layout_name,scorer,weighted_score,raw_score
  Optional: layout_string (enables additional layout analysis columns)

Rankings output:
  CSV with columns: layout, [letters, positions, qwerty_letters, qwerty_positions], [metric_ranks], total_rank_sum, [metric_values]
  - letters: layout letters in keyboard order
  - positions: corresponding QWERTY positions for layout letters  
  - qwerty_letters: what letter is at each QWERTY position
  - qwerty_positions: QWERTY reference positions
  Layouts ordered by total rank sum (lower = better overall performance)
  
Rank-based coloring:
  When --rankings is used, parallel plot lines are colored from dark red (best) to light red (worst) based on total rank sum.
        """
    )
    
    parser.add_argument('--tables', nargs='+', required=True,
                       help='One or more CSV files containing layout scoring data from score_layouts.py')
    parser.add_argument('--metrics', nargs='*',
                       help='Specific metrics to include (in order). If not specified, all available metrics are used alphabetically.')
    parser.add_argument('--use-raw', action='store_true',
                       help='Use raw scores instead of weighted scores (if available)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (if not specified, plots are shown)')
    parser.add_argument('--rankings', 
                       help='Create rankings table and save to CSV file (e.g., --rankings rankings.csv)')
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
    
    # Filter and order metrics based on user specification
    metrics = filter_and_order_metrics(dfs, args.metrics, args.verbose)
    
    if not metrics:
        print("Error: No valid metrics found")
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print_summary_stats(dfs, table_names, metrics)
    
    # Create rankings table if requested
    rankings_df = None
    if args.rankings:
        if args.verbose:
            print(f"\nCreating rankings table...")
        rankings_df = create_rankings_table(dfs, table_names, metrics, args.use_raw, args.rankings)
    
    # Create plots (unless only rankings requested)
    if args.output is not None or not args.rankings:
        if args.verbose:
            print(f"\nCreating visualization plots...")
            print(f"Tables: {len(dfs)}")
            print(f"Total layouts: {sum(len(df) for df in dfs)}")
            print(f"Metrics to plot: {len(metrics)} - {', '.join(metrics)}")
            score_type = "raw" if args.use_raw else "weighted"
            print(f"Using {score_type} scores")
            if rankings_df is not None and len(rankings_df) > 0:
                print(f"Using rank-based coloring for parallel plot")
        
        # Generate parallel coordinates plot with optional rankings
        create_parallel_plot(dfs, table_names, metrics, args.use_raw, args.output, rankings_df)

        # Generate heatmap plot (unchanged)
        create_heatmap_plot(dfs, table_names, metrics, args.use_raw, args.output)

if __name__ == "__main__":
    main()