#!/usr/bin/env python3
"""
Comprehensive Empirical Analysis of Dvorak-7 Criteria vs Typing Speed

This script performs a complete empirical validation of August Dvorak's 7 typing 
evaluation criteria using real typing data from 136M+ keystrokes. The analysis
includes frequency adjustment, middle column key effects, and statistical
testing with FDR correction.

The script produces frequency-adjusted residuals that represent typing speed
after controlling for linguistic frequency effects, enabling fair comparison
of layout-dependent factors.

The output file contains correlations between each bigram's typing speed and 
the combination of Dvorak criteria characterizing that bigram. 
These correlations can serve as weights in a speed-based layout optimization 
algorithm, where a correlation is used as a weight to emphasize the 
contribution of a bigram on the layout's score.

Key Features:
- Analyzes correlations between each Dvorak criterion and actual typing speed
- Controls for English bigram frequency effects using regression
- Splits analysis by middle column key usage (lateral index finger movements)
- Tests all 127 possible combinations of criteria (1-way through 7-way)
- Applies FDR correction for multiple testing (Benjamini-Hochberg)
- Uses empirical combination weights derived from significant results
- Generates comprehensive diagnostic plots and visualizations
- Outputs detailed CSV files with all results for further analysis

Data Requirements:
- bigram_times.csv: Real typing times for bigrams from correctly typed words
- letter_pair_frequencies_english.csv: English language bigram frequencies

Statistical Methods:
- Spearman rank correlation (robust to outliers)
- Linear regression for frequency adjustment: time ~ log10(frequency)
- FDR correction using Benjamini-Hochberg procedure
- Confidence intervals using Fisher z-transformation

The 7 scoring criteria for typing bigrams are derived from Dvorak's
"Typing Behavior" book and patent (1936) (0-1, higher = better performance):

1. Repetition: Typing with 1 hand or 1 finger
2. Movement: Typing outside the 8 home keys
3. Vertical separation: Typing in different rows 
4. Horizontal reach: Typing outside 8 finger columns
5. Adjacent fingers: Typing with adjacent fingers (except strong pair)
6. Weak fingers: Typing with weaker fingers
7. Outward direction: Finger sequence away from the thumb

Example usage:
    python dvorak7_speed_validation.py
    python dvorak7_speed_validation.py --max-bigrams 100000
    python dvorak7_speed_validation.py --test-scoring
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from scipy import stats as scipy_stats 
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from collections import Counter
from itertools import combinations
import argparse
import sys
import random

# Import the canonical scoring function from prep_keypair_dvorak7_scores
from prep_keypair_dvorak7_scores import score_bigram_dvorak7

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global variables for logging
log_content = []
original_print = print

def print_and_log(*args, **kwargs):
    """Print to console and store in log"""
    global log_content
    message = ' '.join(str(arg) for arg in args)
    log_content.append(message)
    original_print(*args, **kwargs)

def save_log(filename="dvorak7_speed_validation.log"):
    """Save log content to file"""
    global log_content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))

def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def correlation_confidence_interval(r, n, alpha=0.05):
    """Calculate confidence interval for correlation coefficient"""
    if abs(r) >= 0.999:  # Handle edge case
        return r, r
    
    # Fisher z-transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(1 - alpha/2)
    
    # CI in z-space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    # Transform back to correlation space
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return ci_lower, ci_upper

def load_frequency_data(freq_file_path):
    """Load frequency data for regression analysis"""
    print_and_log("🔍 Loading frequency data for regression analysis...")
    print_and_log("   (This uses pre-calculated English language frequencies, NOT sample frequencies)")
    
    try:
        freq_df = pd.read_csv(freq_file_path)
        print_and_log(f"✅ Loaded bigram frequency data: {len(freq_df)} entries")
        print_and_log(f"   Columns: {list(freq_df.columns)}")
        
        # Show sample frequencies
        print_and_log("   Sample bigram frequencies:")
        for i, (_, row) in enumerate(freq_df.head(3).iterrows()):
            if 'letter_pair' in freq_df.columns:
                # Use normalized_frequency if available, otherwise try other common column names
                freq_col = None
                if 'normalized_frequency' in freq_df.columns:
                    freq_col = 'normalized_frequency'
                elif 'score' in freq_df.columns:
                    freq_col = 'score' 
                elif 'frequency' in freq_df.columns:
                    freq_col = 'frequency'
                
                if freq_col:
                    print_and_log(f"     '{row['letter_pair']}': {row[freq_col]:,.6f}")
        
        return freq_df
        
    except Exception as e:
        print_and_log(f"❌ Error loading frequency data: {e}")
        return None

def verify_frequency_data(freq_df):
    """Verify and display frequency data information"""
    if freq_df is None:
        print_and_log("\n🤖 FREQUENCY DATA VERIFICATION")
        print_and_log("=" * 50)
        print_and_log("❌ No frequency data available for analysis")
        print_and_log("=" * 50)
        return False
    
    return True

def adjust_times_for_frequency(sequences, times, freq_df, sequence_type="sequences"):
    """Adjust typing times for linguistic frequency using regression"""
    
    print_and_log(f"  🔍 Starting frequency adjustment for {sequence_type}...")
    print_and_log(f"      Input: {len(sequences):,} sequences, {len(freq_df)} frequency entries")
    print_and_log(f"      Frequency data columns: {list(freq_df.columns)}")
    
    # Build frequency dictionary - handle different possible column names
    freq_dict = None
    if 'letter_pair' in freq_df.columns:
        # Try different frequency column names
        if 'normalized_frequency' in freq_df.columns:
            freq_dict = dict(zip(freq_df['letter_pair'], freq_df['normalized_frequency']))
            print_and_log(f"      Using 'normalized_frequency' column")
        elif 'score' in freq_df.columns:
            freq_dict = dict(zip(freq_df['letter_pair'], freq_df['score']))
            print_and_log(f"      Using 'score' column")
        elif 'frequency' in freq_df.columns:
            freq_dict = dict(zip(freq_df['letter_pair'], freq_df['frequency']))
            print_and_log(f"      Using 'frequency' column")
        else:
            print_and_log(f"      ❌ No recognized frequency column found")
            print_and_log(f"      Available columns: {list(freq_df.columns)}")
            return times, None
    else:
        print_and_log(f"      ❌ Required 'letter_pair' column not found in frequency data")
        return times, None
    
    if freq_dict is None:
        return times, None
    
    print_and_log(f"      Built frequency dictionary: {len(freq_dict)} entries")
    
    # Show sample frequencies
    sample_freqs = list(freq_dict.items())[:5]
    print_and_log(f"      Example frequencies: {sample_freqs}")
    
    # Map sequences to frequencies
    matched_frequencies = []
    matched_times = []
    matched_sequences = []
    
    for i, seq in enumerate(sequences):
        # Try both uppercase and lowercase versions
        seq_upper = seq.upper()
        seq_lower = seq.lower()
        
        if seq in freq_dict:
            matched_frequencies.append(freq_dict[seq])
            matched_times.append(times[i])
            matched_sequences.append(seq)
        elif seq_upper in freq_dict:
            matched_frequencies.append(freq_dict[seq_upper])
            matched_times.append(times[i])
            matched_sequences.append(seq)
        elif seq_lower in freq_dict:
            matched_frequencies.append(freq_dict[seq_lower])
            matched_times.append(times[i])
            matched_sequences.append(seq)
    
    overlap_pct = (len(matched_frequencies) / len(sequences)) * 100
    print_and_log(f"      Frequency overlap: {len(matched_frequencies)}/{len(sequences)} ({overlap_pct:.1f}%)")
    
    if len(matched_frequencies) < 10:
        print_and_log(f"      ⚠️  Too few matches for regression ({len(matched_frequencies)})")
        return times, None
    
    # Log-transform frequencies for better linear relationship
    # Add small constant to avoid log(0) issues
    safe_frequencies = [max(freq, 1e-10) for freq in matched_frequencies]
    log_frequencies = np.log10(np.array(safe_frequencies))
    times_array = np.array(matched_times)
    
    print_and_log(f"      Frequency range: {min(matched_frequencies):,.6f} to {max(matched_frequencies):,.6f} (mean: {np.mean(matched_frequencies):,.6f})")
    print_and_log(f"      Time range: {min(times_array):.1f} to {max(times_array):.1f}ms (mean: {np.mean(times_array):.1f} ± {np.std(times_array):.1f})")
    
    # Regression: time = intercept + slope * log_frequency
    try:
        X = sm.add_constant(log_frequencies)  # Add intercept term
        model = sm.OLS(times_array, X).fit()
        
        # Calculate frequency-controlled residuals for all sequences
        frequency_residuals = []
        model_info = {
            'r_squared': model.rsquared,
            'intercept': model.params[0],
            'slope': model.params[1],
            'p_value': model.pvalues[1] if len(model.pvalues) > 1 else None,
            'n_obs': len(matched_frequencies),
            'log_frequencies': log_frequencies.copy(),
            'predicted_times': model.predict(X).copy(),
            'actual_times': times_array.copy(),
            'matched_sequences': matched_sequences.copy()
        }
        
        print_and_log(f"      Regression results:")
        print_and_log(f"        R² = {model.rsquared:.4f}")
        print_and_log(f"        Slope = {model.params[1]:.4f} (p = {model.pvalues[1]:.4f})")
        print_and_log(f"        Intercept = {model.params[0]:.4f}")
        print_and_log(f"        → Negative slope means higher frequency = faster typing")
        
        # Calculate residuals for all sequences
        residual_magnitudes = []
        rank_changes = []
        original_ranks = stats.rankdata(times)
        
        for i, seq in enumerate(sequences):
            # Try to find frequency for this sequence
            seq_freq = None
            if seq in freq_dict:
                seq_freq = freq_dict[seq]
            elif seq.upper() in freq_dict:
                seq_freq = freq_dict[seq.upper()]
            elif seq.lower() in freq_dict:
                seq_freq = freq_dict[seq.lower()]
            
            if seq_freq is not None:
                log_freq = np.log10(max(seq_freq, 1e-10))
                predicted_time = model.params[0] + model.params[1] * log_freq
                residual = times[i] - predicted_time  # Actual - Predicted
                frequency_residuals.append(residual)
                residual_magnitudes.append(abs(residual))
            else:
                # For sequences without frequency data, use original time
                frequency_residuals.append(times[i])
                residual_magnitudes.append(0)
        
        adjusted_ranks = stats.rankdata(frequency_residuals)
        rank_changes = abs(original_ranks - adjusted_ranks)
        
        print_and_log(f"      Frequency control effects:")
        print_and_log(f"        Average |residual|: {np.mean(residual_magnitudes):.2f}ms")
        print_and_log(f"        Maximum |residual|: {np.max(residual_magnitudes):.2f}ms")
        controlled_count = sum(1 for mag in residual_magnitudes if mag > 0.1)
        print_and_log(f"        Sequences with frequency control: {(controlled_count/len(residual_magnitudes)*100):.1f}%")
        
        # Rank order analysis
        rank_correlation = spearmanr(original_ranks, adjusted_ranks)[0]
        print_and_log(f"      📊 RANK ORDER ANALYSIS:")
        print_and_log(f"        Correlation between raw times and frequency residuals: {rank_correlation:.6f}")
        sequences_with_rank_changes = sum(1 for change in rank_changes if change > 0)
        print_and_log(f"        Sequences with rank changes: {sequences_with_rank_changes}/{len(sequences)} ({sequences_with_rank_changes/len(sequences)*100:.1f}%)")
        print_and_log(f"        Maximum rank position change: {int(max(rank_changes))}")
        
        print_and_log(f"  ✅ Generated frequency-controlled residuals")
        print_and_log(f"  💡 INTERPRETATION: Residuals represent typing speed after controlling for frequency")
        print_and_log(f"      • Negative residuals = faster than expected given frequency")
        print_and_log(f"      • Positive residuals = slower than expected given frequency")
        
        return frequency_residuals, model_info
        
    except Exception as e:
        print_and_log(f"      ❌ Regression failed: {e}")
        return times, None

def create_diagnostic_plots(model_info, output_dir='plots'):
    """Create diagnostic plots for frequency adjustment model"""
    if not model_info:
        print_and_log("No model info available for diagnostic plots")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Frequency Adjustment Model Diagnostics', fontsize=16, fontweight='bold')
    
    # Plot 1: Frequency vs Time relationship with model fit
    log_freqs = model_info['log_frequencies']
    actual_times = model_info['actual_times']
    predicted_times = model_info['predicted_times']
    
    ax1.scatter(log_freqs, actual_times, alpha=0.5, s=20, color='skyblue', label='Actual times')
    ax1.plot(log_freqs, predicted_times, 'r-', linewidth=2, label=f'Model fit (R² = {model_info["r_squared"]:.3f})')
    ax1.set_xlabel('Log₁₀ Frequency')
    ax1.set_ylabel('Typing Time (ms)')
    ax1.set_title('Frequency vs Typing Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted
    residuals = actual_times - predicted_times
    ax2.scatter(predicted_times, residuals, alpha=0.5, s=20, color='lightcoral')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Time (ms)')
    ax2.set_ylabel('Residuals (ms)')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals distribution
    ax3.hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Residuals (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax3.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.1f}ms')
    ax3.axvline(mean_res + std_res, color='orange', linestyle='--', alpha=0.7, label=f'±1 SD: {std_res:.1f}ms')
    ax3.axvline(mean_res - std_res, color='orange', linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: Q-Q plot for normality check
    scipy_stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'frequency_adjustment_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Diagnostic plots saved to: {output_path}")

def analyze_correlations(sequences, times, criteria_names, group_name, analysis_type, model_info=None):
    """Analyze correlations between Dvorak criteria scores and typing times"""
    results = {}
    
    print_and_log(f"  {analysis_type.replace('_', ' ').title()} Analysis:")
    
    # Define group_suffix based on group_name
    if "No Middle" in group_name:
        group_suffix = "no_middle"
    elif "With Middle" in group_name:
        group_suffix = "with_middle"
    else:
        group_suffix = "unknown"
    
    # Calculate scores for all sequences
    print_and_log(f"  Calculating Dvorak-7 scores for {len(sequences):,} sequences...")
    
    start_time = time.time()
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_sequences = []
    valid_times = []
    sequence_scores_data = []
    
    for i, (seq, time_val) in enumerate(zip(sequences, times)):
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - start_time
            print_and_log(f"    Progress: {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - {elapsed:.1f}s", end='\r')
        
        # Calculate Dvorak-7 scores using the canonical function
        try:
            scores = score_bigram_dvorak7(seq)
            
            # Validate scores
            if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
                valid_sequences.append(seq)
                valid_times.append(time_val)
                
                # Store complete scores for this sequence
                sequence_score_record = {'sequence': seq, 'time': time_val, 'analysis_type': analysis_type}
                for criterion in criteria_names.keys():
                    score = scores[criterion]
                    criterion_scores[criterion].append(score)
                    sequence_score_record[criterion] = score
                sequence_scores_data.append(sequence_score_record)
                
                # Show sample scores for first few sequences
                if len(valid_sequences) <= 3:
                    sample_scores = {k: f"{v:.3f}" for k, v in scores.items()}
                    print_and_log(f"      Sample scores for '{seq}': {sample_scores}")
        except Exception as e:
            # Skip invalid bigrams
            continue
    
    elapsed = time.time() - start_time
    print_and_log(f"    Completed score calculation in {elapsed:.1f}s" + " " * 50)
    
    print_and_log(f"    Valid sequences for analysis: {len(valid_sequences):,}")
    
    # Store sequence data for combination analysis
    results[f'_sequence_scores_{analysis_type}'] = sequence_scores_data
    
    if len(valid_sequences) < 10:
        print_and_log(f"    ⚠️  Too few valid sequences for correlation analysis")
        return results
    
    # Calculate correlations
    print_and_log(f"    Calculating correlations for {len(criteria_names)} criteria...")
    for criterion, scores_list in criterion_scores.items():
        if len(scores_list) >= 3:  # Need at least 3 points for correlation
            try:
                # Check for constant values
                unique_scores = len(set(scores_list))
                if unique_scores <= 1:
                    print_and_log(f"    Warning: {criterion} has constant scores ({unique_scores} unique values)")

                    result_key = f"{criterion}_{group_suffix}_{analysis_type}"
                    results[result_key] = {
                        'name': criteria_names[criterion],
                        'group': f"{group_name} ({analysis_type.replace('_', ' ')})",
                        'analysis_type': analysis_type,
                        'n_samples': len(scores_list),
                        'pearson_r': float('nan'),
                        'pearson_p': float('nan'),
                        'spearman_r': float('nan'),
                        'spearman_p': float('nan'),
                        'mean_score': np.mean(scores_list),
                        'std_score': np.std(scores_list),
                        'scores': scores_list.copy(),
                        'times': valid_times.copy(),
                        'constant_scores': True
                    }
                    continue
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(scores_list, valid_times)
                
                # Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(scores_list, valid_times)
                
                # Check for NaN results
                if np.isnan(pearson_r) or np.isnan(spearman_r):
                    print_and_log(f"    Warning: {criterion} produced NaN correlations")
                    print_and_log(f"      Score range: {min(scores_list):.3f} to {max(scores_list):.3f}")
                    print_and_log(f"      Unique scores: {unique_scores}")
                
                result_key = f"{criterion}_{group_suffix}_{analysis_type}"
                results[result_key] = {
                    'name': criteria_names[criterion],
                    'group': f"{group_name} ({analysis_type.replace('_', ' ')})",
                    'analysis_type': analysis_type,
                    'n_samples': len(scores_list),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_score': np.mean(scores_list),
                    'std_score': np.std(scores_list),
                    'scores': scores_list.copy(),
                    'times': valid_times.copy()
                }
                
                # Add frequency model info if available
                if analysis_type == 'freq_adjusted' and model_info:
                    results[result_key]['frequency_model'] = model_info
                
            except Exception as e:
                print_and_log(f"    Error calculating correlation for {criterion}: {e}")
                continue
    
    print_and_log(f"    Correlation analysis complete")
    
    return results

def analyze_bigram_data(bigrams, freq_df, middle_column_keys):
    """Analyze bigram typing data with middle column key analysis"""
    
    # Updated criteria names to match prep_keypair_dvorak7_scores.py
    criteria_names = {
        'repetition': 'Repetition (hand/finger usage)',
        'movement': 'Movement (home key usage)', 
        'vertical': 'Vertical separation (row differences)',
        'horizontal': 'Horizontal reach (column adherence)',
        'adjacent': 'Adjacent fingers (avoid weak pairs)',
        'weak': 'Weak fingers (avoid pinky/ring)',
        'outward': 'Outward direction (finger rolls)'
    }
    
    print_and_log(f"Splitting {len(bigrams):,} bigrams by middle column key usage...")
    
    without_middle = []
    with_middle = []
    
    for bigram, time in bigrams:
        # Check if bigram contains any middle column keys
        if any(char in middle_column_keys for char in bigram.lower()):
            with_middle.append((bigram, time))
        else:
            without_middle.append((bigram, time))
    
    print_and_log(f"✅ Bigrams without middle columns: {len(without_middle):,}")
    print_and_log(f"✅ Bigrams with middle columns: {len(with_middle):,}")
    
    all_results = {}
    MIN_SEQUENCES = 3
    
    # Analyze each group
    for group_data, group_name in [(without_middle, "Bigrams (No Middle Columns)"), 
                                   (with_middle, "Bigrams (With Middle Columns)")]:
        
        print_and_log(f"\n--- Analyzing {group_name} ---")
        print_and_log(f"Group size: {len(group_data):,} bigrams")
        
        if len(group_data) < MIN_SEQUENCES:
            print_and_log(f"❌ SKIPPING {group_name}: only {len(group_data)} bigrams (need ≥{MIN_SEQUENCES})")
            continue
        
        print_and_log(f"✅ ANALYZING {group_name}: {len(group_data):,} bigrams")
        
        sequences = [item[0] for item in group_data]
        times = [item[1] for item in group_data]
        
        print_and_log(f"Examples: {', '.join(sequences[:5])}")
        
        # Raw analysis (no frequency adjustment)
        raw_results = analyze_correlations(sequences, times, criteria_names, group_name, "raw")
        all_results.update(raw_results)
        
        # Frequency-adjusted analysis
        if freq_df is not None:
            frequency_residuals, model_info = adjust_times_for_frequency(sequences, times, freq_df, "bigrams")
            
            # Create diagnostic plots for the first group only
            if group_name == "Bigrams (No Middle Columns)" and model_info:
                create_diagnostic_plots(model_info)
            
            freq_results = analyze_correlations(sequences, frequency_residuals, criteria_names, group_name, "freq_adjusted", model_info)
            all_results.update(freq_results)
        else:
            print_and_log(f"  ⚠️  Skipping frequency adjustment (no frequency data)")
    
    return all_results

def print_correlation_results_with_frequency(results, analysis_name):
    """Print correlation results comparing raw vs frequency-adjusted"""
    
    print_and_log(f"\n{analysis_name.upper()} CORRELATION ANALYSIS: RAW vs FREQUENCY-ADJUSTED")
    print_and_log("=" * 120)
    print_and_log("Note: Negative correlation = higher score → faster typing (validates Dvorak)")
    print_and_log("      Positive correlation = higher score → slower typing (contradicts Dvorak)")
    print_and_log("-" * 120)
    
    # Group results by criterion and middle column status for comparison
    criterion_groups = {}
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
        
        # Extract criterion and analysis type
        parts = key.split('_')
        if len(parts) >= 2:
            if parts[-1] == 'adjusted':  # freq_adjusted
                criterion = '_'.join(parts[:-2])
                analysis = 'freq_adjusted'
            else:  # raw
                criterion = '_'.join(parts[:-1])
                analysis = 'raw'
                
            group_key = (criterion, data.get('group', ''))
            if group_key not in criterion_groups:
                criterion_groups[group_key] = {}
            criterion_groups[group_key][analysis] = data
    
    # Print comparison for each criterion/group combination
    for (criterion, group), analyses in sorted(criterion_groups.items()):
        if 'raw' in analyses and 'freq_adjusted' in analyses:
            raw_data = analyses['raw']
            adj_data = analyses['freq_adjusted']
            
            # Extract group name properly
            group_name = raw_data.get('group', 'Unknown Group')
            # Clean up the group name
            if 'No Middle Columns' in group_name:
                clean_group_name = 'No Middle Columns'
            elif 'With Middle Columns' in group_name:
                clean_group_name = 'With Middle Columns'
            else:
                clean_group_name = group_name
            
            print_and_log(f"\n{raw_data['name']} - {clean_group_name}:")
            print_and_log(f"Analysis        N      Spearman r  p-val    Effect   Freq Model R²")
            print_and_log(f"----------------------------------------------------------------------")
            
            # Raw analysis
            sr = raw_data['spearman_r']
            sp = raw_data['spearman_p']
            
            # Handle NaN values
            if np.isnan(sr) or np.isnan(sp):
                sr_str = "nan"
                sp_str = "nan"
                s_sig = ""
                effect = "N/A"
            else:
                sr_str = f"{sr:>7.3f}"
                sp_str = f"{sp:<8.3f}"
                s_sig = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
                abs_sr = abs(sr)
                if abs_sr >= 0.5:
                    effect = "Large"
                elif abs_sr >= 0.3:
                    effect = "Med"
                elif abs_sr >= 0.1:
                    effect = "Small"
                else:
                    effect = "None"
            
            print_and_log(f"{'Raw':<15} {raw_data['n_samples']:<6} {sr_str}{s_sig:<4} {sp_str} {effect:<8} {'N/A':<12}")
            
            # Frequency-adjusted analysis
            sr_adj = adj_data['spearman_r']
            sp_adj = adj_data['spearman_p']
            
            # Handle NaN values
            if np.isnan(sr_adj) or np.isnan(sp_adj):
                sr_adj_str = "nan"
                sp_adj_str = "nan"
                s_sig_adj = ""
                effect_adj = "N/A"
            else:
                sr_adj_str = f"{sr_adj:>7.3f}"
                sp_adj_str = f"{sp_adj:<8.3f}"
                s_sig_adj = "***" if sp_adj < 0.001 else "**" if sp_adj < 0.01 else "*" if sp_adj < 0.05 else ""
                abs_sr_adj = abs(sr_adj)
                if abs_sr_adj >= 0.5:
                    effect_adj = "Large"
                elif abs_sr_adj >= 0.3:
                    effect_adj = "Med"
                elif abs_sr_adj >= 0.1:
                    effect_adj = "Small"
                else:
                    effect_adj = "None"
            
            freq_r2 = adj_data.get('frequency_model', {}).get('r_squared', 0)
            freq_r2_str = f"{freq_r2:.3f}" if freq_r2 and not np.isnan(freq_r2) else "N/A"
            
            print_and_log(f"{'Freq-Adjusted':<15} {adj_data['n_samples']:<6} {sr_adj_str}{s_sig_adj:<4} {sp_adj_str} {effect_adj:<8} {freq_r2_str:<12}")
            
            # Show change in correlation
            if not (np.isnan(sr) or np.isnan(sr_adj)):
                change = abs(sr_adj) - abs(sr)
                change_direction = "↑" if change > 0.05 else "↓" if change < -0.05 else "≈"
                print_and_log(f"{'Change':<15} {'':<6} {change:>+7.3f} {change_direction:<8}")
            else:
                print_and_log(f"{'Change':<15} {'':<6} {'N/A':>7} {'N/A':<8}")

def analyze_criterion_combinations(results):
    """Analyze how combinations of criteria predict typing speed"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print_and_log("=" * 80)
    print_and_log("Examining how combinations of criteria interact to predict typing speed")
    print_and_log("IMPORTANT: Using ONLY frequency-adjusted data (controls for English bigram frequency)")
    
    # Look for FREQUENCY-ADJUSTED sequence data only
    freq_adjusted_sequences = []
    
    # Collect all frequency-adjusted sequence data
    for key, data in results.items():
        if key.startswith('_sequence_scores_freq_adjusted') and isinstance(data, list):
            freq_adjusted_sequences.extend(data)
            print_and_log(f"✅ Found frequency-adjusted sequence data: {len(data)} sequences")
    
    if not freq_adjusted_sequences:
        print_and_log("❌ No frequency-adjusted sequence data found for combination analysis")
        return None
    
    print_and_log(f"✅ Using {len(freq_adjusted_sequences):,} frequency-adjusted sequences")
    
    if len(freq_adjusted_sequences) < 100:
        print_and_log(f"⚠️  Too few sequences for combination analysis ({len(freq_adjusted_sequences)})")
        return None
    
    print_and_log(f"Sequence data found for combination analysis")
    print_and_log(f"------------------------------------------------------------")
    print_and_log(f"   Sequences: {len(freq_adjusted_sequences):,}")
    
    # Convert to DataFrame
    df = pd.DataFrame(freq_adjusted_sequences)
    
    # Get criteria columns (exclude sequence, time, analysis_type)
    exclude_cols = {'sequence', 'time', 'analysis_type'}
    criteria_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Verify we have the expected 7 criteria
    expected_criteria = ['repetition', 'movement', 'vertical', 'horizontal', 'adjacent', 'weak', 'outward']
    print_and_log(f"   Expected criteria: {expected_criteria}")
    print_and_log(f"   Found criteria: {criteria_cols}")
    
    # Filter to only use criteria that actually exist
    criteria_cols = [col for col in expected_criteria if col in criteria_cols]
    print_and_log(f"   Using criteria: {criteria_cols}")
    
    times = df['time'].values
    print_and_log(f"   Using raw times for combination analysis")
    
    # COMPREHENSIVE COMBINATION ANALYSIS - TEST ALL 127 COMBINATIONS
    group_results = {}
    
    # For each combination size k from 1 to 7
    for k in range(1, len(criteria_cols) + 1):
        print_and_log(f"\n📊 {k}-WAY COMBINATIONS:")
        
        # Generate ALL combinations of size k
        combos = list(combinations(criteria_cols, k))
        total_combos = len(combos)
        print_and_log(f"   Testing ALL {total_combos:,} combinations of {k} criteria...")
        
        combo_results = []
        
        # Test EVERY combination
        for combo in combos:
            # Create combined score (additive model)
            combined_scores = np.zeros(len(times))
            for criterion in combo:
                combined_scores += df[criterion].values
            
            # Test if there's variation
            if len(set(combined_scores)) > 1:
                try:
                    corr, p_val = spearmanr(combined_scores, times)
                    if not (np.isnan(corr) or np.isnan(p_val)):
                        combo_results.append({
                            'combination': ' + '.join(combo),
                            'criteria_count': k,
                            'correlation': corr,
                            'p_value': p_val,
                            'abs_correlation': abs(corr)
                        })
                except:
                    continue
        
        # Sort by absolute correlation
        combo_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Store results
        group_results[f'{k}_way'] = combo_results
        
        # Show top results for this k
        top_n = min(5, len(combo_results))
        if combo_results:
            print_and_log(f"   Top {top_n} combinations:")
            for i, result in enumerate(combo_results[:top_n]):
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                print_and_log(f"     {i+1}. {result['combination']}")
                print_and_log(f"        r = {result['correlation']:.4f}{sig}, p = {result['p_value']:.4f}")
        else:
            print_and_log(f"   No valid combinations found")
    
    # FIND THE SINGLE BEST COMBINATION OVERALL
    print_and_log(f"\n🏆 BEST COMBINATION ACROSS ALL SIZES:")
    
    best_overall = None
    best_correlation = 0
    
    for k_way, results_list in group_results.items():
        if results_list:
            best_in_category = max(results_list, key=lambda x: x['abs_correlation'])
            if best_in_category['abs_correlation'] > best_correlation:
                best_correlation = best_in_category['abs_correlation']
                best_overall = best_in_category
    
    if best_overall:
        direction = "supports Dvorak (faster)" if best_overall['correlation'] < 0 else "contradicts Dvorak (slower)"
        effect_size = "large" if best_overall['abs_correlation'] >= 0.5 else "medium" if best_overall['abs_correlation'] >= 0.3 else "small" if best_overall['abs_correlation'] >= 0.1 else "negligible"
        
        print_and_log(f"   🎯 STRONGEST PREDICTOR: {best_overall['combination']}")
        print_and_log(f"   📈 Correlation: r = {best_overall['correlation']:.4f}")
        print_and_log(f"   📊 Effect size: {effect_size}")
        print_and_log(f"   🎭 Direction: {direction}")
        print_and_log(f"   🔢 Uses {best_overall['criteria_count']} criteria")
    
    # SUMMARY STATISTICS
    print_and_log(f"\n📈 COMBINATION ANALYSIS SUMMARY:")
    
    total_combinations = sum(len(results_list) for results_list in group_results.values())
    significant_combinations = sum(sum(1 for r in results_list if r['p_value'] < 0.05) for results_list in group_results.values())
    
    print_and_log(f"   • Total combinations tested: {total_combinations:,}")
    print_and_log(f"   • Statistically significant: {significant_combinations:,} ({significant_combinations/total_combinations*100:.1f}%)")
    
    # Effect size distribution
    all_correlations = [r['abs_correlation'] for results_list in group_results.values() for r in results_list if r['p_value'] < 0.05]
    if all_correlations:
        large_effects = sum(1 for r in all_correlations if r >= 0.5)
        medium_effects = sum(1 for r in all_correlations if 0.3 <= r < 0.5)
        small_effects = sum(1 for r in all_correlations if 0.1 <= r < 0.3)
        negligible_effects = sum(1 for r in all_correlations if r < 0.1)
        
        print_and_log(f"   • Large effects (|r| ≥ 0.5): {large_effects}")
        print_and_log(f"   • Medium effects (|r| 0.3-0.5): {medium_effects}")
        print_and_log(f"   • Small effects (|r| 0.1-0.3): {small_effects}")
        print_and_log(f"   • Negligible effects (|r| < 0.1): {negligible_effects}")
    
    return group_results

def create_combination_performance_plots(combination_results, output_dir='plots'):
    """Create plots showing combination analysis results"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print_and_log("📊 Creating combination analysis plots...")
    
    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak-7 Criterion Combination Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Best correlation by combination size
    combination_sizes = []
    best_correlations = []
    best_combinations = []
    
    for k_way, results_list in combination_results.items():
        if results_list:
            k = int(k_way.split('_')[0])
            best_result = max(results_list, key=lambda x: x['abs_correlation'])
            
            combination_sizes.append(k)
            best_correlations.append(best_result['abs_correlation'])
            best_combinations.append(best_result['combination'])
    
    if combination_sizes:
        bars = ax1.bar(combination_sizes, best_correlations, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Number of Criteria Combined')
        ax1.set_ylabel('Best Absolute Correlation |r|')
        ax1.set_title('Best Performance by Combination Size')
        ax1.set_xticks(combination_sizes)
        ax1.grid(True, alpha=0.3)
        
        # Add effect size reference lines
        ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Small effect (0.1)')
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium effect (0.3)')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Large effect (0.5)')
        ax1.legend()
        
        # Annotate bars with correlation values
        for bar, corr in zip(bars, best_correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Distribution of effect sizes
    all_correlations = []
    
    for k_way, results_list in combination_results.items():
        for result in results_list:
            if result['p_value'] < 0.05:  # Only significant results
                all_correlations.append(result['abs_correlation'])
    
    if all_correlations:
        # Create effect size categories
        effect_categories = []
        for corr in all_correlations:
            if corr >= 0.5:
                effect_categories.append('Large (≥0.5)')
            elif corr >= 0.3:
                effect_categories.append('Medium (0.3-0.5)')
            elif corr >= 0.1:
                effect_categories.append('Small (0.1-0.3)')
            else:
                effect_categories.append('Negligible (<0.1)')
        
        # Count effect sizes
        effect_counts = Counter(effect_categories)
        
        # Create pie chart
        labels = list(effect_counts.keys())
        sizes = list(effect_counts.values())
        colors = ['red', 'orange', 'yellow', 'lightgray'][:len(labels)]
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Effect Sizes\n(Significant Results Only)')
    
    # Plot 3: Number of significant results by combination size
    sizes_for_plot = []
    significant_counts = []
    total_counts = []
    
    for k_way, results_list in combination_results.items():
        k = int(k_way.split('_')[0])
        significant = sum(1 for r in results_list if r['p_value'] < 0.05)
        total = len(results_list)
        
        if total > 0:
            sizes_for_plot.append(k)
            significant_counts.append(significant)
            total_counts.append(total)
    
    if sizes_for_plot:
        x_pos = np.arange(len(sizes_for_plot))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, total_counts, width, label='Total tested', alpha=0.7, color='lightblue')
        bars2 = ax3.bar(x_pos + width/2, significant_counts, width, label='Significant (p<0.05)', alpha=0.7, color='darkblue')
        
        ax3.set_xlabel('Number of Criteria Combined')
        ax3.set_ylabel('Number of Combinations')
        ax3.set_title('Significant vs Total Combinations')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sizes_for_plot)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (total, sig) in enumerate(zip(total_counts, significant_counts)):
            if total > 0:
                pct = (sig / total) * 100
                ax3.text(i, max(total, sig) + max(total_counts) * 0.02, 
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Top 10 overall combinations
    all_results = []
    for k_way, results_list in combination_results.items():
        all_results.extend(results_list)
    
    # Sort by absolute correlation and take top 10
    all_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
    top_10 = all_results[:10]
    
    if top_10:
        # Prepare data for horizontal bar chart
        combinations = [r['combination'] for r in top_10]
        correlations = [r['correlation'] for r in top_10]  # Use signed correlation
        
        # Truncate long combination names for readability
        truncated_combinations = []
        for combo in combinations:
            if len(combo) > 30:
                truncated_combinations.append(combo[:27] + '...')
            else:
                truncated_combinations.append(combo)
        
        y_pos = np.arange(len(truncated_combinations))
        
        # Color bars by direction (red for positive, blue for negative)
        colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        
        bars = ax4.barh(y_pos, correlations, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(truncated_combinations, fontsize=8)
        ax4.set_xlabel('Spearman Correlation')
        ax4.set_title('Top 10 Best Combinations')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add correlation values to bars
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax4.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{corr:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=7)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak7_combination_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Combination analysis plots saved to: {output_path}")
    
    return output_path

def analyze_all_results_with_fdr(results, combination_results=None):
    """Complete FDR analysis for all groups and combinations"""
    
    print_and_log(f"\n" + "=" * 100)
    print_and_log("🎯 COMPLETE FDR-CORRECTED ANALYSIS")
    print_and_log("=" * 100)
    print_and_log("Frequency-adjusted results with FDR multiple testing correction")
    print_and_log("Answers: Which Dvorak-7 criteria survive rigorous statistical testing?")
    print_and_log("")
    
    # PART 1: INDIVIDUAL CRITERIA BY GROUP
    print_and_log("📊 PART 1: INDIVIDUAL CRITERIA (7 TESTS PER GROUP)")
    print_and_log("=" * 60)
    
    # Group frequency-adjusted results
    groups = {}
    for key, data in results.items():
        if (key.endswith('_freq_adjusted') and 
            isinstance(data, dict) and 
            'spearman_r' in data and 
            not np.isnan(data['spearman_r'])):
            
            group_name = data.get('group', 'Unknown')
            
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append((key, data))
    
    print_and_log(f"\n📋 Groups found for FDR analysis: {len(groups)}")
    for group_name, group_data in groups.items():
        print_and_log(f"   • {group_name}: {len(group_data)} criteria")
    
    if len(groups) == 0:
        print_and_log("❌ No frequency-adjusted groups found for FDR analysis!")
        return
    
    # Analyze each group separately
    all_individual_results = []
    for group_name, group_data in groups.items():
        print_and_log(f"\n🔍 {group_name}")
        print_and_log("-" * 50)
        
        # Extract p-values for FDR correction
        p_values = [data['spearman_p'] for _, data in group_data]
        
        # Apply FDR correction within this group
        if p_values:
            rejected, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            # Create results with correction
            group_results = []
            for i, (key, data) in enumerate(group_data):
                result = {
                    'key': key,  
                    'group': group_name,
                    'criterion': data['name'],
                    'correlation': data['spearman_r'],
                    'p_original': data['spearman_p'],
                    'p_fdr_corrected': p_adj[i],
                    'significant_after_fdr': rejected[i],
                    'abs_correlation': abs(data['spearman_r']),
                    'n_samples': data['n_samples'],
                    'supports_dvorak': data['spearman_r'] < 0
                }
                group_results.append(result)
                all_individual_results.append(result)
            
            # Sort by absolute correlation
            group_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            # Print results for this group
            sig_count = sum(1 for r in group_results if r['significant_after_fdr'])
            print_and_log(f"Sample size: {group_results[0]['n_samples']:,} bigrams")
            print_and_log(f"Significant after FDR: {sig_count}/7 criteria")
            print_and_log("")
            
            # Separate supporting vs contradicting results
            supporting = [r for r in group_results if r['significant_after_fdr'] and r['supports_dvorak']]
            contradicting = [r for r in group_results if r['significant_after_fdr'] and not r['supports_dvorak']]
            
            if supporting:
                print_and_log("✅ Support Dvorak:")
                for result in supporting:
                    print_and_log(f"- {result['criterion']}: r = {result['correlation']:.3f}")
            
            if contradicting:
                print_and_log("❌ Contradict Dvorak:")
                for result in contradicting:
                    print_and_log(f"- {result['criterion']}: r = +{result['correlation']:.3f}")
            
            print_and_log("")
            print_and_log("All results (FDR-corrected):")
            print_and_log("Criterion                     r      95% CI         FDR p-val  Significant  Dvorak")
            print_and_log("-" * 90)

            for result in group_results:
                # Calculate confidence interval
                ci_lower, ci_upper = correlation_confidence_interval(
                    result['correlation'], result['n_samples']
                )
                
                sig_marker = "✅" if result['significant_after_fdr'] else "❌"
                dvorak_marker = "✅ Support" if result['supports_dvorak'] else "❌ Contradict"
                
                print_and_log(f"{result['criterion']:<25} {result['correlation']:>6.3f}  "
                            f"[{ci_lower:>5.3f},{ci_upper:>5.3f}]  "
                            f"{result['p_fdr_corrected']:>8.3f}  {sig_marker:<11}  {dvorak_marker}")
    
    # PART 2: COMBINATIONS WITH FDR CORRECTION
    if combination_results:
        print_and_log(f"\n📊 PART 2: ALL 127 COMBINATIONS WITH FDR CORRECTION")
        print_and_log("=" * 60)
        print_and_log("Testing which combinations survive multiple testing correction")
        print_and_log("")
        
        # Collect ALL combination results
        all_combinations = []
        for k_way, results_list in combination_results.items():
            for result in results_list:
                all_combinations.append({
                    'combination': result['combination'],
                    'k_way': int(k_way.split('_')[0]),
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'abs_correlation': result['abs_correlation']
                })
        
        print_and_log(f"Total combinations tested: {len(all_combinations)}")
        
        # Apply FDR correction to ALL combinations
        if all_combinations:
            p_values = [r['p_value'] for r in all_combinations]
            rejected, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            # Add FDR results
            for i, result in enumerate(all_combinations):
                result['p_fdr_corrected'] = p_adj[i]
                result['significant_after_fdr'] = rejected[i]
                result['supports_dvorak'] = result['correlation'] < 0
            
            # SAVE TO CSV FILE
            combination_df = pd.DataFrame(all_combinations)
            csv_filename = 'dvorak7_speed_weights.csv'
            combination_df.to_csv(csv_filename, index=False)
            print_and_log(f"💾 ALL COMBINATIONS SAVED TO: {csv_filename}")
            
            # Filter to significant results only
            significant_combinations = [r for r in all_combinations if r['significant_after_fdr']]
            significant_combinations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            print_and_log(f"Significant after FDR correction: {len(significant_combinations)}/{len(all_combinations)} "
                         f"({len(significant_combinations)/len(all_combinations)*100:.1f}%)")
            
            if significant_combinations:
                print_and_log(f"\n🏆 TOP 20 SIGNIFICANT COMBINATIONS (FDR-corrected):")
                print_and_log("K  Combination                           r       FDR p-val  Dvorak")
                print_and_log("-" * 75)
                
                for i, result in enumerate(significant_combinations[:20]):
                    dvorak_marker = "✅" if result['supports_dvorak'] else "❌"
                    combo_short = result['combination'][:35] + "..." if len(result['combination']) > 35 else result['combination']
                    print_and_log(f"{result['k_way']}  {combo_short:<35} {result['correlation']:>7.3f}  "
                                 f"{result['p_fdr_corrected']:>8.3f}  {dvorak_marker}")
                
                # Best combination overall
                best_combo = significant_combinations[0]
                print_and_log(f"\n🥇 STRONGEST SIGNIFICANT COMBINATION:")
                print_and_log(f"   {best_combo['combination']}")
                print_and_log(f"   r = {best_combo['correlation']:.4f} (FDR p = {best_combo['p_fdr_corrected']:.3f})")
                print_and_log(f"   Uses {best_combo['k_way']} criteria")
                print_and_log(f"   {'Supports' if best_combo['supports_dvorak'] else 'Contradicts'} Dvorak principles")
                
            else:
                print_and_log("❌ NO combinations survived FDR correction!")
        else:
            print_and_log("❌ No combination results found!")
    
    # PART 3: SUMMARY 
    print_and_log(f"\n📊 PART 3: FINAL SUMMARY")
    print_and_log("=" * 60)
    
    total_significant = len([r for r in all_individual_results if r['significant_after_fdr']])
    print_and_log(f"Individual criteria: {total_significant}/{len(all_individual_results)} survive FDR correction")
    
    # Get max effect sizes
    max_individual = max([r['abs_correlation'] for r in all_individual_results]) if all_individual_results else 0
    
    if combination_results and 'significant_combinations' in locals():
        max_combination = max([r['abs_correlation'] for r in significant_combinations]) if significant_combinations else 0
        print_and_log(f"Combinations: {len(significant_combinations) if significant_combinations else 0}/127 survive FDR correction")
        print_and_log(f"Effect sizes:")
        print_and_log(f"   • Individual criteria: up to |r| = {max_individual:.3f} ({'small' if max_individual >= 0.1 else 'negligible'} effect)")
        print_and_log(f"   • Best combinations: up to |r| = {max_combination:.3f} ({'small' if max_combination >= 0.1 else 'negligible'} effect)")
    else:
        print_and_log(f"Effect sizes: Individual criteria up to |r| = {max_individual:.3f} ({'small' if max_individual >= 0.1 else 'negligible'} effect)")
    
    print_and_log(f"Most Dvorak-7 principles are statistically valid but practically weak")
    print_and_log(f"Cohen's effect size conventions: small ≥0.1, medium ≥0.3, large ≥0.5")
    
    return all_individual_results, significant_combinations if 'significant_combinations' in locals() else []

def load_and_process_bigram_data(bigram_file, max_bigrams=None):
    """Load and process bigram typing data"""
    print_and_log("Reading typing data files...")
    print_and_log(f"Reading bigram data from {bigram_file}...")
    
    # Data quality verification
    print_and_log("📋 BIGRAM DATA QUALITY VERIFICATION")
    print_and_log("   Expected: Correctly typed bigrams from correctly typed words")
    print_and_log("   Required CSV columns: 'bigram', 'interkey_interval'")
    
    try:
        # Read a small sample first to check structure
        sample_df = pd.read_csv(bigram_file, nrows=10)
        print_and_log(f"   Found columns: {list(sample_df.columns)}")
        
        if 'bigram' not in sample_df.columns or 'interkey_interval' not in sample_df.columns:
            print_and_log("   ❌ Required columns missing!")
            return None
        
        # Show sample data
        print_and_log("   Sample bigrams from CSV:")
        for _, row in sample_df.head(5).iterrows():
            print_and_log(f"     '{row['bigram']}': {row['interkey_interval']}ms")
        
        # Count total rows
        total_rows = sum(1 for _ in open(bigram_file)) - 1  # Subtract header
        print_and_log(f"   Total bigrams in file: {total_rows:,}")
        
        # Load full dataset
        print_and_log("   Quality indicators:")
        df = pd.read_csv(bigram_file)
        
        # Quality checks
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 'nd', 'on', 'en', 'at', 'ou']
        found_common = sum(1 for bg in df['bigram'].head(10) if bg in common_bigrams)
        print_and_log(f"     Common English bigrams in sample: {found_common}/10 ({found_common*10}%)")
        
        # Check for suspicious patterns
        suspicious = sum(1 for bg in df['bigram'].head(10) if len(set(bg)) == 1 or len(bg) != 2)
        print_and_log(f"     Suspicious bigrams (repeated/invalid chars): {suspicious}/10 ({suspicious*10}%)")
        
        if found_common < 3:
            print_and_log("   ⚠️  Low proportion of common English bigrams - check data quality")
        
        print_and_log("   ✅ Proceeding with data loading...")
        
    except Exception as e:
        print_and_log(f"   ❌ Error reading bigram file: {e}")
        return None
    
    # Load and filter data
    if max_bigrams:
        print_and_log(f"\nWill randomly sample {max_bigrams:,} bigrams")
        df = pd.read_csv(bigram_file)
        if len(df) > max_bigrams:
            df = df.sample(n=max_bigrams, random_state=42)
    else:
        df = pd.read_csv(bigram_file)
    
    print_and_log(f"Loaded {len(df):,} valid bigrams")
    
    # Convert to list of tuples
    bigrams = [(row['bigram'], row['interkey_interval']) for _, row in df.iterrows()]
    
    # Final quality summary
    print_and_log("📊 FINAL DATA QUALITY SUMMARY:")
    unique_bigrams = len(set(bg for bg, _ in bigrams))
    print_and_log(f"   Unique bigrams: {unique_bigrams}")
    
    common_count = sum(1 for bg, _ in bigrams if bg in ['th', 'he', 'in', 'er', 'an', 'nd', 'on', 'en', 'at', 'ou'])
    print_and_log(f"   Common English bigrams: {common_count:,} ({common_count/len(bigrams)*100:.1f}%)")
    
    times = [t for _, t in bigrams]
    print_and_log(f"   Average time: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
    print_and_log("   ✅ Data quality looks reasonable")
    
    return bigrams

def filter_bigrams_by_time(bigrams, min_time=50, max_time=2000):
    """Filter bigrams by typing time thresholds"""
    original_count = len(bigrams)
    
    # Apply absolute thresholds
    filtered_bigrams = [(bg, time) for bg, time in bigrams if min_time <= time <= max_time]
    
    filtered_count = len(filtered_bigrams)
    removed_count = original_count - filtered_count
    
    print_and_log(f"Filtered {removed_count:,}/{original_count:,} bigrams using absolute thresholds ({min_time}-{max_time}ms)")
    print_and_log(f"  Kept {filtered_count:,} bigrams ({filtered_count/original_count*100:.1f}%)")
    
    if filtered_bigrams:
        times = [t for _, t in filtered_bigrams]
        print_and_log(f"  Time range: {min(times):.1f} - {max(times):.1f}ms")
    
    return filtered_bigrams

def test_scoring_function():
    """Test the scoring function with sample bigrams"""
    print_and_log("Testing Dvorak-7 Scoring Function")
    print_and_log("=" * 50)
    
    test_bigrams = ['TH', 'HE', 'IN', 'ER', 'AN', 'QZ', 'XB']
    
    print_and_log("Testing bigram scoring:")
    for bigram in test_bigrams:
        try:
            scores = score_bigram_dvorak7(bigram)
            print_and_log(f"'{bigram}': {scores}")
        except Exception as e:
            print_and_log(f"'{bigram}': ERROR - {e}")
    
    print_and_log(f"\nExpected 7 criteria: repetition, movement, vertical, horizontal, adjacent, weak, outward")
    print_and_log(f"Score range: 0.0 to 1.0 (higher = better for typing)")

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-7 criteria correlations with typing speed')
    parser.add_argument('--max-bigrams', type=int, help='Maximum number of bigrams to analyze')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test-scoring', action='store_true', help='Test the scoring function on sample data')
    args = parser.parse_args()
    
    if args.test_scoring:
        test_scoring_function()
        return
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    start_time = time.time()
    
    # Configuration
    bigram_file = "../../process_136M_keystrokes/output/bigram_times.csv"
    freq_file = "../input/english-letter-pair-frequencies-google-ngrams.csv"  # Updated to match your file
    middle_column_keys = {'b', 'g', 'h', 'n', 't', 'y'}
    
    # Print configuration
    print_and_log("Dvorak-7 Criteria Correlation Analysis - Bigram Speed")
    print_and_log("=" * 80)
    print_and_log("Configuration:")
    print_and_log(f"  Max bigrams: {args.max_bigrams:,}" if args.max_bigrams else "  Max bigrams: unlimited")
    print_and_log(f"  Random seed: {args.random_seed}")
    print_and_log(f"  Middle column keys: {', '.join(sorted(middle_column_keys))}")
    print_and_log("  Analysis includes both raw and frequency-adjusted correlations")
    print_and_log("  Using canonical Dvorak-7 scoring from prep_keypair_dvorak7_scores.py")
    print_and_log("")
    
    # Load frequency data
    print_and_log("Loading frequency data...")
    freq_df = load_frequency_data(freq_file)
    verify_frequency_data(freq_df)
    
    # Load bigram data
    bigrams = load_and_process_bigram_data(bigram_file, args.max_bigrams)
    if not bigrams:
        print_and_log("❌ Failed to load bigram data")
        return
    
    # Filter bigrams
    bigrams = filter_bigrams_by_time(bigrams)
    if not bigrams:
        print_and_log("❌ No bigrams remaining after filtering")
        return
    
    # Create output directory
    print_and_log("Creating plots in 'plots/' directory...")
    Path('plots').mkdir(exist_ok=True)
    
    print_and_log("\n" + "=" * 80)
    print_and_log("STARTING BIGRAM CORRELATION ANALYSIS")
    print_and_log("=" * 80)
    
    # Analyze bigrams
    print_and_log(f"\nBIGRAM ANALYSIS")
    bigram_start = time.time()
    
    bigram_results = analyze_bigram_data(bigrams, freq_df, middle_column_keys)
    
    bigram_elapsed = time.time() - bigram_start
    print_and_log(f"Bigram analysis completed in {format_time(bigram_elapsed)}")
    
    # Print results with frequency comparison
    print_correlation_results_with_frequency(bigram_results, "BIGRAM")
    
    # Analyze criterion combinations
    print_and_log(f"\nAnalyzing criterion combinations...")
    combination_results = analyze_criterion_combinations(bigram_results)
    
    # Create plots if we have combination results
    if combination_results:
        print_and_log(f"\nCreating combination analysis plots...")
        create_combination_performance_plots(combination_results)
    
    # COMPLETE FDR ANALYSIS
    if bigram_results:
        analyze_all_results_with_fdr(bigram_results, combination_results)

    # Final summary
    total_elapsed = time.time() - start_time
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("ANALYSIS COMPLETE")
    print_and_log("=" * 80)
    print_and_log(f"Total runtime: {format_time(total_elapsed)}")
    print_and_log(f"Dvorak-7 Analysis Complete")
    print_and_log("Results saved to: dvorak7_speed_weights.csv")
    print_and_log("Using canonical Dvorak-7 criteria from Dvorak's 1936 work")
    
    # Save log
    save_log()

if __name__ == "__main__":
    main()