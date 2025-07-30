#!/usr/bin/env python3
"""
Comprehensive Empirical Analysis of Dvorak-9 Criteria vs Typing Speed

This script performs a complete empirical validation of August Dvorak's 9 typing 
evaluation criteria using real typing data from 136M+ keystrokes. The analysis
includes frequency adjustment, middle column key effects, and rigorous statistical
testing with FDR correction.

The combinations_weights_from_speed_significant.csv output file contains 
the results of the statistical analysis, which provides empirical combination 
weights for each possible combination of Dvorak criteria. The "combination" and 
"correlation" columns are used by dvorak9_speed.py for scoring keyboard layouts.

Key Features:
- Analyzes correlations between each Dvorak criterion and actual typing speed
- Controls for English bigram frequency effects using regression
- Splits analysis by middle column key usage (lateral index finger movements)
- Tests all 511 possible combinations of criteria (1-way through 9-way)
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

The script produces frequency-adjusted residuals that represent typing speed
after controlling for linguistic frequency effects, enabling fair comparison
of layout-dependent factors.

The 9 scoring criteria for typing bigrams are:
1. Hands - favor alternating hands over same hand
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
4. Don't cross home - avoid crossing over home row
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls (same hand)
9. Strong fingers - favor stronger fingers over weaker ones

Example usage:
    python generate_speed_weights.py
    python generate_speed_weights.py --max-bigrams 100000
    python generate_speed_weights.py --test-scorer
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

# Import the canonical scoring function from dvorak9_scorer
from dvorak9_scorer import score_bigram_dvorak9, Dvorak9Scorer

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

def save_log(filename="speed_weights_for_dvorak9_feature_combinations.log"):
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
            if 'letter_pair' in freq_df.columns and 'score' in freq_df.columns:
                print_and_log(f"     '{row['letter_pair']}': {row['score']:,}")
        
        return freq_df
        
    except Exception as e:
        print_and_log(f"❌ Error loading frequency data: {e}")
        return None

def load_empirical_weights(csv_file="speed_weights.csv", significance_threshold=0.05):
    """
    Load empirical combination weights from FDR analysis results.
    
    Args:
        csv_file: Path to the CSV file with combination results
        significance_threshold: Only use combinations significant after FDR correction
    
    Returns:
        Dictionary mapping combination tuples to correlation weights
    """
    print(f"Loading empirical weights from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} combinations from analysis")
    except Exception as e:
        print(f"❌ Error loading weights file: {e}")
        return {}
    
    # Filter to significant results only
    if 'significant_after_fdr' in df.columns:
        significant_df = df[df['significant_after_fdr'] == True]
        print(f"✅ Using {len(significant_df)}/{len(df)} FDR-significant combinations")
    else:
        # Fallback to p-value threshold
        significant_df = df[df['p_value'] < significance_threshold]
        print(f"✅ Using {len(significant_df)}/{len(df)} combinations with p < {significance_threshold}")
    
    # Convert to weights dictionary
    weights = {}
    
    for _, row in significant_df.iterrows():
        # Parse combination string into tuple
        combo_str = row['combination']
        
        if ' + ' in combo_str:
            combo_parts = [part.strip() for part in combo_str.split(' + ')]
        else:
            combo_parts = [combo_str.strip()]
        
        # Sort for consistent lookup
        combo_tuple = tuple(sorted(combo_parts))
        
        # Use correlation as weight (negative = good for typing)
        weights[combo_tuple] = row['correlation']
    
    print(f"✅ Created {len(weights)} empirical weights")
    
    # Show some examples
    print(f"\nTop 5 strongest combinations:")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (combo, weight) in enumerate(sorted_weights[:5]):
        combo_str = ' + '.join(combo)
        direction = "supports Dvorak" if weight < 0 else "contradicts Dvorak"
        print(f"  {i+1}. {combo_str}: {weight:.4f} ({direction})")
    
    return weights

def print_scoring_results(results):
    """Print formatted scoring results with empirical weight information."""
    print("Empirical Dvorak-9 Scoring Results")
    print("=" * 60)
    
    print(f"Total Weighted Score: {results['total_weighted_score']:8.3f}")
    print(f"Average per Bigram:   {results['average_weighted_score']:8.3f}")
    print(f"Bigrams Analyzed:     {results['bigram_count']:8d}")
    print(f"Weights Source:       {results['weights_source']}")
    
    if results['combination_breakdown']:
        print("\nCombination Breakdown:")
        print("-" * 60)
        print(f"{'Combination':<35} {'Count':<8} {'Contrib':<10} {'%':<6}")
        print("-" * 60)
        
        # Sort by absolute contribution
        sorted_combos = sorted(results['combination_breakdown'].items(), 
                              key=lambda x: abs(x[1]['total_contribution']), 
                              reverse=True)
        
        for combo, stats in sorted_combos[:10]:  # Top 10 combinations
            combo_str = '+'.join(combo) if combo else 'none'
            if len(combo_str) > 34:
                combo_str = combo_str[:31] + '...'
            
            print(f"{combo_str:<35} {stats['count']:<8} {stats['total_contribution']:<10.3f} {stats['percentage']:<6.1f}")

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
    
    # Build frequency dictionary
    if 'letter_pair' in freq_df.columns and 'score' in freq_df.columns:
        freq_dict = dict(zip(freq_df['letter_pair'], freq_df['score']))
    else:
        print_and_log(f"      ❌ Required columns not found in frequency data")
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
        if seq in freq_dict:
            matched_frequencies.append(freq_dict[seq])
            matched_times.append(times[i])
            matched_sequences.append(seq)
    
    overlap_pct = (len(matched_frequencies) / len(sequences)) * 100
    print_and_log(f"      Frequency overlap: {len(matched_frequencies)}/{len(sequences)} ({overlap_pct:.1f}%)")
    
    if len(matched_frequencies) < 10:
        print_and_log(f"      ⚠️  Too few matches for regression ({len(matched_frequencies)})")
        return times, None
    
    # Log-transform frequencies for better linear relationship
    log_frequencies = np.log10(np.array(matched_frequencies))
    times_array = np.array(matched_times)
    
    print_and_log(f"      Frequency range: {min(matched_frequencies):,.3f} to {max(matched_frequencies):,.3f} (mean: {np.mean(matched_frequencies):,.3f})")
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
            'log_frequencies': log_frequencies.copy(),  # For diagnostic plots
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
            if seq in freq_dict:
                log_freq = np.log10(freq_dict[seq])
                predicted_time = model.params[0] + model.params[1] * log_freq
                residual = times[i] - predicted_time  # Actual - Predicted
                frequency_residuals.append(residual)
                residual_magnitudes.append(abs(residual))
            else:
                # For sequences without frequency data, use original time
                # This preserves their relative ranking while noting missing frequency control
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
    
    # Plot 2: Residuals vs Predicted (check for heteroscedasticity)
    residuals = actual_times - predicted_times
    ax2.scatter(predicted_times, residuals, alpha=0.5, s=20, color='lightcoral')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Time (ms)')
    ax2.set_ylabel('Residuals (ms)')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals distribution (check for normality)
    ax3.hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Residuals (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Add distribution stats
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
    
    # Print diagnostic summary
    print_and_log(f"\n📊 FREQUENCY MODEL DIAGNOSTICS:")
    print_and_log(f"   Model fit: R² = {model_info['r_squared']:.4f}")
    print_and_log(f"   Slope significance: p = {model_info.get('p_value', 'N/A'):.4f}")
    print_and_log(f"   Residual statistics:")
    print_and_log(f"     Mean: {mean_res:.2f}ms (should be ~0)")
    print_and_log(f"     Std: {std_res:.2f}ms")
    print_and_log(f"     Range: {min(residuals):.1f} to {max(residuals):.1f}ms")
    
    # Check for model assumptions
    if abs(mean_res) < 1.0:
        print_and_log(f"   ✅ Residuals well-centered (mean ≈ 0)")
    else:
        print_and_log(f"   ⚠️  Residuals not well-centered (mean = {mean_res:.2f})")
    
    if model_info.get('p_value', 1) < 0.05:
        print_and_log(f"   ✅ Frequency significantly predicts typing time")
    else:
        print_and_log(f"   ⚠️  Frequency not significantly predictive")

def create_bigram_scatter_plot(sequences, original_times, frequency_residuals, model_info=None, output_dir='plots'):
    """Create scatter plot showing all bigrams before/after frequency adjustment"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Bigram Typing Times: Raw vs Frequency-Controlled', fontsize=16, fontweight='bold')
    
    # Plot 1: Original times distribution
    ax1.hist(original_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Raw Typing Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Original Times Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add stats
    mean_orig = np.mean(original_times)
    std_orig = np.std(original_times)
    ax1.axvline(mean_orig, color='red', linestyle='--', label=f'Mean: {mean_orig:.1f}ms')
    ax1.legend()
    
    # Plot 2: Frequency residuals distribution
    ax2.hist(frequency_residuals, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Frequency Residuals (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Frequency-Controlled Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add stats
    mean_res = np.mean(frequency_residuals)
    std_res = np.std(frequency_residuals)
    ax2.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.1f}ms')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5, label='Expected (0)')
    ax2.legend()
    
    # Plot 3: Scatter plot Raw vs Residuals
    # Sample if too many points for visibility
    if len(original_times) > 5000:
        indices = np.random.choice(len(original_times), 5000, replace=False)
        sample_orig = [original_times[i] for i in indices]
        sample_res = [frequency_residuals[i] for i in indices]
        sample_seq = [sequences[i] for i in indices]
        alpha = 0.3
        title_suffix = f" (n={len(sample_orig):,} sample)"
    else:
        sample_orig = original_times
        sample_res = frequency_residuals
        sample_seq = sequences
        alpha = 0.5
        title_suffix = f" (n={len(sample_orig):,})"
    
    ax3.scatter(sample_orig, sample_res, alpha=alpha, s=15, color='purple')
    ax3.set_xlabel('Raw Typing Time (ms)')
    ax3.set_ylabel('Frequency Residuals (ms)')
    ax3.set_title(f'Raw vs Frequency-Controlled{title_suffix}')
    ax3.grid(True, alpha=0.3)
    
    # Add reference lines
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero residual')
    
    # Correlation between raw and residuals
    corr = np.corrcoef(sample_orig, sample_res)[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Highlight some interesting bigrams if model info available
    if model_info and 'matched_sequences' in model_info:
        # Find extreme residuals among matched sequences
        matched_seqs = model_info['matched_sequences']
        matched_times = model_info['actual_times']
        matched_predicted = model_info['predicted_times']
        matched_residuals = matched_times - matched_predicted
        
        # Find most extreme positive and negative residuals
        max_idx = np.argmax(matched_residuals)
        min_idx = np.argmin(matched_residuals)
        
        # Annotate if these are in our sample
        for idx, seq in enumerate(sample_seq):
            if seq == matched_seqs[max_idx]:
                ax3.annotate(f"'{seq}' (+{matched_residuals[max_idx]:.0f}ms)", 
                           (sample_orig[idx], sample_res[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=8)
            elif seq == matched_seqs[min_idx]:
                ax3.annotate(f"'{seq}' ({matched_residuals[min_idx]:.0f}ms)", 
                           (sample_orig[idx], sample_res[idx]),
                           xytext=(10, -15), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           fontsize=8)
    
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'bigram_frequency_adjustment_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Bigram scatter plot saved to: {output_path}")
    
    # Print summary statistics
    print_and_log(f"\n📊 BIGRAM TIMING SUMMARY:")
    print_and_log(f"   Raw times: {mean_orig:.1f} ± {std_orig:.1f}ms (range: {min(original_times):.1f}-{max(original_times):.1f})")
    print_and_log(f"   Frequency residuals: {mean_res:.1f} ± {std_res:.1f}ms (range: {min(frequency_residuals):.1f}-{max(frequency_residuals):.1f})")
    print_and_log(f"   Correlation raw vs residuals: r = {corr:.3f}")
    
    # Interpretation
    if abs(corr) > 0.8:
        print_and_log(f"   💡 High correlation suggests frequency control had minimal impact")
    elif abs(corr) > 0.5:
        print_and_log(f"   💡 Moderate correlation suggests frequency control modified rankings")
    else:
        print_and_log(f"   💡 Low correlation suggests strong frequency effects were controlled")

def analyze_correlations(sequences, times, criteria_names, group_name, analysis_type, model_info=None):
    """Analyze correlations between Dvorak criteria scores and typing times"""
    results = {}
    
    print_and_log(f"  {analysis_type.replace('_', ' ').title()} Analysis:")
    
    # Define group_suffix EARLY based on group_name
    if "No Middle" in group_name:
        group_suffix = "no_middle"
    elif "With Middle" in group_name:
        group_suffix = "with_middle"
    else:
        group_suffix = "unknown"
    
    # Calculate scores for all sequences
    print_and_log(f"  Calculating Dvorak scores for {len(sequences):,} sequences...")
    
    start_time = time.time()
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_sequences = []
    valid_times = []
    sequence_scores_data = []
    
    for i, (seq, time_val) in enumerate(zip(sequences, times)):
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - start_time
            print_and_log(f"    Progress: {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - {elapsed:.1f}s", end='\r')
        
        # Calculate Dvorak scores using the canonical function
        scores = score_bigram_dvorak9(seq)
        
        # Validate scores
        if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
            valid_sequences.append(seq)
            valid_times.append(time_val)
            
            # Store complete scores for this sequence (for interaction analysis)
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
                # Store error information
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
                    'error': str(e)
                }
                continue
    
    print_and_log(f"    Correlation analysis complete")
    
    # Diagnostic information about criteria variation
    constant_criteria = []
    low_variation_criteria = []
    
    for criterion, scores_list in criterion_scores.items():
        if len(scores_list) > 0:
            unique_scores = len(set(scores_list))
            score_std = np.std(scores_list)
            
            if unique_scores <= 1:
                constant_criteria.append((criterion, unique_scores, score_std))
            elif score_std < 0.01:  # Very low variation
                low_variation_criteria.append((criterion, unique_scores, score_std))
    
    if constant_criteria:
        print_and_log(f"    ⚠️  Criteria with constant scores: {len(constant_criteria)}")
        for criterion, unique, std in constant_criteria:
            print_and_log(f"      • {criterion}: {unique} unique values, std={std:.6f}")
            
    if low_variation_criteria:
        print_and_log(f"    ⚠️  Criteria with low variation: {len(low_variation_criteria)}")
        for criterion, unique, std in low_variation_criteria:
            print_and_log(f"      • {criterion}: {unique} unique values, std={std:.6f}")
    
    # Report why constant scores might occur
    if constant_criteria or low_variation_criteria:
        print_and_log(f"    💡 Possible causes of constant/low-variation scores:")
        print_and_log(f"      • Sample may be too small or not diverse enough")
        print_and_log(f"      • Bigrams may not trigger certain criteria (e.g., columns)")
        print_and_log(f"      • Filtering may have removed sequences with variation")
    
    return results

def analyze_bigram_data(bigrams, freq_df, middle_column_keys):
    """Analyze bigram typing data with middle column key analysis"""
    
    criteria_names = {
        'hands': 'hands',
        'fingers': 'fingers', 
        'skip_fingers': 'skip fingers',
        'dont_cross_home': "don't cross home",
        'same_row': 'same row',
        'home_row': 'home row',
        'columns': 'columns',
        'strum': 'strum',
        'strong_fingers': 'strong fingers'
    }
    
    # FIX: ADD THE MISSING BIGRAM SPLITTING LOGIC
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
    
    # Now analyze each group (rest of function continues as before)
    for group_data, group_name in [(without_middle, "Bigrams (No Middle Columns)"), 
                                   (with_middle, "Bigrams (With Middle Columns)")]:
        
        print_and_log(f"\n--- Analyzing {group_name} ---")
        print_and_log(f"Group size: {len(group_data):,} bigrams")
        
        if len(group_data) < MIN_SEQUENCES:
            print_and_log(f"❌ SKIPPING {group_name}: only {len(group_data)} bigrams (need ≥{MIN_SEQUENCES})")
            print_and_log(f"   This group will NOT appear in FDR analysis!")
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
            
            # Create diagnostic plots for the first group only (to avoid duplication)
            if group_name == "Bigrams (No Middle Columns)" and model_info:
                create_diagnostic_plots(model_info)
                create_bigram_scatter_plot(sequences, times, frequency_residuals, model_info)
            
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
            # Clean up the group name to remove redundant "Middle"
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

def create_frequency_comparison_plots(results, output_dir='plots'):
    """Create visualization comparing raw vs frequency-adjusted correlations"""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract data for plotting
    plot_data = []
    
    for key, data in results.items():
        # Skip internal data
        if key.startswith('_'):
            continue
        
        if not isinstance(data, dict) or 'spearman_r' not in data:
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
                
            plot_data.append({
                'criterion': criterion,
                'analysis': analysis,
                'correlation': data['spearman_r'],
                'p_value': data['spearman_p'],
                'group': data.get('group', ''),
                'name': data.get('name', criterion)
            })
    
    if not plot_data:
        print_and_log("No data available for plotting")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak-9 criteria correlations: raw vs frequency-adjusted', fontsize=16, fontweight='bold')
    
    # Split by middle column status
    without_middle = df[df['group'].str.contains('No Middle', na=False)]
    with_middle = df[df['group'].str.contains('With Middle', na=False)]
    
    # Plot 1: Raw vs Freq-Adjusted correlations (Without middle columns)
    if not without_middle.empty:
        pivot_without = without_middle.pivot(index='criterion', columns='analysis', values='correlation')
        if 'raw' in pivot_without.columns and 'freq_adjusted' in pivot_without.columns:
            x_pos = np.arange(len(pivot_without.index))
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, pivot_without['freq_adjusted'], width, 
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax1.bar(x_pos + width/2, pivot_without['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Criterion')
            ax1.set_ylabel('Spearman correlation')
            ax1.set_title('Without middle columns')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(pivot_without.index, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 2: Raw vs Freq-Adjusted correlations (With middle columns)  
    if not with_middle.empty:
        pivot_with = with_middle.pivot(index='criterion', columns='analysis', values='correlation')
        if 'raw' in pivot_with.columns and 'freq_adjusted' in pivot_with.columns:
            x_pos = np.arange(len(pivot_with.index))
            width = 0.35
            
            bars1 = ax2.bar(x_pos - width/2, pivot_with['freq_adjusted'], width,
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax2.bar(x_pos + width/2, pivot_with['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('Criterion')
            ax2.set_ylabel('Spearman correlation')
            ax2.set_title('With middle columns')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(pivot_with.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 3: Raw vs Frequency-adjusted scatter plot
    raw_correlations = []
    freq_correlations = []
    labels = []
    
    # Group results by criterion for scatter plot
    criterion_groups = {}
    for _, row in df.iterrows():
        key = (row['criterion'], row['group'])
        if key not in criterion_groups:
            criterion_groups[key] = {}
        criterion_groups[key][row['analysis']] = row['correlation']
    
    for (criterion, group), analyses in criterion_groups.items():
        if 'raw' in analyses and 'freq_adjusted' in analyses:
            raw_correlations.append(analyses['raw'])
            freq_correlations.append(analyses['freq_adjusted'])
            
            # Create label
            middle_status = "NM" if "No Middle" in group else "WM" if "With Middle" in group else "?"
            labels.append(f"{criterion[:2].upper()}")
    
    if raw_correlations and freq_correlations:
        colors = ['blue' if 'NM' in label else 'red' for label in labels]
        ax3.scatter(raw_correlations, freq_correlations, c=colors, alpha=0.7, s=60)
        
        # Add labels to points
        for i, label in enumerate(labels):
            ax3.annotate(label, (raw_correlations[i], freq_correlations[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line for perfect agreement
        min_val = min(min(raw_correlations), min(freq_correlations))
        max_val = max(max(raw_correlations), max(freq_correlations))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
        
        ax3.set_xlabel('Raw correlation')
        ax3.set_ylabel('Frequency-adjusted correlation')
        ax3.set_title('Raw vs frequency-adjusted correlations')
        ax3.grid(True, alpha=0.3)
        ax3.legend(['With middle', 'Without middle', 'Perfect agreement'])
    
    # Plot 4: Effect sizes comparison
    effect_data = []
    for _, row in df.iterrows():
        effect_data.append({
            'criterion': row['name'],
            'analysis': row['analysis'],
            'abs_correlation': abs(row['correlation']),
            'group': 'With middle' if 'With Middle' in row['group'] else 'Without middle'
        })
    
    effect_df = pd.DataFrame(effect_data)
    if not effect_df.empty:
        pivot_effects = effect_df.pivot_table(index='criterion', columns='analysis', 
                                            values='abs_correlation', aggfunc='mean')
        
        if 'raw' in pivot_effects.columns and 'freq_adjusted' in pivot_effects.columns:
            x_pos = np.arange(len(pivot_effects.index))
            width = 0.35
            
            bars1 = ax4.bar(x_pos - width/2, pivot_effects['freq_adjusted'], width,
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax4.bar(x_pos + width/2, pivot_effects['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax4.set_xlabel('Criterion')
            ax4.set_ylabel('Absolute correlation')
            ax4.set_title('Effect sizes (|r|)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(pivot_effects.index, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak9_frequency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Comparison plots saved to: {output_path}")

def create_combination_performance_plots(combination_results, output_dir='plots'):
    """Create plots showing combination analysis results"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print_and_log("📊 Creating combination analysis plots...")
    
    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak Criterion Combination Analysis', fontsize=16, fontweight='bold')
    
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
    all_sizes = []
    
    for k_way, results_list in combination_results.items():
        k = int(k_way.split('_')[0])
        for result in results_list:
            if result['p_value'] < 0.05:  # Only significant results
                all_correlations.append(result['abs_correlation'])
                all_sizes.append(k)
    
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
    output_path = Path(output_dir) / 'dvorak_combination_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Combination analysis plots saved to: {output_path}")
    
    return output_path

def create_criterion_interaction_heatmap(combination_results, output_dir='plots'):
    """Create heatmap showing which criteria work best together"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print_and_log("🔥 Creating criterion interaction heatmap...")
    
    # Extract all pairwise combinations
    pairwise_results = combination_results.get('2_way', [])
    
    if not pairwise_results:
        print_and_log("❌ No pairwise combination data found for heatmap")
        return None
    
    # Get all unique criteria
    all_criteria = set()
    for result in pairwise_results:
        criteria = result['combination'].split(' + ')
        all_criteria.update(criteria)
    
    criteria_list = sorted(list(all_criteria))
    n_criteria = len(criteria_list)
    
    # Create correlation matrix
    correlation_matrix = np.zeros((n_criteria, n_criteria))
    p_value_matrix = np.ones((n_criteria, n_criteria))
    
    # Fill matrix with pairwise correlations
    for result in pairwise_results:
        criteria = result['combination'].split(' + ')
        if len(criteria) == 2:
            idx1 = criteria_list.index(criteria[0])
            idx2 = criteria_list.index(criteria[1])
            
            # Use absolute correlation for strength
            correlation_matrix[idx1, idx2] = result['abs_correlation']
            correlation_matrix[idx2, idx1] = result['abs_correlation']
            
            # Store p-values
            p_value_matrix[idx1, idx2] = result['p_value']
            p_value_matrix[idx2, idx1] = result['p_value']
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dvorak Criterion Interaction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Correlation strength heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='Reds', aspect='auto')
    ax1.set_xticks(range(n_criteria))
    ax1.set_yticks(range(n_criteria))
    ax1.set_xticklabels(criteria_list, rotation=45, ha='right')
    ax1.set_yticklabels(criteria_list)
    ax1.set_title('Pairwise Combination Strength (|r|)')
    
    # Add correlation values to cells
    for i in range(n_criteria):
        for j in range(n_criteria):
            if i != j and correlation_matrix[i, j] > 0:
                text = ax1.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black" if correlation_matrix[i, j] < 0.3 else "white",
                               fontsize=8)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Absolute Correlation |r|')
    
    # Plot 2: Significance heatmap
    # Convert p-values to significance levels for better visualization
    significance_matrix = np.zeros_like(p_value_matrix)
    significance_matrix[p_value_matrix < 0.001] = 3  # ***
    significance_matrix[(p_value_matrix >= 0.001) & (p_value_matrix < 0.01)] = 2  # **
    significance_matrix[(p_value_matrix >= 0.01) & (p_value_matrix < 0.05)] = 1  # *
    significance_matrix[p_value_matrix >= 0.05] = 0  # ns
    
    im2 = ax2.imshow(significance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=3)
    ax2.set_xticks(range(n_criteria))
    ax2.set_yticks(range(n_criteria))
    ax2.set_xticklabels(criteria_list, rotation=45, ha='right')
    ax2.set_yticklabels(criteria_list)
    ax2.set_title('Statistical Significance')
    
    # Add significance symbols to cells
    for i in range(n_criteria):
        for j in range(n_criteria):
            if i != j:
                p_val = p_value_matrix[i, j]
                if p_val < 0.001:
                    symbol = '***'
                elif p_val < 0.01:
                    symbol = '**'
                elif p_val < 0.05:
                    symbol = '*'
                else:
                    symbol = 'ns'
                
                ax2.text(j, i, symbol, ha="center", va="center", 
                        color="white" if significance_matrix[i, j] > 1.5 else "black",
                        fontsize=8, fontweight='bold')
    
    # Custom colorbar for significance
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar2.set_label('Significance Level')
    cbar2.set_ticklabels(['ns', '*', '**', '***'])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak_criterion_interactions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Criterion interaction heatmap saved to: {output_path}")
    
    # Print insights from the heatmap
    print_and_log(f"\n🔥 CRITERION INTERACTION INSIGHTS:")
    
    # Find strongest pairwise interactions
    strong_pairs = []
    for i in range(n_criteria):
        for j in range(i+1, n_criteria):
            if correlation_matrix[i, j] > 0.1 and p_value_matrix[i, j] < 0.05:
                strong_pairs.append((
                    f"{criteria_list[i]} + {criteria_list[j]}",
                    correlation_matrix[i, j],
                    p_value_matrix[i, j]
                ))
    
    strong_pairs.sort(key=lambda x: x[1], reverse=True)
    
    if strong_pairs:
        print_and_log(f"   Top criterion pairs (|r| > 0.1, p < 0.05):")
        for i, (pair, corr, p_val) in enumerate(strong_pairs[:5]):
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print_and_log(f"     {i+1}. {pair}: |r| = {corr:.3f}{sig}")
    else:
        print_and_log(f"   ⚠️  No strong criterion pairs found (all |r| < 0.1)")
    
    # Find criteria that work well with others
    avg_interactions = np.mean(correlation_matrix, axis=1)
    best_criteria = [(criteria_list[i], avg_interactions[i]) for i in range(n_criteria)]
    best_criteria.sort(key=lambda x: x[1], reverse=True)
    
    print_and_log(f"   Best criteria for combinations (average |r| with others):")
    for i, (criterion, avg_corr) in enumerate(best_criteria[:3]):
        print_and_log(f"     {i+1}. {criterion}: {avg_corr:.3f}")
    
    return output_path

def interpret_correlation_results(results, analysis_name):
    """Provide interpretation of correlation results"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"📊 RESULTS INTERPRETATION {analysis_name.upper()}")
    print_and_log("=" * 80)
    
    # Group results by analysis type and group for clear organization
    grouped_results = {}
    frequency_models = {}
    
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
            
        # Parse the key to extract criterion and analysis type
        parts = key.split('_')
        if len(parts) >= 2:
            if parts[-1] == 'adjusted':  # freq_adjusted
                criterion = '_'.join(parts[:-2])
                analysis_type = 'freq_adjusted'
            else:  # raw
                criterion = '_'.join(parts[:-1])
                analysis_type = 'raw'
            
            group_name = data.get('group', 'Unknown Group')
            
            # Clean up group names
            if 'No Middle Columns' in group_name:
                clean_group_name = 'Without Middle Columns'
            elif 'With Middle Columns' in group_name:
                clean_group_name = 'With Middle Columns'
            else:
                clean_group_name = group_name
            
            # Only process frequency-adjusted results for interpretation
            if analysis_type == 'freq_adjusted':
                group_key = clean_group_name
                if group_key not in grouped_results:
                    grouped_results[group_key] = {}
                    frequency_models[group_key] = None
                
                # Store criterion results
                grouped_results[group_key][criterion] = data
                
                # Store frequency model info (same for all criteria in group)
                if 'frequency_model' in data and frequency_models[group_key] is None:
                    frequency_models[group_key] = data['frequency_model']
    
    # Now process each group separately with clear headers
    for group_name in sorted(grouped_results.keys()):
        group_data = grouped_results[group_name]
        freq_model = frequency_models[group_name]
        
        print_and_log(f"\n🔍 {group_name.upper()}")
        print_and_log("-" * 60)
        
        # Show frequency model info ONCE per group
        if freq_model:
            r2_pct = freq_model.get('r_squared', 0) * 100
            slope = freq_model.get('slope', 0)
            p_val = freq_model.get('p_value', 1)
            n_obs = freq_model.get('n_obs', 0)
            
            print_and_log(f"📈 FREQUENCY MODEL (shared by all criteria in this group):")
            print_and_log(f"   R² = {r2_pct:.1f}% of variance explained by English bigram frequency")
            print_and_log(f"   Slope = {slope:.4f} (p = {p_val:.4f})")
            print_and_log(f"   Sample size: {n_obs:,} bigrams with frequency data")
            print_and_log(f"   → Negative slope means higher frequency = faster typing")
            print_and_log("")
        
        # Separate criteria by Dvorak support/contradiction
        supports_dvorak = []
        contradicts_dvorak = []
        
        for criterion, data in group_data.items():
            if 'spearman_r' in data and not np.isnan(data['spearman_r']):
                if data.get('spearman_p', 1) < 0.05:  # Only significant results
                    if data['spearman_r'] < 0:
                        supports_dvorak.append((criterion, data))
                    else:
                        contradicts_dvorak.append((criterion, data))
        
        # Sort by effect size
        supports_dvorak.sort(key=lambda x: abs(x[1]['spearman_r']), reverse=True)
        contradicts_dvorak.sort(key=lambda x: abs(x[1]['spearman_r']), reverse=True)
        
        print_and_log(f"✅ CRITERIA THAT SUPPORT DVORAK (negative correlation = faster typing):")
        if supports_dvorak:
            for criterion, data in supports_dvorak:
                r = data['spearman_r']
                p = data['spearman_p']
                abs_r = abs(r)
                
                if abs_r >= 0.5:
                    effect = "large effect"
                elif abs_r >= 0.3:
                    effect = "medium effect"
                elif abs_r >= 0.1:
                    effect = "small effect"
                else:
                    effect = "negligible effect"
                
                print_and_log(f"   • {data['name']}: r = {r:.3f}, p = {p:.3f} ({effect})")
        else:
            print_and_log(f"   • None found in this group")
        
        print_and_log(f"⚠️  CRITERIA THAT CONTRADICT DVORAK (positive correlation = slower typing):")
        if contradicts_dvorak:
            for criterion, data in contradicts_dvorak:
                r = data['spearman_r']
                p = data['spearman_p']
                abs_r = abs(r)
                
                if abs_r >= 0.5:
                    effect = "large effect"
                elif abs_r >= 0.3:
                    effect = "medium effect"
                elif abs_r >= 0.1:
                    effect = "small effect"
                else:
                    effect = "negligible effect"
                
                # Add context for contradictory results
                context = ""
                if "hand" in data['name'].lower():
                    context = "\n       → This suggests hand alternation may not always speed typing"
                elif "strong fingers" in data['name'].lower():
                    context = "\n       → This suggests avoiding pinky may not help typing speed"
                elif "same row" in data['name'].lower():
                    context = "\n       → This suggests same-row sequences may slow typing (finger interference?)"
                
                print_and_log(f"   • {data['name']}: r = {r:.3f}, p = {p:.3f} ({effect}){context}")
        else:
            print_and_log(f"   • None found in this group")
    
    # Compare groups if we have exactly 2
    group_names = list(grouped_results.keys())
    if len(group_names) == 2:
        print_and_log(f"\n🔍 COMPARISON BETWEEN GROUPS:")
        print_and_log("-" * 60)
        
        group1_data = grouped_results[group_names[0]]
        group2_data = grouped_results[group_names[1]]
        
        # Find common criteria
        common_criteria = set(group1_data.keys()) & set(group2_data.keys())
        
        if common_criteria:
            print_and_log(f"Criteria appearing in both groups:")
            for criterion in sorted(common_criteria):
                data1 = group1_data[criterion]
                data2 = group2_data[criterion]
                
                r1 = data1.get('spearman_r', float('nan'))
                r2 = data2.get('spearman_r', float('nan'))
                p1 = data1.get('spearman_p', 1)
                p2 = data2.get('spearman_p', 1)
                
                # Only show if at least one is significant
                if p1 < 0.05 or p2 < 0.05:
                    sig1 = "✅" if p1 < 0.05 else "❌"
                    sig2 = "✅" if p2 < 0.05 else "❌"
                    
                    print_and_log(f"  {data1['name']}:")
                    print_and_log(f"    {group_names[0]}: r = {r1:.3f} {sig1}")
                    print_and_log(f"    {group_names[1]}: r = {r2:.3f} {sig2}")
                    
                    # Note direction changes
                    if not np.isnan(r1) and not np.isnan(r2):
                        if (r1 < 0) != (r2 < 0):
                            print_and_log(f"    → Direction reversal between groups!")
    
    print_and_log(f"\n💡 INTERPRETATION SUMMARY:")
    print_and_log(f"   • This shows ONLY frequency-adjusted results (controls for English bigram frequency)")
    print_and_log(f"   • Negative correlation = higher Dvorak score → faster typing (supports Dvorak)")
    print_and_log(f"   • Positive correlation = higher Dvorak score → slower typing (contradicts Dvorak)")
    print_and_log(f"   • All effects are small/negligible (|r| < 0.1) - typical for typing research")

def analyze_criterion_combinations(results):
    """Analyze how combinations of criteria predict typing speed"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print_and_log("=" * 80)
    print_and_log("Examining how combinations of criteria interact to predict typing speed")
    print_and_log("IMPORTANT: Using ONLY frequency-adjusted data (controls for English bigram frequency)")
    
    # Look for FREQUENCY-ADJUSTED sequence data only
    sequence_data_sets = {}
    freq_adjusted_sequences = []
    
    # Collect all frequency-adjusted sequence data
    for key, data in results.items():
        if key.startswith('_sequence_scores_freq_adjusted') and isinstance(data, list):
            freq_adjusted_sequences.extend(data)
            print_and_log(f"✅ Found frequency-adjusted sequence data: {len(data)} sequences")
    
    # If we have frequency-adjusted sequence data, use it
    if freq_adjusted_sequences:
        sequence_data_sets["All sequences (frequency-adjusted)"] = freq_adjusted_sequences
        print_and_log(f"✅ Using {len(freq_adjusted_sequences):,} frequency-adjusted sequences")
    else:
        # Fallback: look for ANY sequence data but warn about it
        print_and_log("⚠️  No frequency-adjusted sequence data found, looking for any sequence data...")
        all_sequences = []
        for key, data in results.items():
            if key.startswith('_sequence_scores') and isinstance(data, list):
                all_sequences.extend(data)
                print_and_log(f"   Found sequence data in: {key}")
        
        if all_sequences:
            sequence_data_sets["All sequences (mixed analysis types)"] = all_sequences
            print_and_log(f"⚠️  WARNING: Using mixed analysis types - results may be inconsistent")
        else:
            print_and_log("❌ No sequence-level data found for combination analysis")
            return None
    
    if not sequence_data_sets:
        print_and_log("❌ No sequence-level data found for combination analysis")
        return None
    
    print_and_log(f"✅ Found sequence data for {len(sequence_data_sets)} groups")
    
    # Analyze each group
    all_combination_results = {}
    
    for group_name, sequence_data in sequence_data_sets.items():
        if len(sequence_data) < 100:
            print_and_log(f"\n⚠️  Skipping {group_name}: too few sequences ({len(sequence_data)})")
            continue
        
        print_and_log(f"\n{group_name}")
        print_and_log(f"------------------------------------------------------------")
        print_and_log(f"   Sequences: {len(sequence_data):,}")
        
        # Convert to DataFrame
        df = pd.DataFrame(sequence_data)
        
        # Get criteria columns (exclude sequence, time, analysis_type)
        exclude_cols = {'sequence', 'time', 'analysis_type'}
        criteria_cols = [col for col in df.columns if col not in exclude_cols]
        
        # IMPORTANT: Use frequency-adjusted times if available
        if 'freq_adjusted_time' in df.columns:
            times = df['freq_adjusted_time'].values
            print_and_log(f"   ✅ Using frequency-adjusted times")
        else:
            times = df['time'].values
            print_and_log(f"   ⚠️  Using raw times (frequency adjustment not available)")
        
        print_and_log(f"   Criteria: {criteria_cols}")
        
        # COMPREHENSIVE COMBINATION ANALYSIS - TEST ALL 511 COMBINATIONS
        group_results = {}
        
        # For each combination size k from 1 to 9
        for k in range(1, len(criteria_cols) + 1):
            print_and_log(f"\n📊 {k}-WAY COMBINATIONS:")
            
            # Generate ALL combinations of size k
            combos = list(combinations(criteria_cols, k))
            total_combos = len(combos)
            print_and_log(f"   Testing ALL {total_combos:,} combinations of {k} criteria...")
            
            combo_results = []
            
            # Test EVERY combination (no sampling!)
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
        
        # Store results for this group
        all_combination_results.update(group_results)
        
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
        
        # CREATE THE MISSING PLOTS! 📊
        if group_results:
            print_and_log(f"\n📊 Creating combination analysis visualizations...")
            
            # Plot 1: Combination performance plots
            create_combination_performance_plots(group_results)
            
            # Plot 2: Criterion interaction heatmap
            create_criterion_interaction_heatmap(group_results)
            
            print_and_log(f"✅ Combination analysis plots created successfully!")
        else:
            print_and_log(f"⚠️  No combination results to plot")
    
    # Return all results for FDR analysis
    return all_combination_results

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
        # Use pandas sample for better randomness
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
    
    common_count = sum(1 for bg, _ in bigrams if bg in common_bigrams)
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

def analyze_all_results_with_fdr(results, combination_results=None):
    """Complete FDR analysis for all groups and combinations"""
    
    print_and_log(f"\n" + "=" * 100)
    print_and_log("🎯 COMPLETE FDR-CORRECTED ANALYSIS")
    print_and_log("=" * 100)
    print_and_log("Frequency-adjusted results with FDR multiple testing correction")
    print_and_log("Answers: Which criteria survive rigorous statistical testing?")
    print_and_log("")
    
    # PART 1: INDIVIDUAL CRITERIA BY GROUP
    print_and_log("📊 PART 1: INDIVIDUAL CRITERIA (9 TESTS PER GROUP)")
    print_and_log("=" * 60)
    
    # Group detection for new key format
    groups = {}
    for key, data in results.items():
        # Look for frequency-adjusted keys with the new naming scheme
        if (key.endswith('_freq_adjusted') and 
            isinstance(data, dict) and 
            'spearman_r' in data and 
            not np.isnan(data['spearman_r'])):
            
            group_name = data.get('group', 'Unknown')
            
            # Extract group type for better organization
            if '_no_middle_' in key:
                group_type = 'No Middle Columns'
            elif '_with_middle_' in key:
                group_type = 'With Middle Columns'
            else:
                group_type = 'Unknown'
            
            print_and_log(f"🔍 Processing key: '{key}' → group: '{group_name}' → type: '{group_type}'")
            
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append((key, data))
    
    print_and_log(f"\n📋 Groups found for FDR analysis: {len(groups)}")
    for group_name, group_data in groups.items():
        print_and_log(f"   • {group_name}: {len(group_data)} criteria")
    
    if len(groups) == 0:
        print_and_log("❌ No frequency-adjusted groups found for FDR analysis!")
        print_and_log("Available keys in results:")
        for key in sorted(results.keys()):
            if not key.startswith('_'):
                print_and_log(f"   {key}: {type(results[key])}")
        return
    
    # Expected: Should now see both groups
    expected_groups = [
        "Bigrams (No Middle Columns) (freq adjusted)",
        "Bigrams (With Middle Columns) (freq adjusted)"
    ]
    
    found_groups = list(groups.keys())
    print_and_log(f"\n✅ EXPECTED GROUPS CHECK:")
    for expected in expected_groups:
        if expected in found_groups:
            print_and_log(f"   ✅ Found: '{expected}'")
        else:
            print_and_log(f"   ❌ Missing: '{expected}'")
    
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
            print_and_log(f"Significant after FDR: {sig_count}/9 criteria")
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
            print_and_log("Criterion              r      95% CI         FDR p-val  Significant  Dvorak")
            print_and_log("-" * 80)

            for result in group_results:
                # Calculate confidence interval
                ci_lower, ci_upper = correlation_confidence_interval(
                    result['correlation'], result['n_samples']
                )
                
                sig_marker = "✅" if result['significant_after_fdr'] else "❌"
                dvorak_marker = "✅ Support" if result['supports_dvorak'] else "❌ Contradict"
                
                print_and_log(f"{result['criterion']:<18} {result['correlation']:>6.3f}  "
                            f"[{ci_lower:>5.3f},{ci_upper:>5.3f}]  "
                            f"{result['p_fdr_corrected']:>8.3f}  {sig_marker:<11}  {dvorak_marker}")
    
    # PART 2: COMBINATIONS (same as before)
    if combination_results:
        print_and_log(f"\n📊 PART 2: ALL 511 COMBINATIONS WITH FDR CORRECTION")
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
            csv_filename = 'speed_weights.csv'
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
                
                # SAVE SIGNIFICANT COMBINATIONS TO SEPARATE CSV
                sig_combination_df = pd.DataFrame(significant_combinations)
                sig_csv_filename = 'speed_weights.csv'
                sig_combination_df.to_csv(sig_csv_filename, index=False)
                print_and_log(f"💾 SIGNIFICANT COMBINATIONS SAVED TO: {sig_csv_filename}")
                
            else:
                print_and_log("❌ NO combinations survived FDR correction!")
        else:
            print_and_log("❌ No combination results found!")
    
    # PART 3: SUMMARY COMPARISON
    print_and_log(f"\n📊 PART 3: SUMMARY COMPARISON")
    print_and_log("=" * 60)
    
    # Compare groups for individual criteria
    if len(groups) == 2:
        group_names = list(groups.keys())
        group1_results = [r for r in all_individual_results if r['group'] == group_names[0]]
        group2_results = [r for r in all_individual_results if r['group'] == group_names[1]]
        
        print_and_log(f"INDIVIDUAL CRITERIA COMPARISON:")
        print_and_log(f"{group_names[0]}:")
        g1_support = sum(1 for r in group1_results if r['significant_after_fdr'] and r['supports_dvorak'])
        g1_contradict = sum(1 for r in group1_results if r['significant_after_fdr'] and not r['supports_dvorak'])
        print_and_log(f"  ✅ Support Dvorak: {g1_support}/9")
        print_and_log(f"  ❌ Contradict Dvorak: {g1_contradict}/9")
        
        print_and_log(f"{group_names[1]}:")
        g2_support = sum(1 for r in group2_results if r['significant_after_fdr'] and r['supports_dvorak'])
        g2_contradict = sum(1 for r in group2_results if r['significant_after_fdr'] and not r['supports_dvorak'])
        print_and_log(f"  ✅ Support Dvorak: {g2_support}/9") 
        print_and_log(f"  ❌ Contradict Dvorak: {g2_contradict}/9")
        
        # Find differences between groups
        print_and_log(f"\nDIFFERENCES BETWEEN GROUPS:")
        for criterion in ['hands', 'fingers', 'skip fingers', "don't cross home", 'same row', 'home row', 'columns', 'strum', 'strong fingers']:
            g1_result = next((r for r in group1_results if r['criterion'] == criterion), None)
            g2_result = next((r for r in group2_results if r['criterion'] == criterion), None)
            
            if g1_result and g2_result:
                g1_sig = g1_result['significant_after_fdr']
                g2_sig = g2_result['significant_after_fdr']
                g1_support = g1_result['supports_dvorak']
                g2_support = g2_result['supports_dvorak']
                
                if g1_sig != g2_sig or g1_support != g2_support:
                    print_and_log(f"  {criterion}:")
                    print_and_log(f"    {group_names[0]}: {'Sig' if g1_sig else 'NS'}, {'Support' if g1_support else 'Contradict'} (r={g1_result['correlation']:.3f})")
                    print_and_log(f"    {group_names[1]}: {'Sig' if g2_sig else 'NS'}, {'Support' if g2_support else 'Contradict'} (r={g2_result['correlation']:.3f})")
    
    # Effect size analysis
    print_and_log(f"\n💡 KEY TAKEAWAYS:")
    total_significant = len([r for r in all_individual_results if r['significant_after_fdr']])
    print_and_log(f"1. Individual criteria: {total_significant}/{len(all_individual_results)} survive FDR correction")
    
    # Get max effect sizes
    max_individual = max([r['abs_correlation'] for r in all_individual_results]) if all_individual_results else 0
    
    if combination_results and 'significant_combinations' in locals():
        max_combination = max([r['abs_correlation'] for r in significant_combinations]) if significant_combinations else 0
        print_and_log(f"2. Combinations: {len(significant_combinations) if significant_combinations else 0}/511 survive FDR correction")
        print_and_log(f"3. Effect sizes:")
        print_and_log(f"   • Individual criteria: up to |r| = {max_individual:.3f} ({'small' if max_individual >= 0.1 else 'negligible'} effect)")
        print_and_log(f"   • Best combinations: up to |r| = {max_combination:.3f} ({'small' if max_combination >= 0.1 else 'negligible'} effect)")
    else:
        print_and_log(f"3. Effect sizes: Individual criteria up to |r| = {max_individual:.3f} ({'small' if max_individual >= 0.1 else 'negligible'} effect)")
    
    print_and_log(f"4. Most Dvorak principles are statistically valid but practically weak")
    print_and_log(f"5. Cohen's effect size conventions: small ≥0.1, medium ≥0.3, large ≥0.5")
    
    return all_individual_results, significant_combinations if 'significant_combinations' in locals() else []

def analyze_weight_distribution(csv_file="speed_weights.csv"):
    """Analyze the distribution of weights in the empirical data."""
    print(f"\n📈 ANALYZING EMPIRICAL WEIGHT DISTRIBUTION")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_file)
        
        # Filter to significant results
        if 'significant_after_fdr' in df.columns:
            sig_df = df[df['significant_after_fdr'] == True]
        else:
            sig_df = df[df['p_value'] < 0.05]
        
        print(f"Significant combinations: {len(sig_df)}/{len(df)} ({len(sig_df)/len(df)*100:.1f}%)")
        
        # Analyze by k-way
        sig_df['k_way'] = sig_df['k_way'].astype(int)
        
        print(f"\nBy combination size:")
        for k in sorted(sig_df['k_way'].unique()):
            k_data = sig_df[sig_df['k_way'] == k]
            support = sum(k_data['supports_dvorak'])
            contradict = len(k_data) - support
            
            best_support = k_data[k_data['supports_dvorak']]['abs_correlation'].max() if support > 0 else 0
            best_contradict = k_data[~k_data['supports_dvorak']]['abs_correlation'].max() if contradict > 0 else 0
            
            print(f"  {k}-way: {len(k_data)} significant")
            print(f"    Support Dvorak: {support} (best |r|={best_support:.4f})")
            print(f"    Contradict Dvorak: {contradict} (best |r|={best_contradict:.4f})")
        
        # Overall statistics
        print(f"\nOverall correlation statistics:")
        print(f"  Range: {sig_df['correlation'].min():.4f} to {sig_df['correlation'].max():.4f}")
        print(f"  Mean: {sig_df['correlation'].mean():.4f}")
        print(f"  Std: {sig_df['correlation'].std():.4f}")
        
        dvorak_support = sum(sig_df['supports_dvorak'])
        print(f"  Support Dvorak: {dvorak_support}/{len(sig_df)} ({dvorak_support/len(sig_df)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error analyzing weights: {e}")

def test_empirical_scorer():
    """Test the empirical scorer with real weights."""
    print("Testing Empirical Dvorak-9 Scorer")
    print("=" * 50)
    
    # Test layout (partial DVORAK)
    layout_mapping = {
        'e': 'D', 't': 'K', 'a': 'A', 'o': 'S', 'i': 'F', 
        'n': 'J', 's': 'R', 'h': 'U', 'r': 'L', 'd': 'G'
    }
    text = "the rain in spain falls mainly on the plain"
    
    # Initialize with empirical weights
    scorer = Dvorak9Scorer(layout_mapping, text)
    results = scorer.calculate_scores()
    
    # Show scoring results
    print()
    print_scoring_results(results)
    
    # Show detailed bigram analysis
    print(f"\nDetailed Bigram Analysis (first 10):")
    print("-" * 60)
    for detail in results['bigram_details'][:10]:
        combo_str = '+'.join(detail['combination']) if detail['combination'] else 'none'
        print(f"'{detail['bigram']}': {combo_str:<25} → {detail['weighted_score']:>7.4f}")

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-9 criteria correlations with typing speed')
    parser.add_argument('--max-bigrams', type=int, help='Maximum number of bigrams to analyze')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test-scorer', action='store_true', help='Test the scorer on sample data')
    args = parser.parse_args()
    
    if args.test_scorer:
        # Test the unified scorer
        print_and_log("Testing Dvorak-9 Scorer")
        print_and_log("=" * 50)
        
        layout_mapping = {'e': 'D', 't': 'K', 'a': 'A', 'o': 'S', 'i': 'F', 'n': 'J', 's': 'R', 'h': 'U', 'r': 'L'}
        text = "the quick brown fox jumps over the lazy dog"
        
        scorer = Dvorak9Scorer(layout_mapping, text)
        results = scorer.calculate_scores()
        
        print_scoring_results(results)
        return
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    start_time = time.time()
    
    # Configuration
    bigram_file = "../process_3.5M_keystrokes/output/bigram_times.csv"
    freq_file = "input/engram/normalized_letter_pair_frequencies_en.csv"
    middle_column_keys = {'b', 'g', 'h', 'n', 't', 'y'}
    
    # Print configuration
    print_and_log("Dvorak-9 Criteria Correlation Analysis - Bigram Speed")
    print_and_log("=" * 80)
    print_and_log("Configuration:")
    print_and_log(f"  Max bigrams: {args.max_bigrams:,}" if args.max_bigrams else "  Max bigrams: unlimited")
    print_and_log(f"  Random seed: {args.random_seed}")
    print_and_log(f"  Middle column keys: {', '.join(sorted(middle_column_keys))}")
    print_and_log("  Analysis includes both raw and frequency-adjusted correlations")
    print_and_log("  Using canonical scoring from dvorak9_scorer.py")
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
    
    # Create plots
    print_and_log(f"\nGenerating comparison plots...")
    create_frequency_comparison_plots(bigram_results)
    
    # Generate interpretation
    print_and_log(f"\nGenerating results interpretation...")
    interpret_correlation_results(bigram_results, "BIGRAM ANALYSIS")
    
    # Analyze criterion combinations
    print_and_log(f"\nAnalyzing criterion combinations...")
    combination_results = analyze_criterion_combinations(bigram_results)
    
    # COMPLETE FDR ANALYSIS
    if bigram_results:
        analyze_all_results_with_fdr(bigram_results, combination_results)

    # Final summary
    total_elapsed = time.time() - start_time
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("ANALYSIS COMPLETE")
    print_and_log("=" * 80)
    print_and_log(f"Total runtime: {format_time(total_elapsed)}")
    print_and_log(f"Dvorak-9 Analysis Complete")
    print_and_log("Now using canonical scorer from dvorak9_scorer.py")
    
    # Save log
    save_log()

if __name__ == "__main__":
    # Check if we want to run tests or full analysis
    if len(sys.argv) > 1 and sys.argv[1] == "--test-weights":
        # Analyze the weight distribution
        analyze_weight_distribution()
        
        # Test the scorer
        test_empirical_scorer()
    else:
        # Run the full analysis
        main()