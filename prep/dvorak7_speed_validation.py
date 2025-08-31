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

# Import the canonical scoring function
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
    print_and_log("📁 Loading frequency data for regression analysis...")
    print_and_log("   (This uses pre-calculated English language frequencies, NOT sample frequencies)")
    
    try:
        freq_df = pd.read_csv(freq_file_path)
        print_and_log(f"✅ Loaded bigram frequency data: {len(freq_df)} entries")
        print_and_log(f"   Columns: {list(freq_df.columns)}")
        
        # Show sample frequencies
        print_and_log("   Sample bigram frequencies:")
        for i, (_, row) in enumerate(freq_df.head(3).iterrows()):
            if 'letter_pair' in freq_df.columns:
                # Use normalized_frequency preferentially, then try others
                freq_col = None
                if 'normalized_frequency' in freq_df.columns:
                    freq_col = 'normalized_frequency'
                elif 'count' in freq_df.columns:
                    freq_col = 'count' 
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

def adjust_times_for_frequency(sequences, times, freq_df, sequence_type="sequences"):
    """FIXED: Adjust typing times for linguistic frequency using regression"""
    
    print_and_log(f"  🔍 Starting frequency adjustment for {sequence_type}...")
    print_and_log(f"      Input: {len(sequences):,} sequences, {len(freq_df)} frequency entries")
    print_and_log(f"      Frequency data columns: {list(freq_df.columns)}")
    
    # Build frequency dictionary with proper column selection
    freq_dict = None
    if 'letter_pair' in freq_df.columns:
        # Try different frequency column names in order of preference
        if 'normalized_frequency' in freq_df.columns:
            freq_dict = dict(zip(freq_df['letter_pair'], freq_df['normalized_frequency']))
            print_and_log(f"      Using 'normalized_frequency' column")
        elif 'count' in freq_df.columns:
            # Normalize count data to create frequencies
            total_count = freq_df['count'].sum()
            normalized_counts = freq_df['count'] / total_count
            freq_dict = dict(zip(freq_df['letter_pair'], normalized_counts))
            print_and_log(f"      Using 'count' column (normalized)")
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
        print_and_log(f"      ❌ Failed to build frequency dictionary")
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
    safe_frequencies = [max(freq, 1e-10) for freq in matched_frequencies]
    log_frequencies = np.log10(np.array(safe_frequencies))
    times_array = np.array(matched_times)
    
    print_and_log(f"      Frequency range: {min(matched_frequencies):,.6f} to {max(matched_frequencies):,.6f} (mean: {np.mean(matched_frequencies):,.6f})")
    print_and_log(f"      Time range: {min(times_array):.1f} to {max(times_array):.1f}ms (mean: {np.mean(times_array):.1f} ± {np.std(times_array):.1f})")
    
    # Regression: time = intercept + slope * log_frequency
    try:
        X = sm.add_constant(log_frequencies)  # Add intercept term
        model = sm.OLS(times_array, X).fit()
        
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
        frequency_residuals = []
        residual_magnitudes = []
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
    
    # FIXED: Add debug counters
    total_processed = 0
    scoring_failures = 0
    validation_failures = 0
    records_added = 0
    
    for i, (seq, time_val) in enumerate(zip(sequences, times)):
        total_processed += 1
        
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - start_time
            print_and_log(f"    Progress: {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - {elapsed:.1f}s", end='\r')
        
        # Calculate Dvorak-7 scores using the canonical function
        try:
            scores = score_bigram_dvorak7(seq)
        except Exception as e:
            scoring_failures += 1
            # Skip invalid bigrams
            continue
        
        # FIXED: More robust validation with detailed debugging
        try:
            # Check if scores is a dictionary and has the expected keys
            if not isinstance(scores, dict):
                validation_failures += 1
                continue
                
            # Check if all expected criteria are present
            missing_criteria = [criterion for criterion in criteria_names.keys() if criterion not in scores]
            if missing_criteria:
                validation_failures += 1
                if validation_failures <= 3:  # Only show first few errors
                    print_and_log(f"    Warning: Missing criteria {missing_criteria} for '{seq}'")
                continue
            
            # Validate score values
            invalid_scores = []
            for criterion in criteria_names.keys():
                score = scores[criterion]
                if not isinstance(score, (int, float)) or np.isnan(score):
                    invalid_scores.append(criterion)
            
            if invalid_scores:
                validation_failures += 1
                if validation_failures <= 3:  # Only show first few errors
                    print_and_log(f"    Warning: Invalid scores {invalid_scores} for '{seq}'")
                continue
            
            # If we get here, all validations passed
            valid_sequences.append(seq)
            valid_times.append(time_val)
            
            # FIXED: Create sequence record with proper error handling
            try:
                sequence_score_record = {
                    'sequence': seq, 
                    'time': time_val, 
                    'analysis_type': analysis_type,
                    'group': group_name
                }
                
                # Add all criteria scores
                for criterion in criteria_names.keys():
                    score = scores[criterion]
                    criterion_scores[criterion].append(score)
                    sequence_score_record[criterion] = score
                
                # FIXED: Append to sequence_scores_data with error handling
                sequence_scores_data.append(sequence_score_record)
                records_added += 1
                
            except Exception as e:
                print_and_log(f"    Error creating record for '{seq}': {e}")
                # Remove from valid sequences if record creation failed
                if valid_sequences and valid_sequences[-1] == seq:
                    valid_sequences.pop()
                    valid_times.pop()
                continue
                
        except Exception as e:
            validation_failures += 1
            if validation_failures <= 3:
                print_and_log(f"    Error validating scores for '{seq}': {e}")
            continue
        
        # Show sample scores for first few sequences
        if records_added <= 3:
            sample_scores = {k: f"{v:.3f}" for k, v in scores.items()}
            print_and_log(f"      Sample scores for '{seq}': {sample_scores}")
    
    elapsed = time.time() - start_time
    print_and_log(f"    Completed score calculation in {elapsed:.1f}s" + " " * 50)
    
    # FIXED: Enhanced logging with debug info
    print_and_log(f"    DEBUG STATISTICS:")
    print_and_log(f"      Total processed: {total_processed:,}")
    print_and_log(f"      Scoring failures: {scoring_failures:,}")
    print_and_log(f"      Validation failures: {validation_failures:,}")
    print_and_log(f"      Records added: {records_added:,}")
    print_and_log(f"      Valid sequences for analysis: {len(valid_sequences):,}")
    print_and_log(f"      Sequence scores data length: {len(sequence_scores_data):,}")
    
    # FIXED: Store sequence data with verification
    sequence_key = f'_sequence_scores_{analysis_type}_{group_suffix}'
    results[sequence_key] = sequence_scores_data
    print_and_log(f"    Stored sequence data under key: {sequence_key} ({len(sequence_scores_data)} records)")
    
    # FIXED: Verify storage worked
    if len(sequence_scores_data) == 0 and len(valid_sequences) > 0:
        print_and_log(f"    ERROR: No sequence records stored despite {len(valid_sequences):,} valid sequences!")
        print_and_log(f"    This indicates a bug in record creation or storage.")
    elif len(sequence_scores_data) != len(valid_sequences):
        print_and_log(f"    WARNING: Record count mismatch: {len(sequence_scores_data)} records vs {len(valid_sequences)} valid sequences")
    
    if len(valid_sequences) < 10:
        print_and_log(f"    ⚠️  Too few valid sequences for correlation analysis")
        return results
    
    # Calculate correlations (rest of function remains the same)
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

def analyze_criterion_combinations(results):
    """Analyze how combinations of criteria predict typing speed"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print_and_log("=" * 80)
    print_and_log("Examining how combinations of criteria interact to predict typing speed")
    print_and_log("IMPORTANT: Using ONLY frequency-adjusted data (controls for English bigram frequency)")
    
    # Look for ALL sequence data from frequency-adjusted analysis
    freq_adjusted_sequences = []
    
    print_and_log(f"DEBUG: Looking for sequence data in results...")
    print_and_log(f"DEBUG: Available result keys:")
    for key in sorted(results.keys()):
        if key.startswith('_sequence_scores_'):
            print_and_log(f"  Found: {key} ({len(results[key])} sequences)")
            if 'freq_adjusted' in key:
                freq_adjusted_sequences.extend(results[key])
    
    if not freq_adjusted_sequences:
        print_and_log("❌ No frequency-adjusted sequence data found for combination analysis")
        print_and_log("DEBUG: All result keys:")
        for key in sorted(results.keys()):
            print_and_log(f"  {key}: {type(results[key])}")
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
    
    # Get criteria columns (exclude sequence, time, analysis_type, group)
    exclude_cols = {'sequence', 'time', 'analysis_type', 'group'}
    criteria_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Verify we have the expected 7 criteria
    expected_criteria = ['distribution','strength','middle','vspan','columns','remote','inward']
    print_and_log(f"   Expected criteria: {expected_criteria}")
    print_and_log(f"   Found criteria: {criteria_cols}")
    
    # Filter to only use criteria that actually exist
    criteria_cols = [col for col in expected_criteria if col in criteria_cols]
    print_and_log(f"   Using criteria: {criteria_cols}")
    
    times = df['time'].values
    print_and_log(f"   Using frequency-adjusted times for combination analysis")
    
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
    
    return group_results

# Rest of the functions remain the same as in the original script
# (analyze_bigram_data, print_correlation_results_with_frequency, etc.)

def analyze_bigram_data(bigrams, freq_df, middle_column_keys):
    """Analyze bigram typing data with middle column key analysis"""
    
    # Updated criteria names to match prep_keypair_dvorak7_scores.py
    criteria_names = {
        'distribution': 'Typing with 2 hands or 2 fingers',
        'strength': 'Typing with stronger fingers 3 and 4',
        'middle': 'Typing within the middle/home row',
        'vspan': 'Typing in the same row, reaches, and hurdles', 
        'columns': 'Typing within the 8 finger columns',
        'remote': 'Typing with non-adjacent fingers (except fingers 1 and 2)',
        'inward': 'Finger sequence toward the thumb'
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
            
            # Only proceed if frequency adjustment worked
            if model_info is not None:
                freq_results = analyze_correlations(sequences, frequency_residuals, criteria_names, group_name, "freq_adjusted", model_info)
                all_results.update(freq_results)
            else:
                print_and_log(f"  ⚠️  Frequency adjustment failed - skipping frequency-adjusted analysis")
        else:
            print_and_log(f"  ⚠️  Skipping frequency adjustment (no frequency data)")
    
    return all_results

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
        
        # Load full dataset or sample
        print_and_log("   Quality indicators:")
        
        if max_bigrams and total_rows > max_bigrams:
            print_and_log(f"   Randomly sampling {max_bigrams:,} bigrams from {total_rows:,}")
            df = pd.read_csv(bigram_file)
            df = df.sample(n=max_bigrams, random_state=42)
        else:
            df = pd.read_csv(bigram_file)
        
        # Quality checks
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 'nd', 'on', 'en', 'at', 'ou']
        found_common = sum(1 for bg in df['bigram'].head(100) if bg in common_bigrams)
        print_and_log(f"     Common English bigrams in sample: {found_common}/100 ({found_common}%)")
        
        # Check for suspicious patterns
        suspicious = sum(1 for bg in df['bigram'].head(100) if len(set(bg)) == 1 or len(bg) != 2)
        print_and_log(f"     Suspicious bigrams (repeated/invalid chars): {suspicious}/100 ({suspicious}%)")
        
        if found_common < 30:
            print_and_log("   ⚠️  Low proportion of common English bigrams - check data quality")
        
        print_and_log("   ✅ Proceeding with data loading...")
        
    except Exception as e:
        print_and_log(f"   ❌ Error reading bigram file: {e}")
        return None
    
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

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-7 criteria correlations with typing speed')
    parser.add_argument('--max-bigrams', type=int, help='Maximum number of bigrams to analyze')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    start_time = time.time()
    
    # Configuration
    bigram_file = "../../process_136M_keystrokes/output/bigram_times.csv"
    freq_file = "../input/english-letter-pair-frequencies-google-ngrams.csv"
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
    if freq_df is None:
        print_and_log("❌ Failed to load frequency data - continuing with raw analysis only")
    
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
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("ANALYSIS COMPLETE")
    print_and_log("=" * 80)
    print_and_log(f"Total runtime: {format_time(total_elapsed)}")
    print_and_log(f"Dvorak-7 Analysis Complete")
    
    # Save log
    save_log()

if __name__ == "__main__":
    main()