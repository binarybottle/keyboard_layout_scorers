#!/usr/bin/env python3
"""
Generate Dvorak-9 empirical weights based on comfort scores for 30-key coverage.

This script uses the extended comfort dataset (30 keys) to generate empirical weights
for Dvorak-9 layout scoring. Higher comfort scores = better layouts.

Key features:
- Uses extended comfort scores covering all 30 standard typing keys
- Positive correlation = good (higher Dvorak score = more comfortable)
- Tests all 511 possible combinations of 9 Dvorak criteria
- Applies FDR correction for multiple testing
- No artificial assumptions needed - uses real comfort data throughout

Usage:
    python generate_comfort_weights.py
    python generate_comfort_weights.py --comfort-file input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv
    python generate_comfort_weights.py --min-uncertainty 0.1
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict, Counter
from itertools import combinations
import argparse
from statsmodels.stats.multitest import multipletests

# Import the canonical scoring function
from dvorak9_scorer import score_bigram_dvorak9

def load_comfort_data(comfort_file="input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv"):
    """Load extended comfort scores for position-pairs (3-key coverage)"""
    print(f"Loading comfort data from {comfort_file}...")
    
    try:
        df = pd.read_csv(comfort_file)
        print(f"✅ Loaded {len(df)} position-pairs with comfort scores")
        print(f"   Columns: {list(df.columns)}")
        
        # Remove rows with missing key_pair values
        original_count = len(df)
        df = df.dropna(subset=['key_pair'])
        filtered_count = len(df)
        
        if filtered_count < original_count:
            print(f"   Removed {original_count - filtered_count} rows with missing key_pair values")
            print(f"   Working with {filtered_count} valid position-pairs")
        
        # Show sample data
        print("   Sample comfort scores:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            source = row.get('source', 'unknown')
            print(f"     '{row['key_pair']}': {row['comfort_score']:.3f} (±{row['uncertainty']:.3f}) [{source}]")
        
        # Data quality checks
        print(f"   Comfort score range: {df['comfort_score'].min():.3f} to {df['comfort_score'].max():.3f}")
        print(f"   Average uncertainty: {df['uncertainty'].mean():.3f}")
        
        # Show source breakdown if available
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print(f"   Data sources:")
            for source, count in source_counts.items():
                print(f"     • {source}: {count} pairs ({count/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading comfort data: {e}")
        return None

def filter_by_uncertainty(df, min_uncertainty=None, max_uncertainty=None):
    """Filter comfort data by uncertainty thresholds"""
    if min_uncertainty is None and max_uncertainty is None:
        return df
    
    original_count = len(df)
    
    if min_uncertainty is not None:
        df = df[df['uncertainty'] >= min_uncertainty]
    
    if max_uncertainty is not None:
        df = df[df['uncertainty'] <= max_uncertainty]
    
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    print(f"Filtered {removed_count}/{original_count} pairs by uncertainty")
    print(f"  Kept {filtered_count} pairs ({filtered_count/original_count*100:.1f}%)")
    
    return df

def analyze_comfort_correlations(comfort_df):
    """Analyze correlations between Dvorak criteria and comfort scores"""
    
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
    
    print(f"\nAnalyzing comfort correlations for {len(comfort_df)} position-pairs...")
    print("Using extended 30-key comfort dataset with actual comfort scores for all bigrams")
    
    # Convert position-pairs to sequences for Dvorak scoring
    sequences = []
    comfort_scores = []
    uncertainties = []
    
    for _, row in comfort_df.iterrows():
        pos_pair = row['key_pair']
        
        # Skip NaN or invalid values  
        if pd.isna(pos_pair) or not isinstance(pos_pair, str):
            continue
            
        if len(pos_pair) == 2:
            sequences.append(pos_pair.lower())
            comfort_scores.append(row['comfort_score'])
            uncertainties.append(row['uncertainty'])
    
    print(f"✅ Processed {len(sequences)} valid bigrams from extended dataset")
    
    # Calculate Dvorak scores for ALL sequences
    print("Calculating Dvorak criterion scores...")
    
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_indices = []
    
    for i, seq in enumerate(sequences):
        scores = score_bigram_dvorak9(seq)
        
        if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
            valid_indices.append(i)
            for criterion in criteria_names.keys():
                criterion_scores[criterion].append(scores[criterion])
    
    valid_sequences = [sequences[i] for i in valid_indices]
    valid_comfort = [comfort_scores[i] for i in valid_indices]
    valid_uncertainty = [uncertainties[i] for i in valid_indices]
    
    print(f"✅ Valid sequences for correlation: {len(valid_sequences)}")
    
    # Calculate correlations for individual criteria
    print(f"\n📊 INDIVIDUAL CRITERIA ANALYSIS:")
    results = {}
    
    for criterion, scores_list in criterion_scores.items():
        if len(scores_list) >= 3:
            try:
                # Check for constant values
                unique_scores = len(set(scores_list))
                if unique_scores <= 1:
                    print(f"    {criterion}: constant scores (all {scores_list[0]:.3f})")
                    results[criterion] = {
                        'name': criteria_names[criterion],
                        'correlation': float('nan'),
                        'p_value': float('nan'),
                        'abs_correlation': float('nan'),
                        'n_samples': len(scores_list),
                        'supports_dvorak': None,
                        'constant_scores': True
                    }
                else:
                    spearman_r, spearman_p = spearmanr(scores_list, valid_comfort)
                    
                    results[criterion] = {
                        'name': criteria_names[criterion],
                        'correlation': spearman_r,
                        'p_value': spearman_p,
                        'abs_correlation': abs(spearman_r),
                        'n_samples': len(scores_list),
                        'supports_dvorak': spearman_r > 0,  # Positive = supports Dvorak for comfort
                        'scores': scores_list.copy(),
                        'comfort_scores': valid_comfort.copy()
                    }
                    
                    direction = "supports" if spearman_r > 0 else "contradicts"
                    print(f"    {criterion}: r = {spearman_r:.3f}, p = {spearman_p:.3f} ({direction} Dvorak)")
                
            except Exception as e:
                print(f"    Error calculating correlation for {criterion}: {e}")
                continue
    
    # Store sequence data for combination analysis
    sequence_data = []
    for i, seq in enumerate(valid_sequences):
        record = {
            'sequence': seq,
            'comfort_score': valid_comfort[i],
            'uncertainty': valid_uncertainty[i]
        }
        
        for criterion in criteria_names.keys():
            if i < len(criterion_scores[criterion]):
                record[criterion] = criterion_scores[criterion][i]
        
        sequence_data.append(record)
    
    results['_sequence_data'] = sequence_data
    
    return results

def analyze_comfort_combinations(sequence_data):
    """Analyze combinations of criteria for comfort prediction"""
    
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print("=" * 80)
    print(f"Testing all 511 combinations on {len(sequence_data)} bigrams")
    print("Using extended 30-key comfort dataset")
    
    if len(sequence_data) < 10:
        print("❌ Too few sequences for combination analysis")
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(sequence_data)
    
    # Get criteria columns
    exclude_cols = {'sequence', 'comfort_score', 'uncertainty'}
    criteria_cols = [col for col in df.columns if col not in exclude_cols]
    
    comfort_scores = df['comfort_score'].values
    
    # Check for variation in comfort scores
    unique_comfort = len(set(comfort_scores))
    if unique_comfort <= 1:
        print(f"⚠️  No variation in comfort scores (all = {comfort_scores[0]:.3f})")
        print("Cannot calculate meaningful correlations")
        return {}
    
    print(f"Comfort score range: {min(comfort_scores):.3f} to {max(comfort_scores):.3f}")
    print(f"Testing all combinations of {len(criteria_cols)} criteria...")
    
    # Test all combinations (1-way through 9-way)
    all_results = {}
    
    for k in range(1, len(criteria_cols) + 1):
        print(f"\n📊 {k}-WAY COMBINATIONS:")
        
        combos = list(combinations(criteria_cols, k))
        combo_results = []
        
        print(f"   Testing ALL {len(combos):,} combinations of {k} criteria...")
        
        for combo in combos:
            # Create combined score (additive model)
            combined_scores = np.zeros(len(comfort_scores))
            for criterion in combo:
                combined_scores += df[criterion].values
            
            # Test correlation with comfort
            if len(set(combined_scores)) > 1:
                try:
                    corr, p_val = spearmanr(combined_scores, comfort_scores)
                    if not (np.isnan(corr) or np.isnan(p_val)):
                        combo_results.append({
                            'combination': ' + '.join(combo),
                            'criteria_count': k,
                            'correlation': corr,
                            'p_value': p_val,
                            'abs_correlation': abs(corr),
                            'supports_dvorak': corr > 0  # Positive = supports for comfort
                        })
                except:
                    continue
        
        # Sort by absolute correlation
        combo_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        all_results[f'{k}_way'] = combo_results
        
        # Show top results
        if combo_results:
            print(f"   Top 5 combinations:")
            for i, result in enumerate(combo_results[:5]):
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                direction = "supports" if result['supports_dvorak'] else "contradicts"
                print(f"     {i+1}. {result['combination']}")
                print(f"        r = {result['correlation']:.4f}{sig}, {direction} Dvorak")
        else:
            print(f"   No valid combinations found")
    
    return all_results

def apply_fdr_correction(individual_results, combination_results):
    """Apply FDR correction to all results"""
    
    print(f"\n" + "=" * 80)
    print("COMFORT-BASED FDR ANALYSIS")
    print("=" * 80)
    print("Using extended 30-key comfort dataset with FDR multiple testing correction")
    
    # PART 1: Individual criteria
    print(f"\n📊 INDIVIDUAL CRITERIA (9 tests):")
    
    individual_p_values = []
    individual_data = []
    
    for criterion, data in individual_results.items():
        if criterion != '_sequence_data' and isinstance(data, dict) and 'p_value' in data:
            if not np.isnan(data['p_value']):
                individual_p_values.append(data['p_value'])
                individual_data.append((criterion, data))
    
    significant_individual = []
    if individual_p_values:
        rejected, p_adj, _, _ = multipletests(individual_p_values, alpha=0.05, method='fdr_bh')
        
        print("Criterion              r      p-val    FDR p-val  Significant  Dvorak")
        print("-" * 75)
        
        for i, (criterion, data) in enumerate(individual_data):
            sig_marker = "✅" if rejected[i] else "❌"
            dvorak_marker = "✅ Support" if data.get('supports_dvorak') else "❌ Contradict"
            
            if rejected[i]:
                significant_individual.append(data)
            
            print(f"{data['name']:<18} {data['correlation']:>6.3f}  "
                  f"{data['p_value']:>6.3f}  {p_adj[i]:>8.3f}  {sig_marker:<11}  {dvorak_marker}")
    
    # PART 2: All combinations
    print(f"\n📊 ALL COMBINATIONS (511 tests):")
    
    all_combinations = []
    for k_way, results_list in combination_results.items():
        for result in results_list:
            all_combinations.append({
                'combination': result['combination'],
                'k_way': result['criteria_count'],
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'abs_correlation': result['abs_correlation'],
                'supports_dvorak': result['supports_dvorak']
            })
    
    significant_combinations = []
    if all_combinations:
        p_values = [r['p_value'] for r in all_combinations]
        rejected, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Add FDR results
        for i, result in enumerate(all_combinations):
            result['p_fdr_corrected'] = p_adj[i]
            result['significant_after_fdr'] = rejected[i]
        
        # Save all results
        df = pd.DataFrame(all_combinations)
        df.to_csv('comfort_weights.csv', index=False)
        print(f"💾 ALL COMBINATIONS SAVED TO: comfort_weights.csv")

        # Filter significant
        significant_combinations = [r for r in all_combinations if r['significant_after_fdr']]
        significant_combinations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"Significant after FDR: {len(significant_combinations)}/{len(all_combinations)} "
              f"({len(significant_combinations)/len(all_combinations)*100:.1f}%)")
        
        if significant_combinations:
            print(f"\nTop 10 significant combinations:")
            print("K  Combination                           r       FDR p-val  Dvorak")
            print("-" * 75)
            
            for i, result in enumerate(significant_combinations[:10]):
                dvorak_marker = "✅" if result['supports_dvorak'] else "❌"
                combo_short = result['combination'][:35] + "..." if len(result['combination']) > 35 else result['combination']
                print(f"{result['k_way']}  {combo_short:<35} {result['correlation']:>7.3f}  "
                      f"{result['p_fdr_corrected']:>8.3f}  {dvorak_marker}")
            
            # Save significant results
            sig_df = pd.DataFrame(significant_combinations)
            sig_df.to_csv('comfort_weights.csv', index=False)
            print(f"💾 SIGNIFICANT COMBINATIONS SAVED TO: comfort_weights.csv")
            
            # Best combination overall
            best_combo = significant_combinations[0]
            print(f"\n🏆 STRONGEST SIGNIFICANT COMBINATION:")
            print(f"   {best_combo['combination']}")
            print(f"   r = {best_combo['correlation']:.4f} (FDR p = {best_combo['p_fdr_corrected']:.3f})")
            print(f"   Uses {best_combo['k_way']} criteria")
            print(f"   {'Supports' if best_combo['supports_dvorak'] else 'Contradicts'} Dvorak principles")
            
        else:
            print("❌ No combinations survived FDR correction!")
    else:
        print("❌ No combination results found!")
    
    # PART 3: Summary
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    print(f"Extended 30-key comfort analysis results:")
    
    if significant_individual:
        print(f"  • Individual criteria: {len(significant_individual)}/9 significant after FDR")
        best_individual = max(significant_individual, key=lambda x: x['abs_correlation'])
        print(f"    Best: {best_individual['name']} (r = {best_individual['correlation']:.3f})")
    else:
        print(f"  • Individual criteria: 0/9 significant after FDR")
    
    if significant_combinations:
        print(f"  • Combinations: {len(significant_combinations)}/511 significant after FDR")
        best_combo = significant_combinations[0]
        print(f"    Best: {best_combo['combination'][:50]}...")
        print(f"          r = {best_combo['correlation']:.3f}")
        print(f"          Uses {best_combo['k_way']} criteria")
    else:
        print(f"  • Combinations: 0/511 significant after FDR")
    
    print(f"  • Effect sizes: All correlations represent comfort-criterion relationships")
    print(f"  • Positive correlation = higher Dvorak score → more comfortable")
    
    return significant_combinations

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Generate Dvorak-9 weights based on 32-key comfort scores')
    parser.add_argument('--comfort-file', default='input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv',
                       help='Path to extended comfort scores CSV file')
    parser.add_argument('--min-uncertainty', type=float,
                       help='Minimum uncertainty threshold for filtering')
    parser.add_argument('--max-uncertainty', type=float,
                       help='Maximum uncertainty threshold for filtering')
    
    args = parser.parse_args()
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    print("Dvorak-9 Comfort-Based Weight Generation (30-Key Extended)")
    print("=" * 60)
    print(f"Input file: {args.comfort_file}")
    if args.min_uncertainty:
        print(f"Min uncertainty: {args.min_uncertainty}")
    if args.max_uncertainty:
        print(f"Max uncertainty: {args.max_uncertainty}")
    print()
    
    # Load comfort data
    comfort_df = load_comfort_data(args.comfort_file)
    if comfort_df is None:
        return
    
    # Filter by uncertainty if specified
    if args.min_uncertainty or args.max_uncertainty:
        comfort_df = filter_by_uncertainty(comfort_df, args.min_uncertainty, args.max_uncertainty)
    
    # Analyze individual criteria correlations
    individual_results = analyze_comfort_correlations(comfort_df)
    
    # Analyze combinations
    sequence_data = individual_results.get('_sequence_data', [])
    combination_results = analyze_comfort_combinations(sequence_data)
    
    # Apply FDR correction
    significant_combinations = apply_fdr_correction(individual_results, combination_results)
    
    print(f"\n" + "=" * 60)
    print("COMFORT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Generated comfort-based weights for Dvorak-9 scoring")
    print(f"✅ Covers all 30 standard typing keys using extended comfort dataset")
    print(f"✅ Use 'comfort_weights.csv' with dvorak9_scorer.py")
    print(f"✅ Interpretation: Positive correlation = supports Dvorak (more comfortable)")

if __name__ == "__main__":
    main()