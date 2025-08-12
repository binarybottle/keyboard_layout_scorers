#!/usr/bin/env python3
"""
Keyboard Layout Scorer using precomputed score table.

A comprehensive tool for evaluating keyboard layouts using frequency-weighted scoring.
Scoring methods include engram, comfort, comfort-key, distance, time, dvorak7, and dvorak7-speed.

Setup:
1. Generate individual score files (keypair_*_scores.csv) using your scoring scripts
2. Run: python prep_scoring_tables.py --input-dir tables/
   This creates: tables/keypair_scores.csv and tables/key_scores.csv
3. Run this script to score layouts using all available methods

Default behavior:
- Score table: tables/keypair_scores.csv (created by prep_scoring_tables.py)
- Key scores: tables/key_scores.csv (created by prep_scoring_tables.py)  
- Frequency data: input/english-letter-pair-frequencies-google-ngrams.csv
- Letter frequencies: input/english-letter-frequencies-google-ngrams.csv
- Scoring mode: Frequency-weighted (prioritizes common English letter combinations)
- Score mapping: Letter-pair frequencies → Key-pair scores (distance/time inverted)

Dvorak-7 Speed Scoring:
- dvorak7-speed: Empirically-weighted Dvorak-7 using 19.4M typing speed correlations
- Provides both pure Dvorak-7 scores and speed-weighted scores
- Based on actual typing data analysis with FDR correction
- Requires: input/dvorak7_speed_weights.csv (from empirical analysis)

Usage:
    # Single layout evaluation (all available scoring methods)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
    
    # Force raw (unweighted) scoring
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --raw
    
    # Compare multiple layouts  
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;"
    
    # Pure vs speed-weighted Dvorak-7 comparison
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --scorer dvorak7-speed
    
    # Both pure and speed-weighted Dvorak-7
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --scorers dvorak7,dvorak7-speed
    
    # Verbose output (shows both weighted and raw scores)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --verbose
    
    # Minimal CSV output for scripts/automation
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv-output
    
    # Single scoring method only
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --scorer distance
    
    # Save detailed results to file
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv results.csv
    
    # Custom files (when defaults aren't suitable)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-table custom.csv --frequency-file custom_freq.csv

Key features:
- Automatic frequency weighting using English bigram frequencies
- Smart score inversion (distance/time: higher is worse → lower scores)  
- Letter-pair → Key-pair mapping (e.g., "TH" frequency weights T→H key transition)
- Multiple output formats: detailed, CSV, minimal CSV, score-only
- Graceful fallback to raw scoring if frequency file missing
- Support for all scoring methods in the unified score table
- Dynamic engram and comfort-key scoring (requires prep_scoring_tables.py output)
- Empirical Dvorak-7 speed scoring with both pure and speed-weighted results
"""

import sys
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

class LayoutScorer:
    """Unified layout scorer using pre-computed score table."""
    
    def __init__(self, score_table_path: str, frequency_file: Optional[str] = None, use_raw: bool = False, verbose: bool = False):
        """Initialize scorer with score table and optional frequency data."""
        self.verbose = verbose
        self.use_raw = use_raw
        self.score_table = self._load_score_table(score_table_path)
        self.available_scorers = self._detect_available_scorers()
        
        # Always load frequency data for letter-pairs (needed for consistent scoring)
        if frequency_file:
            self.bigram_frequencies = self._load_frequency_data(frequency_file)
        else:
            self.bigram_frequencies = None
        
        # Load additional data for dynamic scoring methods (engram and comfort-key)
        self.letter_frequencies = self._load_letter_frequencies()
        self.key_comfort_scores = self._load_key_comfort_scores()  # from tables/key_scores.csv
        
        # Load Dvorak-7 speed weights if available
        self.dvorak7_speed_weights = self._load_dvorak7_speed_weights()
        
        # Add dynamic scorers to available list
        if self.letter_frequencies is not None and self.key_comfort_scores is not None:
            if 'comfort-key' not in self.available_scorers:
                self.available_scorers.append('comfort-key')
            if 'comfort' in self.available_scorers and 'engram' not in self.available_scorers:
                self.available_scorers.append('engram')
        
        # Add dvorak7-speed to available scorers if weights are loaded
        if self.dvorak7_speed_weights is not None:
            if 'dvorak7-speed' not in self.available_scorers:
                self.available_scorers.append('dvorak7-speed')
        
        if self.verbose:
            print(f"Score table: {score_table_path}")
            if frequency_file:
                print(f"Frequency file: {frequency_file}")
            print(f"Loaded score table with {len(self.score_table)} key pairs")
            print(f"Available scorers: {', '.join(self.available_scorers)}")
            if self.letter_frequencies is not None:
                print(f"Loaded letter frequencies for {len(self.letter_frequencies)} letters")
            if self.key_comfort_scores is not None:
                print(f"Loaded key comfort scores for {len(self.key_comfort_scores)} keys")
            if self.dvorak7_speed_weights is not None:
                print(f"Loaded Dvorak-7 speed weights for empirical scoring")
            if self.use_raw:
                print("Using raw (unweighted) scoring only")
            elif self.bigram_frequencies is not None:
                print(f"Loaded frequency data for {len(self.bigram_frequencies)} bigrams")
                total_freq = sum(self.bigram_frequencies.values())
                print(f"Total frequency count: {total_freq:,}")
                print("Using frequency-weighted scoring by default")
            else:
                print("No frequency file found - using raw scoring")
    
    def _load_score_table(self, filepath: str) -> pd.DataFrame:
            """Load the unified score table."""
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Score table not found: {filepath}")
            
            try:
                # CRITICAL: Prevent pandas from converting 'NA' to NaN by preserving literal 'NA' strings
                df = pd.read_csv(filepath, 
                            dtype={'key_pair': 'str'},
                            keep_default_na=False,
                            na_values=['', 'NULL', 'null', 'NaN', 'nan'])
            except Exception as e:
                raise ValueError(f"Error reading score table: {e}")
            
            if 'key_pair' not in df.columns:
                raise ValueError("Score table must have 'key_pair' column")
            
            # Check for any truly missing values in key_pair column after loading
            # (but don't treat 'NA' as missing)
            na_count = df['key_pair'].isna().sum()
            if na_count > 0:
                print(f"Warning: Found {na_count} truly missing values in key_pair column")
                # Remove rows with truly missing key_pairs
                df = df.dropna(subset=['key_pair'])
                print(f"Removed missing rows, remaining: {len(df)} rows")
            
            # Check for empty string values
            empty_count = (df['key_pair'].astype(str).str.strip() == '').sum()
            if empty_count > 0:
                print(f"Warning: Found {empty_count} empty string values in key_pair column")
                # Remove these rows
                df = df[df['key_pair'].astype(str).str.strip() != '']
                print(f"Removed empty rows, remaining: {len(df)} rows")
            
            # Debug: Confirm 'NA' is preserved
            if self.verbose and 'NA' in df['key_pair'].values:
                print(f"✓ 'NA' key pair preserved in score table")
            
            # Set key_pair as index for fast lookup
            df = df.set_index('key_pair')
            return df

    def _load_frequency_data(self, filepath: str) -> Dict[str, float]:
        """Load bigram frequency data from CSV file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Frequency file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Error reading frequency file: {e}")
        
        # Try to detect column names
        possible_bigram_cols = ['bigram', 'pair', 'key_pair', 'letter_pair']
        possible_freq_cols = ['normalized_frequency', 'frequency']

        bigram_col = None
        freq_col = None
        
        for col in possible_bigram_cols:
            if col in df.columns:
                bigram_col = col
                break
        
        for col in possible_freq_cols:
            if col in df.columns:
                freq_col = col
                break
        
        if bigram_col is None:
            raise ValueError(f"Could not find bigram column. Expected one of: {possible_bigram_cols}")
        
        if freq_col is None:
            raise ValueError(f"Could not find frequency column. Expected one of: {possible_freq_cols}")
        
        if self.verbose:
            print(f"Using bigram column: '{bigram_col}', frequency column: '{freq_col}'")
        
        # Load frequencies
        frequencies = {}
        total_count = 0
        
        for _, row in df.iterrows():
            bigram = str(row[bigram_col]).strip().upper()
            freq = float(row[freq_col])
            
            if len(bigram) == 2:
                frequencies[bigram] = freq
                total_count += freq
        
        if self.verbose:
            print(f"Loaded {len(frequencies)} bigram frequencies")
            print(f"Total frequency count: {total_count:,}")
        
        return frequencies
    
    def _load_letter_frequencies(self) -> Optional[Dict[str, float]]:
        """Load individual letter frequency data."""
        filepath = "input/english-letter-frequencies-google-ngrams.csv"
        
        if not Path(filepath).exists():
            if self.verbose:
                print(f"Letter frequency file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            if self.verbose:
                print(f"Error reading letter frequency file: {e}")
            return None
        
        # Try to detect column names
        possible_letter_cols = ['letter', 'char', 'character']
        possible_freq_cols = ['normalized_frequency', 'frequency']

        letter_col = None
        freq_col = None
        
        for col in possible_letter_cols:
            if col in df.columns:
                letter_col = col
                break
        
        for col in possible_freq_cols:
            if col in df.columns:
                freq_col = col
                break
        
        if letter_col is None or freq_col is None:
            if self.verbose:
                print(f"Could not find required columns in letter frequency file")
            return None
        
        # Load frequencies
        frequencies = {}
        total_count = 0
        
        for _, row in df.iterrows():
            letter = str(row[letter_col]).strip().upper()
            freq = float(row[freq_col])
            
            if len(letter) == 1:
                frequencies[letter] = freq
                total_count += freq
        
        ## Normalize frequencies to proportions (sum to 1.0)
        #if total_count > 0:
        #    for letter in frequencies:
        #        frequencies[letter] = frequencies[letter] / total_count
        
        if self.verbose:
            print(f"Loaded and normalized letter frequencies for {len(frequencies)} letters")
            print(f"Total frequency sum: {sum(frequencies.values()):.6f}")
        
        return frequencies
    
    def _load_key_comfort_scores(self) -> Optional[Dict[str, float]]:
        """Load key comfort scores from tables/key_scores.csv (created by prep_scoring_tables.py)."""
        filepath = "tables/key_scores.csv"
        
        if not Path(filepath).exists():
            if self.verbose:
                print(f"Key comfort scores file not found: {filepath}")
                print("Run 'python prep_scoring_tables.py --input-dir tables/' to create this file")
            return None
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            if self.verbose:
                print(f"Error reading key comfort scores file: {e}")
            return None
        
        if 'key' not in df.columns or 'comfort_score' not in df.columns:
            if self.verbose:
                print("Key comfort scores file must have 'key' and 'comfort_score' columns")
            return None
        
        # Load comfort scores
        scores = {}
        for _, row in df.iterrows():
            key = str(row['key']).strip().upper()
            score = float(row['comfort_score'])
            scores[key] = score
        
        if self.verbose:
            print(f"Loaded key comfort scores for {len(scores)} keys from {filepath}")
        
        return scores
    
    def _load_dvorak7_speed_weights(self) -> Optional[Dict]:
        """Load Dvorak-7 empirical speed weights."""
        filepath = "input/dvorak7_speed_weights.csv"
        
        if not Path(filepath).exists():
            if self.verbose:
                print(f"Dvorak-7 speed weights not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['combination', 'k_way', 'correlation', 'significant_after_fdr']
            for col in required_cols:
                if col not in df.columns:
                    if self.verbose:
                        print(f"Required column '{col}' not found in speed weights file")
                    return None
            
            # Filter to only significant results
            significant_df = df[df['significant_after_fdr'] == True]
            
            # Use full keyboard weights as default (from "with middle columns" analysis)
            # This covers all 32 keys including middle columns (B,G,H,N,T,Y)
            weights = {
                'vertical': -0.105,      # Row separation 
                'repetition': -0.101,    # Hand/finger alternation
                'adjacent': -0.088,      # Avoid adjacent fingers
                'movement': -0.002,      # Home row usage (minimal effect)
                'weak': +0.065,          # Avoid weak fingers (contradicts Dvorak)
                'outward': +0.024,       # Finger rolls (contradicts Dvorak) 
                'horizontal': +0.019     # Column adherence (contradicts Dvorak)
            }
            
            if self.verbose:
                print(f"Loaded Dvorak-7 speed weights: {len(significant_df)} combinations")
                print(f"Using full keyboard weights (32-key layout)")
            
            return {
                'combinations': significant_df,
                'weights': weights,
                'best_combination': significant_df.iloc[0] if len(significant_df) > 0 else None
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading Dvorak-7 speed weights: {e}")
            return None
    
    def _detect_available_scorers(self) -> List[str]:
        """Detect available scoring methods from table columns."""
        scorers = []
        
        # Look for normalized score columns
        for col in self.score_table.columns:
            if col.endswith('_normalized'):
                scorer_name = col.replace('_normalized', '').replace('_score', '')
                scorers.append(scorer_name)
        
        return sorted(list(set(scorers)))
       
    def _compute_comfort_key_score(self, letter_pair: str, layout_mapping: Dict[str, str]) -> Optional[float]:
        """Compute comfort-key score for a letter pair."""
        if len(letter_pair) != 2:
            return None
        
        if self.letter_frequencies is None or self.key_comfort_scores is None:
            return None
        
        letter1, letter2 = letter_pair[0], letter_pair[1]
        
        # Get keys for these letters
        if letter1 not in layout_mapping or letter2 not in layout_mapping:
            return None
        
        key1 = layout_mapping[letter1]
        key2 = layout_mapping[letter2]
        
        # Get letter frequencies (now normalized to sum to 1.0)
        freq1 = self.letter_frequencies.get(letter1, 0.0)
        freq2 = self.letter_frequencies.get(letter2, 0.0)
        
        # Get key comfort scores (already normalized 0-1)
        comfort1 = self.key_comfort_scores.get(key1, 0.0)
        comfort2 = self.key_comfort_scores.get(key2, 0.0)
        
        # Compute frequency-weighted average comfort score
        total_frequency = freq1 + freq2
        if total_frequency > 0:
            comfort_key_score = (comfort1 * freq1 + comfort2 * freq2) / total_frequency
        else:
            # Fallback to simple average if no frequency data
            comfort_key_score = (comfort1 + comfort2) / 2.0
        
        return comfort_key_score
    
    def _score_layout_dvorak7_speed(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score layout using empirical Dvorak-7 speed weights."""
        # Import the canonical scoring function
        try:
            from prep_keypair_dvorak7_scores import score_bigram_dvorak7
        except ImportError:
            raise ValueError("prep_keypair_dvorak7_scores module not found - required for dvorak7-speed scoring")
        
        if self.dvorak7_speed_weights is None:
            raise ValueError("Dvorak-7 speed weights not available")
        
        # Get empirical weights (from full 32-key analysis)
        weights = self.dvorak7_speed_weights['weights']
        
        # Use actual English bigrams from frequency data
        if self.bigram_frequencies:
            letter_pairs = list(self.bigram_frequencies.keys())
        else:
            # Fallback to common English bigrams
            letter_pairs = ['TH', 'HE', 'IN', 'ER', 'AN', 'ND', 'ON', 'EN', 'AT', 'OU',
                          'ED', 'HA', 'TO', 'OR', 'IT', 'IS', 'HI', 'ES', 'NG', 'VE']
        
        # Initialize scoring
        pure_scores = []
        speed_weighted_scores = []
        
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        weighted_total_pure = 0.0
        weighted_total_speed = 0.0
        total_frequency = 0.0
        
        for letter_pair in letter_pairs:
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                
                # Only score bigrams where both letters exist in the layout
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    try:
                        # Get Dvorak-7 individual criteria scores
                        criteria_scores = score_bigram_dvorak7(letter_pair)
                        
                        # Calculate pure Dvorak-7 score (simple average)
                        pure_score = sum(criteria_scores.values()) / len(criteria_scores)
                        pure_scores.append(pure_score)
                        
                        # Calculate speed-weighted score using empirical weights
                        speed_score = 0.0
                        total_abs_weight = 0.0
                        
                        for criterion, score in criteria_scores.items():
                            weight = weights.get(criterion, 0.0)
                            # Negative correlation = good for typing, so negate for positive contribution
                            contribution = score * (-weight)
                            speed_score += contribution
                            total_abs_weight += abs(weight)
                        
                        # Normalize to keep in reasonable range
                        if total_abs_weight > 0:
                            speed_score = speed_score / total_abs_weight
                        else:
                            speed_score = pure_score
                        
                        speed_weighted_scores.append(speed_score)
                        
                        # Frequency weighting (if enabled)
                        if use_frequency:
                            frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                            weighted_total_pure += pure_score * frequency
                            weighted_total_speed += speed_score * frequency
                            total_frequency += frequency
                        
                    except Exception as e:
                        # Skip bigrams that can't be scored
                        if self.verbose:
                            print(f"Warning: Could not score bigram '{letter_pair}': {e}")
                        continue
        
        # Calculate final scores
        if use_frequency and total_frequency > 0:
            pure_average = weighted_total_pure / total_frequency
            speed_average = weighted_total_speed / total_frequency
        else:
            pure_average = np.mean(pure_scores) if pure_scores else 0.0
            speed_average = np.mean(speed_weighted_scores) if speed_weighted_scores else 0.0
        
        results = {
            # Primary scores
            'average_score': speed_average,  # Primary: speed-weighted
            'raw_average_score': pure_average,  # Secondary: pure Dvorak-7
            
            # Detailed breakdown
            'pure_dvorak7_score': pure_average,
            'speed_weighted_score': speed_average,
            'improvement_ratio': speed_average / pure_average if pure_average > 0 else 1.0,
            
            # Layout coverage analysis
            'letters_in_layout': len(layout_mapping),
            'bigrams_scored': len(pure_scores),
            'coverage': len(pure_scores) / len(letter_pairs) if letter_pairs else 0.0,
            
            # Standard fields for compatibility
            'pair_count': len(pure_scores),
            'total_score': weighted_total_speed if use_frequency else sum(speed_weighted_scores),
            'raw_total_score': weighted_total_pure if use_frequency else sum(pure_scores),
        }
        
        if use_frequency:
            results.update({
                'total_frequency': total_frequency,
                'frequency_coverage': total_frequency / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            })
        
        return results
    
    def _score_layout_with_method(self, layout_mapping: Dict[str, str], scorer: str) -> Dict[str, float]:
        """Score a layout using a specific scoring method with NA validation."""
        # Validate layout mapping for NA values
        for letter, key_pos in layout_mapping.items():
            if str(letter) in ['NA', 'NAN', 'nan'] or str(key_pos) in ['NA', 'NAN', 'nan']:
                raise ValueError(f"Invalid mapping detected: '{letter}' -> '{key_pos}'")
        
        # Handle dynamic scoring methods
        if scorer == 'comfort-key':
            return self._score_layout_comfort_key(layout_mapping)
        elif scorer == 'engram':
            return self._score_layout_engram(layout_mapping)
        elif scorer == 'dvorak7-speed':
            return self._score_layout_dvorak7_speed(layout_mapping)
        
        # Handle table-based scoring methods
        score_col = f"{scorer}_score_normalized"
        if score_col not in self.score_table.columns:
            score_col = f"{scorer}_normalized"
            if score_col not in self.score_table.columns:
                raise ValueError(f"Scorer '{scorer}' not found in score table")
        
        # Determine if this scorer should be inverted
        invert_scores = scorer in ['distance', 'time']
        
        # Use actual English bigrams from frequency data
        if self.bigram_frequencies:
            letter_pairs2 = list(self.bigram_frequencies.keys())
        else:
            raise ValueError("Bigram frequencies required for table-based scoring methods")

        # Filter to only letter-pairs where both letters exist in this layout
        letter_pairs = []
        for letter_pair in letter_pairs2:
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    letter_pairs.append(letter_pair)
        
        # Initialize scoring variables
        raw_total_score = 0.0
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        
        for letter_pair in letter_pairs:
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                
                # Validate letters
                if str(letter1) in ['NA', 'NAN', 'nan'] or str(letter2) in ['NA', 'NAN', 'nan']:
                    raise ValueError(f"Invalid letter in pair: '{letter_pair}' ('{letter1}', '{letter2}')")
                
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    key1 = layout_mapping[letter1]
                    key2 = layout_mapping[letter2]
                    
                    # Validate keys individually (not the combined key_pair)
                    if str(key1) in ['NA', 'NAN', 'nan'] or str(key2) in ['NA', 'NAN', 'nan']:
                        raise ValueError(f"Invalid key in mapping: '{letter1}' -> '{key1}', '{letter2}' -> '{key2}'")

                    key_pair = str(key1) + str(key2)  # Force string concatenation

                    # Additional length validation
                    if len(key_pair) != 2:
                        raise ValueError(f"Invalid key pair length: '{key_pair}' from '{letter1}' -> '{key1}', '{letter2}' -> '{key2}'")                    

                    # Look up key-pair score in table
                    if key_pair in self.score_table.index:
                        raw_score = self.score_table.loc[key_pair, score_col]
                    else:
                        raise ValueError(f"Missing score for key pair: '{key_pair}' (from letters '{letter1}{letter2}' -> keys '{key1}{key2}')")

                    # Rest of scoring logic continues...
                    if invert_scores:
                        score = 1.0 - raw_score
                    else:
                        score = raw_score

                    raw_total_score += score
                    raw_count += 1

                    if use_frequency:
                        frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                        weighted_total_score += score * frequency
                        total_frequency += frequency
                        if frequency > 0:
                            frequency_coverage += frequency
                else:
                    raise ValueError(f"Letter pair '{letter_pair}' contains letters not in layout mapping")
        
        # Calculate results
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        frequency_weighted_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0

        letters = list(layout_mapping.keys())
        for letter1 in letters:
            for letter2 in letters:
                if letter1 != letter2:  # Skip self-pairs
                    letter_pair = letter1 + letter2
                    key1 = layout_mapping[letter1]
                    key2 = layout_mapping[letter2]
                    key_pair = key1 + key2
                    
                    # Get comfort score
                    if key_pair in self.score_table.index:
                        comfort_score = self.score_table.loc[key_pair, score_col]
                    else:
                        comfort_score = 1.0
                        raise ValueError(f"Missing comfort score for key pair: {key_pair}")
                    
                    # Invert if needed
                    if invert_scores:
                        comfort_score = 1.0 - comfort_score
                    
                    # Check if bigram_frequencies is None before calling .get()
                    if self.bigram_frequencies is not None:
                        frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                    else:
                        frequency = 1.0  # Default frequency for raw mode

        # Determine primary score based on mode  
        primary_score = frequency_weighted_score
        results = {
            'average_score': frequency_weighted_score if use_frequency else raw_average,
            'raw_average_score': raw_average,
            'total_score': weighted_total_score if use_frequency else raw_total_score,
            'raw_total_score': raw_total_score,
            'pair_count': raw_count,
            'coverage': raw_count / len(letter_pairs2) if letter_pairs2 else 0.0,
        }
        
        if use_frequency:
            results.update({
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            })
        
        return results

    def _score_layout_comfort_key(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score a layout using comfort-key method."""
        if self.letter_frequencies is None or self.key_comfort_scores is None:
            missing_files = []
            if self.letter_frequencies is None:
                missing_files.append("input/english-letter-frequencies-google-ngrams.csv")
            if self.key_comfort_scores is None:
                missing_files.append("tables/key_scores.csv")
            raise ValueError(
                f"Letter frequencies and key comfort scores required for comfort-key scoring. "
                f"Missing files: {missing_files}. "
                f"Run 'python prep_scoring_tables.py --input-dir tables/' to create key_scores.csv"
            )
        
        # Use actual English bigrams from frequency data
        if self.bigram_frequencies:
            letter_pairs2 = list(self.bigram_frequencies.keys())
        else:
            raise ValueError("Bigram frequencies required for table-based scoring methods")
                
        # Initialize scoring variables
        raw_total_score = 0.0
        raw_count = 0
        
        # Frequency-weighted scoring (if not using raw mode)
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        
        for letter_pair in letter_pairs2:
            comfort_key_score = self._compute_comfort_key_score(letter_pair, layout_mapping)
            
            if comfort_key_score is not None:
                # Raw scoring (treat all letter-pairs equally)
                raw_total_score += comfort_key_score
                raw_count += 1
                
                # Frequency-weighted scoring (if enabled)
                if use_frequency:
                    frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                    weighted_total_score += comfort_key_score * frequency
                    total_frequency += frequency
                    if frequency > 0:
                        frequency_coverage += frequency
            else:
                raise ValueError(f"Comfort-key score not available for letter pair: {letter_pair}")
        
        # Calculate results
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        
        # Determine primary score based on mode
        if use_frequency:
            # Frequency weighting is primary
            primary_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
            results = {
                'average_score': primary_score,  # Primary score
                'total_score': weighted_total_score,
                'raw_average_score': raw_average,  # Secondary 
                'raw_total_score': raw_total_score,
                'pair_count': raw_count,
                'coverage': raw_count / len(letter_pairs2) if letter_pairs2 else 0.0,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            }
        else:
            # Raw scoring is primary
            results = {
                'average_score': raw_average,  # Primary score
                'total_score': raw_total_score,
                'pair_count': raw_count,
                'coverage': raw_count / len(letter_pairs2) if letter_pairs2 else 0.0
            }
        
        return results
    
    def _score_layout_engram(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score a layout using engram method (comfort * comfort-key)."""
        if self.letter_frequencies is None or self.key_comfort_scores is None:
            missing_files = []
            if self.letter_frequencies is None:
                missing_files.append("input/english-letter-frequencies-google-ngrams.csv")
            if self.key_comfort_scores is None:
                missing_files.append("tables/key_scores.csv")
            raise ValueError(
                f"Letter frequencies and key comfort scores required for engram scoring. "
                f"Missing files: {missing_files}. "
                f"Run 'python prep_scoring_tables.py --input-dir tables/' to create key_scores.csv"
            )
        
        # Check if comfort scoring is available
        comfort_col = "comfort_score_normalized"
        if comfort_col not in self.score_table.columns:
            comfort_col = "comfort_normalized"
            if comfort_col not in self.score_table.columns:
                raise ValueError("Comfort scores not found in score table - required for engram scoring")
        
        # Use actual English bigrams from frequency data
        if self.bigram_frequencies:
            letter_pairs2 = list(self.bigram_frequencies.keys())
        else:
            raise ValueError("Bigram frequencies required for table-based scoring methods")
        
        # Initialize scoring variables
        raw_total_score = 0.0
        raw_count = 0
        
        # Frequency-weighted scoring (if not using raw mode)
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        
        for letter_pair in letter_pairs2:
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    key1 = layout_mapping[letter1]
                    key2 = layout_mapping[letter2]
                    key_pair = key1 + key2
                    
                    # Get comfort score from table
                    if key_pair in self.score_table.index:
                        comfort_score = self.score_table.loc[key_pair, comfort_col]
                        
                        # Get comfort-key score
                        comfort_key_score = self._compute_comfort_key_score(letter_pair, layout_mapping)
                        
                        if comfort_key_score is not None:
                            # Compute engram score
                            engram_score = comfort_score * comfort_key_score
                            
                            # Raw scoring (treat all letter-pairs equally)
                            raw_total_score += engram_score
                            raw_count += 1
                            
                            # Frequency-weighted scoring (if enabled)
                            if use_frequency:
                                frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                                weighted_total_score += engram_score * frequency
                                total_frequency += frequency
                                if frequency > 0:
                                    frequency_coverage += frequency
                        else:
                            raise ValueError(f"Comfort-key score not available for letter pair: {letter_pair}")
                    else:
                        raise ValueError(f"Comfort score not available for key pair: {key_pair}")
                else:
                    # Letter not in layout mapping
                    raise ValueError(f"Letter pair '{letter_pair}' contains letters not in layout mapping: {letter1}, {letter2}")

        # Calculate results
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        
        # Determine primary score based on mode
        if use_frequency:
            # Frequency weighting is primary
            primary_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
            results = {
                'average_score': primary_score,  # Primary score
                'total_score': weighted_total_score,
                'raw_average_score': raw_average,  # Secondary 
                'raw_total_score': raw_total_score,
                'pair_count': raw_count,
                'coverage': raw_count / len(letter_pairs2) if letter_pairs2 else 0.0,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            }
        else:
            # Raw scoring is primary
            results = {
                'average_score': raw_average,  # Primary score
                'total_score': raw_total_score,
                'pair_count': raw_count,
                'coverage': raw_count / len(letter_pairs2) if letter_pairs2 else 0.0
            }
        
        return results
    
    def score_layout(self, layout_mapping: Dict[str, str], scorers: List[str]) -> Dict[str, Dict[str, float]]:
        """Score a layout using specified scoring methods."""
        
        results = {}
        
        for scorer in scorers:
            if scorer not in self.available_scorers:
                print(f"Warning: Scorer '{scorer}' not available. Available: {self.available_scorers}")
                continue
            
            try:
                results[scorer] = self._score_layout_with_method(layout_mapping, scorer)
            except Exception as e:
                print(f"Error scoring with {scorer}: {e}")
        
        return results
    
    def compare_layouts(self, layouts: Dict[str, Dict[str, str]], scorers: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare multiple layouts across scoring methods."""
        
        results = {}
        
        for layout_name, layout_mapping in layouts.items():
            if self.verbose:
                print(f"Scoring layout: {layout_name}")
            
            results[layout_name] = self.score_layout(layout_mapping, scorers)
        
        return results

def parse_layout_string(layout_str: str) -> Dict[str, str]:
    """Parse layout string into character -> position mapping."""
    # Handle format like "name:layout_string"
    if ':' in layout_str:
        name, layout = layout_str.split(':', 1)
        return layout.strip()
    else:
        return layout_str.strip()

def parse_layout_compare(compare_args: List[str]) -> Dict[str, Dict[str, str]]:
    """Parse layout comparison arguments."""
    layouts = {}
    
    for arg in compare_args:
        if ':' not in arg:
            raise ValueError(f"Layout comparison format should be 'name:layout'. Got: {arg}")
        
        name, layout_str = arg.split(':', 1)
        name = str(name).strip()  # Force to string
        layout_str = str(layout_str).strip()  # Force to string
        
        # Create mapping from layout string to QWERTY positions
        qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
        
        if len(layout_str) > len(qwerty_positions):
            raise ValueError(f"Layout '{name}' too long. Max {len(qwerty_positions)} characters.")
        
        mapping = {}
        for i, char in enumerate(layout_str):
            # CRITICAL: Force everything to string and handle apostrophe specially
            char_str = str(char)
            char_upper = char_str.upper()
            pos_str = str(qwerty_positions[i])
            
            # Special handling for apostrophe to prevent NaN conversion
            if char_str == "'":
                char_upper = "'"  # Keep as literal apostrophe
            
            # Debug suspicious conversions
            if char_upper in ['NAN', 'NA', 'nan'] or pos_str in ['NAN', 'NA', 'nan']:
                raise ValueError(f"Suspicious character conversion detected: '{char}' -> '{char_upper}' -> '{pos_str}'")
            
            mapping[char_upper] = pos_str
        
        layouts[name] = mapping
    
    return layouts

def create_layout_mapping(letters: str, positions: str) -> Dict[str, str]:
    """Create mapping from letters to QWERTY positions."""
    if len(letters) != len(positions):
        raise ValueError(f"Letters ({len(letters)}) and positions ({len(positions)}) must have same length")
    
    mapping = {}
    for letter, position in zip(letters, positions):
        # CRITICAL: Ensure both stay as strings
        letter_key = str(letter).upper()
        pos_value = str(position).upper()
        
        # Debug: Check for problematic characters
        if letter_key in ['NAN', 'NA', ''] or pos_value in ['NAN', 'NA', '']:
            print(f"Warning: Problematic character mapping: '{letter}' -> '{letter_key}' -> '{pos_value}'")
        
        mapping[letter_key] = pos_value
    
    return mapping

def print_results(results: Dict[str, float], format_type: str = 'detailed', scorer_name: str = '', use_raw: bool = False, verbose: bool = False):
    """Print scoring results with Dvorak-7 speed analysis."""
    
    if format_type == 'csv_output':
        # Minimal CSV output for programmatic use
        if use_raw or 'raw_average_score' not in results:
            print(f"{results['average_score']:.6f}")
        return
    
    if format_type == 'score_only':
        print(f"{results['average_score']:.6f}")
        return
    
    if format_type == 'csv':
        # CSV output for dvorak7-speed
        if scorer_name == 'dvorak7-speed':
            if use_raw:
                print("scorer,speed_weighted_score,pure_dvorak7_score,improvement_ratio,letters_in_layout,bigrams_scored")
                print(f"{scorer_name},{results['speed_weighted_score']:.6f},{results['pure_dvorak7_score']:.6f},"
                      f"{results['improvement_ratio']:.6f},{results['letters_in_layout']},{results['bigrams_scored']}")
            else:
                print("scorer,speed_weighted_score,pure_dvorak7_score,improvement_ratio,letters_in_layout,bigrams_scored,coverage,frequency_coverage")
                print(f"{scorer_name},{results['speed_weighted_score']:.6f},{results['pure_dvorak7_score']:.6f},"
                      f"{results['improvement_ratio']:.6f},{results['letters_in_layout']},{results['bigrams_scored']},"
                      f"{results['coverage']:.6f},{results.get('frequency_coverage', 0.0):.6f}")
        else:
            # Standard CSV output for other scorers
            if use_raw:
                print("scorer,average_score,total_score,pair_count,coverage")
                print(f"{scorer_name},{results['average_score']:.6f},{results['total_score']:.6f},"
                      f"{results['pair_count']},{results['coverage']:.6f}")
            else:
                print("scorer,average_score,total_score,raw_average_score,raw_total_score,pair_count,coverage,frequency_coverage")
                print(f"{scorer_name},{results['average_score']:.6f},{results['total_score']:.6f},"
                    f"{results['raw_average_score']:.6f},{results['raw_total_score']:.6f},"
                    f"{results['pair_count']},{results['coverage']:.6f},{results['frequency_coverage']:.6f}")
        return
    
    # Detailed format
    if scorer_name == 'dvorak7-speed':
        print("=" * 60)
        print("EMPIRICAL DVORAK-7 SPEED ANALYSIS")
        print("=" * 60)
        
        print(f"Speed-weighted score:     {results['speed_weighted_score']:.6f}")
        print(f"Pure Dvorak-7 score:      {results['pure_dvorak7_score']:.6f}")
        print(f"Improvement ratio:        {results['improvement_ratio']:.3f}x")
        
        improvement = (results['improvement_ratio'] - 1) * 100
        if improvement > 0:
            print(f"→ Speed weighting improves score by {improvement:.1f}%")
        elif improvement < 0:
            print(f"→ Speed weighting reduces score by {abs(improvement):.1f}%")
        else:
            print(f"→ Speed weighting has no net effect")
        
        print(f"\nLayout Analysis:")
        print(f"  Letters in layout:      {results['letters_in_layout']}")
        print(f"  Bigrams scored:         {results['bigrams_scored']}")
        print(f"  Coverage:               {results['coverage']:.1%}")
        
        if 'frequency_coverage' in results:
            print(f"  Frequency coverage:     {results['frequency_coverage']:.1%}")
        
        print(f"\nBased on 19.4M empirical bigrams")
        print(f"Weights from actual typing speed correlations")
        print(f"Using full 32-key layout weights")
    else:
        # Standard detailed output for other scorers
        if use_raw:
            print(f"Average bigram score: {results['average_score']:.6f}")
            print(f"Total score: {results['total_score']:.6f}")
        else:
            print(f"Frequency-weighted average bigram score: {results['average_score']:.6f}")
            
            # Show raw scores if verbose
            if verbose:
                print(f"Raw average bigram score: {results['raw_average_score']:.6f}")
    
    print(f"Pair count: {results['pair_count']}")
    print(f"Coverage (% letter-pairs with precomputed scores): {results['coverage']:.1%}")
    
    if not use_raw and 'frequency_coverage' in results:
        print(f"Frequency coverage (% English frequency that layout covers): {results['frequency_coverage']:.1%}")
    
def print_comparison_summary(comparison_results, format_type='detailed', quiet=False, use_raw=False, verbose=False):
    """Print summary with apostrophe-safe CSV output."""
    if format_type == 'csv_output':
        print("layout_name,scorer,weighted_score,raw_score")
        
        for layout_name, layout_results in comparison_results.items():
            for scorer, results in layout_results.items():
                # Ensure strings are properly escaped
                safe_layout_name = str(layout_name).replace('"', '""')  # Escape quotes
                safe_scorer = str(scorer).replace('"', '""')
                
                weighted_score = results['average_score']
                raw_score = results.get('raw_average_score', results['average_score'])
                
                # Use quoted strings to preserve special characters
                print(f'"{safe_layout_name}","{safe_scorer}",{weighted_score:.6f},{raw_score:.6f}')
        return
    
def save_detailed_comparison_csv(comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
                               filename: str, layout_mappings: Dict[str, Dict[str, str]] = None, 
                               use_raw: bool = False):
    """Save detailed comparison results to CSV."""
    
    rows = []
    for layout_name, layout_results in comparison_results.items():
        for scorer, results in layout_results.items():
            row = {
                'layout_name': layout_name,
                'scorer': scorer,
                'average_score': results['average_score'],
                'total_score': results['total_score'],
                'pair_count': results['pair_count'],
                'coverage': results['coverage']
            }
            
            # Add frequency-weighted vs raw details
            if not use_raw and 'raw_average_score' in results:
                row.update({
                    'raw_average_score': results['raw_average_score'],
                    'raw_total_score': results['raw_total_score'],
                    'total_frequency': results.get('total_frequency', 0),
                    'frequency_coverage': results.get('frequency_coverage', 0.0)
                })
            
            # Add layout mapping if available
            if layout_mappings and layout_name in layout_mappings:
                mapping = layout_mappings[layout_name]
                row['layout_letters'] = ''.join(sorted(mapping.keys()))
                row['layout_positions'] = ''.join(mapping[c] for c in sorted(mapping.keys()))
            
            rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Format floating point numbers
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, float):
                        formatted_row[key] = f"{value:.6f}"
                    else:
                        formatted_row[key] = value
                writer.writerow(formatted_row)

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Unified keyboard layout scorer with automatic defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage (uses default files: tables/keypair_scores.csv and input/english-letter-pair-frequencies-google-ngrams.csv)
  # Note: Run 'python prep_scoring_tables.py --input-dir tables/' first to create required tables
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
  
  # Raw (unweighted) scoring only
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --raw
  
  # Compare layouts with default frequency weighting
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl"
  
  # Pure vs Speed-weighted Dvorak-7 comparison
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --scorer dvorak7-speed
  
  # Both pure and speed-weighted Dvorak-7
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --scorers dvorak7,dvorak7-speed
  
  # Use custom score table and frequency file
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-table custom_scores.csv --frequency-file custom_freqs.csv
  
  # Minimal CSV output for programmatic use (no headers, scores only)
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output
  
  # Verbose output (shows both weighted and raw scores)
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --verbose
  
  # Save to CSV file with frequency weighting
  python score_layouts.py --compare qwerty:"qwerty" dvorak:"',.py" --csv results.csv

Default behavior:
- Uses tables/keypair_scores.csv for key-pair scoring data (created by prep_scoring_tables.py)
- Uses tables/key_scores.csv for individual key comfort scores (created by prep_scoring_tables.py)
- Uses input/english-letter-pair-frequencies-google-ngrams.csv for frequency weighting (if it exists)
- Uses input/english-letter-frequencies-google-ngrams.csv for letter frequencies (if it exists)
- Uses input/dvorak7_speed_weights.csv for empirical Dvorak-7 speed scoring (if it exists)
- Falls back to raw scoring if frequency file is not found
- With --raw: Ignores frequencies and uses raw (unweighted) scoring
- With --verbose: Shows both weighted and raw scores for comparison
- With --csv-output: Minimal CSV format for programmatic use (layout,scorer,weighted_score,raw_score)

Available scoring methods depend on the score table contents (e.g., distance, comfort, dvorak7, time).
Distance and time scores are automatically inverted (1-score) since higher values are worse.
Engram and comfort-key scores are computed dynamically and require letter frequencies and key comfort scores.
dvorak7-speed provides both pure and empirically-weighted Dvorak-7 scores based on 19.4M typing correlations.
        """
    )
    
    # Required arguments (now optional with defaults)
    parser.add_argument(
        '--score-table',
        default="tables/keypair_scores.csv",
        help="Path to unified score table CSV file (default: tables/keypair_scores.csv)"
    )
    
    # Optional frequency weighting (with default)
    parser.add_argument(
        '--frequency-file',
        default="input/english-letter-pair-frequencies-google-ngrams.csv",
        help="Path to bigram frequency CSV file (default: input/english-letter-pair-frequencies-google-ngrams.csv)"
    )
    
    # Raw scoring option
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Use raw (unweighted) scoring only, ignore frequency weighting"
    )
    
    # Minimal CSV output option
    parser.add_argument(
        '--csv-output',
        action='store_true',
        help="Output minimal CSV format to stdout (scores only, no headers, for programmatic use)"
    )
    
    # Scorer selection (mutually exclusive)
    scorer_group = parser.add_mutually_exclusive_group()
    scorer_group.add_argument(
        '--scorer',
        help="Run a single scorer (available scorers depend on table contents)"
    )
    scorer_group.add_argument(
        '--scorers',
        help="Run multiple scorers (comma-separated or 'all')"
    )
    scorer_group.add_argument(
        '--compare',
        nargs='+',
        help="Compare layouts (e.g., qwerty:qwertyuiop dvorak:',.pyfgcrl)"
    )
    
    # Layout definition (for single layout scoring)
    layout_group = parser.add_argument_group('Layout Definition')
    layout_group.add_argument(
        '--letters',
        help="String of characters in the layout"
    )
    layout_group.add_argument(
        '--positions', 
        help="String of corresponding QWERTY positions"
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--format',
        choices=['detailed', 'csv', 'score_only', 'table'],
        default='table',
        help="Output format (default: table)"
    )
    output_group.add_argument(
        '--csv',
        help="Save detailed comparison to CSV file"
    )
    output_group.add_argument(
        '--score-only',
        action='store_true',
        help="Output only primary scores (minimal output)"
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress verbose output"
    )
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed processing information"
    )
    
    return parser

def main() -> int:
    """Main entry point."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Handle output format shortcuts
    if args.score_only:
        args.format = 'score_only'
    elif args.csv_output:
        args.format = 'csv_output'
    
    try:
        # Always load frequency file for letter-pairs (weighting controlled by --raw flag)
        frequency_file = None
        if Path(args.frequency_file).exists():
            frequency_file = args.frequency_file
        elif args.frequency_file != "input/english-letter-pair-frequencies-google-ngrams.csv":
            # User explicitly provided a frequency file that doesn't exist
            print(f"Error: Frequency file not found: {args.frequency_file}")
            return 1
        # If using default frequency file and it doesn't exist, just use raw scoring
        
        # Initialize scorer (suppress verbose output if using csv-output)
        suppress_verbose = args.format == 'csv_output'
        scorer = LayoutScorer(args.score_table, frequency_file, args.raw, 
                             verbose=args.verbose and not suppress_verbose)
        
        if args.compare:
            # Layout comparison mode
            layouts = parse_layout_compare(args.compare)
            
            # Determine which scorers to run
            if args.scorer:
                scorers = [args.scorer] if args.scorer in scorer.available_scorers else []
            elif args.scorers:
                if args.scorers.lower() == 'all':
                    scorers = scorer.available_scorers
                else:
                    requested = [s.strip() for s in args.scorers.split(',')]
                    scorers = [s for s in requested if s in scorer.available_scorers]
                    invalid = [s for s in requested if s not in scorer.available_scorers]
                    if invalid:
                        print(f"Warning: Unknown scorers ignored: {invalid}")
            else:
                scorers = scorer.available_scorers  # Default to all
            
            if not scorers:
                print("Error: No valid scorers specified")
                return 1
            
            if not args.quiet and args.format != 'csv_output':
                print(f"Comparing {len(layouts)} layouts using {len(scorers)} scoring methods...")
                print(f"Layouts: {', '.join(layouts.keys())}")
                print(f"Scorers: {', '.join(scorers)}")
                if args.raw:
                    print("Using raw (unweighted) scoring")
                elif scorer.bigram_frequencies:
                    print("Using frequency-weighted scoring")
                else:
                    print("Using raw scoring (frequency file not found)")
            
            # Run comparison
            results = scorer.compare_layouts(layouts, scorers)
            
            # Handle output
            if args.csv:
                save_detailed_comparison_csv(results, args.csv, layouts, args.raw)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Detailed comparison saved to: {args.csv}")
            else:
                if not args.quiet and args.format != 'csv_output':
                    print(f"\n=== RESULTS ===")
                print_comparison_summary(results, args.format, args.quiet, args.raw, args.verbose)

        else:
            # Single layout mode
            if not args.letters or not args.positions:
                print("Error: Must specify --letters and --positions for single layout scoring")
                return 1
            
            # Create layout mapping
            try:
                layout_mapping = create_layout_mapping(args.letters, args.positions)
            except Exception as e:
                print(f"Error creating layout mapping: {e}")
                return 1
            
            # Determine which scorers to run
            if args.scorer:
                scorers = [args.scorer] if args.scorer in scorer.available_scorers else []
            elif args.scorers:
                if args.scorers.lower() == 'all':
                    scorers = scorer.available_scorers
                else:
                    requested = [s.strip() for s in args.scorers.split(',')]
                    scorers = [s for s in requested if s in scorer.available_scorers]
                    invalid = [s for s in requested if s not in scorer.available_scorers]
                    if invalid:
                        print(f"Warning: Unknown scorers ignored: {invalid}")
            else:
                scorers = scorer.available_scorers  # Default to all
            
            if not scorers:
                print("Error: No valid scorers specified")
                return 1
            
            # Create layout name for display
            layout_name = f"{args.letters} → {args.positions}"
            
            # Run scoring
            results = scorer.score_layout(layout_mapping, scorers)
            
            # Handle output
            if args.csv:
                # Convert to comparison format
                comparison_results = {layout_name: results}
                layout_mappings_for_csv = {layout_name: layout_mapping}
                save_detailed_comparison_csv(comparison_results, args.csv, layout_mappings_for_csv, args.raw)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Results saved to: {args.csv}")
            else:
                # Print to stdout
                if len(scorers) == 1:
                    # Single scorer
                    scorer_name = scorers[0]
                    
                    if not args.quiet and args.format != 'csv_output':
                        print(f"\nLayout: {layout_name}")
                        print(f"\n{scorer_name.upper()} results:")
                        print("=" * 50)
                    print_results(results[scorer_name], args.format, scorer_name, args.raw, args.verbose)
                else:
                    # Multiple scorers
                    comparison_results = {layout_name: results}
                    print_comparison_summary(comparison_results, args.format, args.quiet, args.raw, args.verbose)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if not Path(args.score_table).exists():
            print(f"Score table file not found: {args.score_table}")
            print("Make sure the file exists or specify a different path with --score-table")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())