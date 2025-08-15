#!/usr/bin/env python3
"""
Keyboard Layout Scorer using precomputed score table.

A comprehensive tool for evaluating keyboard layouts using frequency-weighted scoring.
Scoring methods include engram, comfort, comfort-key, dvorak7, and distance.
(Experimental scores related to time, including dvorak7-speed, should be ignored
since they are heavily biased by practice effects.)
Distance (and time) scores are automatically inverted and renamed to efficiency (and speed).

Setup:
1. Generate individual score files (keypair_*_scores.csv) using your scoring scripts
2. Run: python prep_scoring_tables.py --input-dir tables/
   This creates: tables/keypair_scores_detailed.csv and tables/key_scores.csv
3. Run this script to score layouts using all available methods

Default behavior:
- Score table: tables/keypair_scores_detailed.csv (created by prep_scoring_tables.py)
- Key scores: tables/key_scores.csv (created by prep_scoring_tables.py)  
- Frequency data: input/english-letter-pair-frequencies-google-ngrams.csv
- Letter frequencies: input/english-letter-frequencies-google-ngrams.csv
- Scoring mode: Frequency-weighted (prioritizes common English letter combinations)
- Score mapping: Letter-pair frequencies → Key-pair scores (distance/time inverted and renamed)

Usage:
    # Single layout evaluation (all available scoring methods)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
    
    # Force raw (unweighted) scoring
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --raw
    
    # Compare multiple layouts  
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;"
    
    # Both pure and speed-weighted Dvorak-7
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --scorers dvorak7,dvorak7-speed
    
    # Verbose output (shows both weighted and raw scores)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --verbose
    
    # Minimal CSV output for scripts/automation
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv-output
    
    # Save detailed results to file
    python score_layouts.py --compare qwerty:"qwerty" dvorak:"dvorak" --csv results.csv
    
    # Custom files (when defaults aren't suitable)
    python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-table custom.csv --frequency-file custom_freq.csv

Key features:
- Automatic frequency weighting using English bigram frequencies
- Letter-pair → Key-pair mapping (e.g., "TH" frequency weights T→H key transition)
- Multiple output formats: detailed, CSV, minimal CSV, score-only
- Graceful fallback to raw scoring if frequency file missing
- Support for all scoring methods in the unified score table
- Dynamic engram and comfort-key scoring (requires prep_scoring_tables.py output)
"""

import sys
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Import for dvorak7-speed scoring
try:
    sys.path.insert(0, str(Path(__file__).parent / "prep"))
    from prep_keypair_dvorak7_scores import score_bigram_dvorak7
    DVORAK7_AVAILABLE = True
except ImportError:
    DVORAK7_AVAILABLE = False

class LayoutScorer:
    """Unified layout scorer using pre-computed score table."""
    
    def __init__(self, score_table_path: str, frequency_file: Optional[str] = None, use_raw: bool = False, verbose: bool = False):
        """Initialize scorer with score table and optional frequency data."""
        self.verbose = verbose
        self.use_raw = use_raw
        self.score_table = self._load_score_table(score_table_path)
        self.available_scorers = self._detect_available_scorers()
        
        self.bigram_frequencies = None
        if frequency_file:
            self.bigram_frequencies = self._load_frequency_data(frequency_file)
        
        self.letter_frequencies = self._load_letter_frequencies()
        self.key_comfort_scores = self._load_key_comfort_scores()
        self.dvorak7_speed_weights = self._load_dvorak7_speed_weights()
        
        self._update_available_scorers()
        
        if self.verbose:
            self._print_initialization_info(score_table_path, frequency_file)
    
    def _load_score_table(self, filepath: str) -> pd.DataFrame:
        """Load the unified score table with careful NA handling."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Score table not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath, 
                           dtype={'key_pair': 'str'},
                           keep_default_na=False,
                           na_values=['', 'NULL', 'null', 'NaN', 'nan'])
        except Exception as e:
            raise ValueError(f"Error reading score table: {e}")
        
        if 'key_pair' not in df.columns:
            raise ValueError("Score table must have 'key_pair' column")
        
        missing_count = df['key_pair'].isna().sum()
        if missing_count > 0:
            df = df.dropna(subset=['key_pair'])
            if self.verbose:
                print(f"Removed {missing_count} rows with missing key_pair values")
        
        empty_count = (df['key_pair'].astype(str).str.strip() == '').sum()
        if empty_count > 0:
            df = df[df['key_pair'].astype(str).str.strip() != '']
            if self.verbose:
                print(f"Removed {empty_count} rows with empty key_pair values")
        
        if self.verbose and 'NA' in df['key_pair'].values:
            print("'NA' key pair preserved in score table")
        
        return df.set_index('key_pair')

    def _load_frequency_data(self, filepath: str) -> Optional[Dict[str, float]]:
        """Load bigram frequency data from CSV file."""
        if not Path(filepath).exists():
            if self.verbose:
                print(f"Frequency file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            if self.verbose:
                print(f"Error reading frequency file: {e}")
            return None
        
        bigram_col = self._find_column(df, ['bigram', 'pair', 'key_pair', 'letter_pair'])
        freq_col = self._find_column(df, ['normalized_frequency', 'frequency'])
        
        if not bigram_col or not freq_col:
            if self.verbose:
                print("Could not find required columns in frequency file")
            return None
        
        frequencies = {}
        for _, row in df.iterrows():
            bigram = str(row[bigram_col]).strip().upper()
            freq = float(row[freq_col])
            
            if len(bigram) == 2:
                frequencies[bigram] = freq
        
        if self.verbose:
            total_freq = sum(frequencies.values())
            print(f"Loaded {len(frequencies)} bigram frequencies (total: {total_freq:,})")
        
        return frequencies
    
    def _load_letter_frequencies(self) -> Optional[Dict[str, float]]:
        """Load individual letter frequency data."""
        filepath = "input/english-letter-frequencies-google-ngrams.csv"
        
        if not Path(filepath).exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            letter_col = self._find_column(df, ['letter', 'char', 'character'])
            freq_col = self._find_column(df, ['normalized_frequency', 'frequency'])
            
            if not letter_col or not freq_col:
                return None
            
            frequencies = {}
            for _, row in df.iterrows():
                letter = str(row[letter_col]).strip().upper()
                freq = float(row[freq_col])
                
                if len(letter) == 1:
                    frequencies[letter] = freq
            
            if self.verbose:
                print(f"Loaded letter frequencies for {len(frequencies)} letters")
            
            return frequencies
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading letter frequencies: {e}")
            return None
    
    def _load_key_comfort_scores(self) -> Optional[Dict[str, float]]:
        """Load key comfort scores from tables/key_scores.csv."""
        filepath = "tables/key_scores.csv"
        
        if not Path(filepath).exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            if 'key' not in df.columns or 'comfort_score' not in df.columns:
                return None
            
            scores = {}
            for _, row in df.iterrows():
                key = str(row['key']).strip().upper()
                score = float(row['comfort_score'])
                scores[key] = score
            
            if self.verbose:
                print(f"Loaded key comfort scores for {len(scores)} keys")
            
            return scores
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading key comfort scores: {e}")
            return None
    
    def _load_dvorak7_speed_weights(self) -> Optional[Dict]:
        """Load Dvorak-7 empirical speed weights."""
        filepath = "input/dvorak7_speed_weights.csv"
        
        if not Path(filepath).exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            required_cols = ['combination', 'k_way', 'correlation', 'significant_after_fdr']
            for col in required_cols:
                if col not in df.columns:
                    return None
            
            significant_df = df[df['significant_after_fdr'] == True]
            
            if len(significant_df) == 0:
                return None
            
            weights = {}
            for _, row in significant_df.iterrows():
                combination = str(row['combination'])
                correlation = float(row['correlation'])
                weights[combination] = correlation
            
            if self.verbose:
                print(f"Loaded Dvorak-7 speed weights: {len(weights)} combinations")
            
            return {
                'combinations': significant_df,
                'weights': weights,
                'best_combination': significant_df.iloc[0] if len(significant_df) > 0 else None
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading Dvorak-7 speed weights: {e}")
            return None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _detect_available_scorers(self) -> List[str]:
        """Detect available scoring methods from table columns."""
        scorers = []
        
        for col in self.score_table.columns:
            if col.endswith('_normalized'):
                scorer_name = col.replace('_normalized', '').replace('_score', '')
                scorers.append(scorer_name)
        
        return sorted(list(set(scorers)))
    
    def _update_available_scorers(self):
        """Add dynamic scorers to available list."""
        if self.letter_frequencies and self.key_comfort_scores:
            if 'comfort-key' not in self.available_scorers:
                self.available_scorers.append('comfort-key')
            if 'comfort' in self.available_scorers and 'engram' not in self.available_scorers:
                self.available_scorers.append('engram')
        
        if self.dvorak7_speed_weights and DVORAK7_AVAILABLE:
            if 'dvorak7-speed' not in self.available_scorers:
                self.available_scorers.append('dvorak7-speed')
    
    def _print_initialization_info(self, score_table_path: str, frequency_file: Optional[str]):
        """Print initialization information if verbose."""
        print(f"Score table: {score_table_path}")
        if frequency_file:
            print(f"Frequency file: {frequency_file}")
        print(f"Loaded score table with {len(self.score_table)} key pairs")
        print(f"Available scorers: {', '.join(self.available_scorers)}")
        
        if self.use_raw:
            print("Using raw (unweighted) scoring only")
        elif self.bigram_frequencies:
            total_freq = sum(self.bigram_frequencies.values())
            print(f"Loaded frequency data for {len(self.bigram_frequencies)} bigrams")
            print(f"Total frequency count: {total_freq:,}")
            print("Using frequency-weighted scoring by default")
        else:
            print("No frequency file found - using raw scoring")
    
    def _get_display_name(self, scorer: str, was_inverted: bool) -> str:
        """Get user-friendly display name for scorer after inversion."""
        if not was_inverted:
            return scorer
        
        # Rename inverted metrics to reflect efficiency interpretation
        if scorer.startswith('distance_'):
            return scorer.replace('distance_', 'efficiency_')
        elif scorer.startswith('time_'):
            return scorer.replace('time_', 'speed_')
        elif scorer == 'distance':
            return 'efficiency'
        elif scorer == 'time':
            return 'speed'
        
        return scorer
    
    def _should_invert_scores(self, scorer: str) -> bool:
        """Determine if a scorer's values should be inverted."""
        scorer_lower = scorer.lower()
        return (
            scorer in ['distance', 'time'] or
            'distance' in scorer_lower or
            'time' in scorer_lower or
            scorer_lower.endswith('_dist') or
            scorer_lower.endswith('_time')
        )
    
    def _validate_layout_mapping(self, layout_mapping: Dict[str, str]):
        """Validate layout mapping for problematic values."""
        for letter, key_pos in layout_mapping.items():
            letter_str = str(letter)
            key_str = str(key_pos)
            
            if letter_str.upper() in ['NA', 'NAN'] or key_str.upper() in ['NA', 'NAN']:
                raise ValueError(f"Invalid mapping detected: '{letter}' -> '{key_pos}'")
    
    def _get_valid_letter_pairs(self, layout_mapping: Dict[str, str]) -> List[str]:
        """Get letter pairs that can be scored with the current layout."""
        if not self.bigram_frequencies:
            return []
        
        valid_pairs = []
        for letter_pair in self.bigram_frequencies.keys():
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    valid_pairs.append(letter_pair)
        
        return valid_pairs
    
    def _compute_comfort_key_score(self, letter_pair: str, layout_mapping: Dict[str, str]) -> Optional[float]:
        """Compute comfort-key score for a letter pair."""
        if len(letter_pair) != 2 or not self.letter_frequencies or not self.key_comfort_scores:
            return None
        
        letter1, letter2 = letter_pair[0], letter_pair[1]
        
        if letter1 not in layout_mapping or letter2 not in layout_mapping:
            return None
        
        key1 = layout_mapping[letter1]
        key2 = layout_mapping[letter2]
        
        freq1 = self.letter_frequencies.get(letter1, 0.0)
        freq2 = self.letter_frequencies.get(letter2, 0.0)
        
        comfort1 = self.key_comfort_scores.get(key1, 0.0)
        comfort2 = self.key_comfort_scores.get(key2, 0.0)
        
        if freq1 + freq2 > 0:
            return (comfort1 * freq1 + comfort2 * freq2) / (freq1 + freq2)
        else:
            return (comfort1 + comfort2) / 2.0
    
    def _score_layout_with_method(self, layout_mapping: Dict[str, str], scorer: str) -> Dict[str, float]:
        """Score a layout using a specific scoring method."""
        self._validate_layout_mapping(layout_mapping)
        
        if scorer == 'comfort-key':
            return self._score_layout_comfort_key(layout_mapping)
        elif scorer == 'engram':
            return self._score_layout_engram(layout_mapping)
        elif scorer == 'dvorak7-speed':
            return self._score_layout_dvorak7_speed(layout_mapping)
        
        score_col = f"{scorer}_score_normalized"
        if score_col not in self.score_table.columns:
            score_col = f"{scorer}_normalized"
            if score_col not in self.score_table.columns:
                raise ValueError(f"Scorer '{scorer}' not found in score table")
        
        invert_scores = self._should_invert_scores(scorer)
        if self.verbose and invert_scores:
            print(f"Inverting scores for '{scorer}' (distance/time-based metric)")
        
        if not self.bigram_frequencies:
            raise ValueError("Bigram frequencies required for table-based scoring methods")
        
        valid_letter_pairs = self._get_valid_letter_pairs(layout_mapping)
        
        if self.verbose:
            print(f"Processing {len(valid_letter_pairs)} valid pairs out of {len(self.bigram_frequencies)} total")
        
        raw_total_score = 0.0
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        
        # Track both original and processed scores
        raw_total_score = 0.0  # After any inversion
        original_raw_total = 0.0  # Before inversion
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        
        for letter_pair in valid_letter_pairs:
            letter1, letter2 = letter_pair[0], letter_pair[1]
            
            key1 = layout_mapping[letter1]
            key2 = layout_mapping[letter2]
            key_pair = str(key1) + str(key2)
            
            if len(key_pair) != 2:
                raise ValueError(f"Invalid key pair length: '{key_pair}'")
            
            if key_pair not in self.score_table.index:
                raise ValueError(f"Missing score for key pair: '{key_pair}'")
            
            original_raw_score = self.score_table.loc[key_pair, score_col]
            score = (1.0 - original_raw_score) if invert_scores else original_raw_score
            

            
            # Accumulate both original and processed scores
            original_raw_total += original_raw_score
            raw_total_score += score
            raw_count += 1
            
            if use_frequency:
                frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                weighted_total_score += score * frequency
                total_frequency += frequency
                if frequency > 0:
                    frequency_coverage += frequency
        
        # Calculate averages
        processed_raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        original_raw_average = original_raw_total / raw_count if raw_count > 0 else 0.0
        frequency_weighted_average = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
        

        
        results = {
            'average_score': frequency_weighted_average if use_frequency else processed_raw_average,
            'raw_average_score': original_raw_average,  # Show original raw score before inversion
            'total_score': weighted_total_score if use_frequency else raw_total_score,
            'raw_total_score': original_raw_total,  # Show original raw total before inversion
            'pair_count': raw_count,
            'coverage': raw_count / len(self.bigram_frequencies) if self.bigram_frequencies else 0.0,
        }
        
        if use_frequency:
            total_bigram_frequency = sum(self.bigram_frequencies.values())
            results.update({
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / total_bigram_frequency if total_bigram_frequency > 0 else 0.0
            })
        
        return results

    def _score_layout_comfort_key(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score a layout using comfort-key method."""
        if not self.letter_frequencies or not self.key_comfort_scores:
            missing = []
            if not self.letter_frequencies:
                missing.append("input/english-letter-frequencies-google-ngrams.csv")
            if not self.key_comfort_scores:
                missing.append("tables/key_scores.csv")
            raise ValueError(f"Required files missing for comfort-key scoring: {missing}")
        
        if not self.bigram_frequencies:
            raise ValueError("Bigram frequencies required for comfort-key scoring")
        
        valid_letter_pairs = self._get_valid_letter_pairs(layout_mapping)
        
        if self.verbose:
            print(f"Comfort-key: Processing {len(valid_letter_pairs)} valid pairs")
                
        raw_total_score = 0.0
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        
        use_frequency = not self.use_raw
        
        for letter_pair in valid_letter_pairs:
            comfort_key_score = self._compute_comfort_key_score(letter_pair, layout_mapping)
            
            if comfort_key_score is not None:
                raw_total_score += comfort_key_score
                raw_count += 1
                
                if use_frequency:
                    frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                    weighted_total_score += comfort_key_score * frequency
                    total_frequency += frequency
                    if frequency > 0:
                        frequency_coverage += frequency
        
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        
        results = {
            'pair_count': raw_count,
            'coverage': raw_count / len(self.bigram_frequencies) if self.bigram_frequencies else 0.0,
        }
        
        if use_frequency:
            primary_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
            results.update({
                'average_score': primary_score,
                'total_score': weighted_total_score,
                'raw_average_score': raw_average,
                'raw_total_score': raw_total_score,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            })
        else:
            results.update({
                'average_score': raw_average,
                'total_score': raw_total_score,
            })
        
        return results
    
    def _score_layout_engram(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score a layout using engram method (comfort * comfort-key)."""
        if not self.letter_frequencies or not self.key_comfort_scores:
            missing = []
            if not self.letter_frequencies:
                missing.append("input/english-letter-frequencies-google-ngrams.csv")
            if not self.key_comfort_scores:
                missing.append("tables/key_scores.csv")
            raise ValueError(f"Required files missing for engram scoring: {missing}")
        
        comfort_col = "comfort_score_normalized"
        if comfort_col not in self.score_table.columns:
            comfort_col = "comfort_normalized"
            if comfort_col not in self.score_table.columns:
                raise ValueError("Comfort scores not found in score table")
        
        if not self.bigram_frequencies:
            raise ValueError("Bigram frequencies required for engram scoring")
        
        valid_letter_pairs = self._get_valid_letter_pairs(layout_mapping)
        
        if self.verbose:
            print(f"Engram: Processing {len(valid_letter_pairs)} valid pairs")
        
        raw_total_score = 0.0
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        
        use_frequency = not self.use_raw
        
        for letter_pair in valid_letter_pairs:
            letter1, letter2 = letter_pair[0], letter_pair[1]
            
            key1 = layout_mapping[letter1]
            key2 = layout_mapping[letter2]
            key_pair = key1 + key2
            
            if key_pair in self.score_table.index:
                comfort_score = self.score_table.loc[key_pair, comfort_col]
                comfort_key_score = self._compute_comfort_key_score(letter_pair, layout_mapping)
                
                if comfort_key_score is not None:
                    engram_score = comfort_score * comfort_key_score
                    
                    raw_total_score += engram_score
                    raw_count += 1
                    
                    if use_frequency:
                        frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                        weighted_total_score += engram_score * frequency
                        total_frequency += frequency
                        if frequency > 0:
                            frequency_coverage += frequency
        
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        
        results = {
            'pair_count': raw_count,
            'coverage': raw_count / len(self.bigram_frequencies) if self.bigram_frequencies else 0.0,
        }
        
        if use_frequency:
            primary_score = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
            results.update({
                'average_score': primary_score,
                'total_score': weighted_total_score,
                'raw_average_score': raw_average,
                'raw_total_score': raw_total_score,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.bigram_frequencies.values()) if self.bigram_frequencies else 0.0
            })
        else:
            results.update({
                'average_score': raw_average,
                'total_score': raw_total_score,
            })
        
        return results
    
    def _score_layout_dvorak7_speed(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score layout using empirical Dvorak-7 speed weights."""
        if not DVORAK7_AVAILABLE:
            raise ValueError("prep_keypair_dvorak7_scores module not found")
        
        if not self.dvorak7_speed_weights:
            raise ValueError("Dvorak-7 speed weights not available")
        
        weights = self.dvorak7_speed_weights['weights']
        
        if self.bigram_frequencies:
            letter_pairs = list(self.bigram_frequencies.keys())
        else:
            letter_pairs = ['TH', 'HE', 'IN', 'ER', 'AN', 'ND', 'ON', 'EN', 'AT', 'OU',
                          'ED', 'HA', 'TO', 'OR', 'IT', 'IS', 'HI', 'ES', 'NG', 'VE']
        
        pure_scores = []
        speed_weighted_scores = []
        
        use_frequency = self.bigram_frequencies is not None and not self.use_raw
        weighted_total_pure = 0.0
        weighted_total_speed = 0.0
        total_frequency = 0.0
        
        for letter_pair in letter_pairs:
            if len(letter_pair) == 2:
                letter1, letter2 = letter_pair[0], letter_pair[1]
                
                if letter1 in layout_mapping and letter2 in layout_mapping:
                    try:
                        key1 = layout_mapping[letter1]
                        key2 = layout_mapping[letter2]
                        key_pair = key1 + key2
                        criteria_scores = score_bigram_dvorak7(key_pair)

                        pure_score = sum(criteria_scores.values()) / len(criteria_scores)
                        pure_scores.append(pure_score)
                        
                        speed_score = 0.0
                        total_abs_weight = 0.0
                        
                        for criterion, score in criteria_scores.items():
                            weight = weights.get(criterion, 0.0)
                            contribution = score * (-weight)
                            speed_score += contribution
                            total_abs_weight += abs(weight)
                        
                        if total_abs_weight > 0:
                            speed_score = speed_score / total_abs_weight
                        else:
                            speed_score = pure_score
                        
                        speed_weighted_scores.append(speed_score)
                        
                        if use_frequency:
                            frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                            weighted_total_pure += pure_score * frequency
                            weighted_total_speed += speed_score * frequency
                            total_frequency += frequency
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not score bigram '{letter_pair}': {e}")
                        continue
        
        if use_frequency and total_frequency > 0:
            pure_average = weighted_total_pure / total_frequency
            speed_average = weighted_total_speed / total_frequency
        else:
            pure_average = np.mean(pure_scores) if pure_scores else 0.0
            speed_average = np.mean(speed_weighted_scores) if speed_weighted_scores else 0.0
        
        results = {
            'average_score': speed_average,
            'raw_average_score': pure_average,
            'pure_dvorak7_score': pure_average,
            'speed_weighted_score': speed_average,
            'improvement_ratio': speed_average / pure_average if pure_average > 0 else 1.0,
            'letters_in_layout': len(layout_mapping),
            'bigrams_scored': len(pure_scores),
            'coverage': len(pure_scores) / len(letter_pairs) if letter_pairs else 0.0,
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
    
    def score_layout(self, layout_mapping: Dict[str, str], scorers: List[str]) -> Dict[str, Dict[str, float]]:
        """Score a layout using specified scoring methods."""
        results = {}
        
        for scorer in scorers:
            if scorer not in self.available_scorers:
                print(f"Warning: Scorer '{scorer}' not available. Available: {self.available_scorers}")
                continue
            
            try:
                scorer_results = self._score_layout_with_method(layout_mapping, scorer)
                was_inverted = self._should_invert_scores(scorer)
                display_name = self._get_display_name(scorer, was_inverted)
                results[display_name] = scorer_results
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

def safe_string_conversion(value) -> str:
    """Safely convert value to string, preserving apostrophes and avoiding NaN issues."""
    if value == "'":
        return "'"
    
    str_value = str(value).strip()
    
    if str_value.upper() in ['NAN', 'NA']:
        raise ValueError(f"Detected problematic value conversion: {value} -> {str_value}")
    
    return str_value

def create_layout_mapping(letters: str, positions: str) -> Dict[str, str]:
    """Create mapping from letters to QWERTY positions."""
    if len(letters) != len(positions):
        raise ValueError(f"Letters ({len(letters)}) and positions ({len(positions)}) must have same length")
    
    mapping = {}
    for letter, position in zip(letters, positions):
        letter_key = safe_string_conversion(letter).upper()
        pos_value = safe_string_conversion(position).upper()
        mapping[letter_key] = pos_value
    
    return mapping

def parse_layout_compare(compare_args: List[str]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Parse layout comparison arguments with safe string handling."""
    layouts = {}
    layout_strings = {}
    
    for arg in compare_args:
        if ':' not in arg:
            raise ValueError(f"Layout comparison format should be 'name:layout'. Got: {arg}")
        
        name, layout_str = arg.split(':', 1)
        name = safe_string_conversion(name)
        layout_str = safe_string_conversion(layout_str)
        
        layout_strings[name] = layout_str
        
        qwerty_positions = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
        
        if len(layout_str) > len(qwerty_positions):
            raise ValueError(f"Layout '{name}' too long. Max {len(qwerty_positions)} characters.")
        
        mapping = {}
        for i, char in enumerate(layout_str):
            char_upper = safe_string_conversion(char).upper()
            pos_str = safe_string_conversion(qwerty_positions[i])
            mapping[char_upper] = pos_str
        
        layouts[name] = mapping
    
    return layouts, layout_strings

def format_csv_output(comparison_results, layout_strings=None):
    """Format minimal CSV output for programmatic use."""
    lines = ["layout_name,scorer,weighted_score,raw_score,layout_string"]
    
    for layout_name, layout_results in comparison_results.items():
        for scorer, results in layout_results.items():
            safe_layout_name = str(layout_name).replace('"', '""')
            safe_scorer = str(scorer).replace('"', '""')
            
            weighted_score = float(results['average_score'])
            raw_score = float(results.get('raw_average_score', results['average_score']))
            
            layout_string = layout_strings.get(layout_name, "") if layout_strings else ""
            safe_layout_string = str(layout_string).replace('"', '""')
            
            lines.append(f'"{safe_layout_name}","{safe_scorer}",{weighted_score:.6f},{raw_score:.6f},"{safe_layout_string}"')
    
    return '\n'.join(lines)

def format_score_only_output(comparison_results):
    """Format score-only output."""
    scores = []
    for layout_name, layout_results in comparison_results.items():
        for scorer, results in layout_results.items():
            score = float(results['average_score'])
            scores.append(f"{score:.6f}")
    return '\n'.join(scores)

def format_comparison_table(comparison_results, layout_strings=None, quiet=False):
    """Format comparison results as a table."""
    if not comparison_results:
        return "No results to display."
    
    lines = []
    first_layout = list(comparison_results.values())[0]
    all_scorers = sorted(first_layout.keys())
    layout_names = list(comparison_results.keys())
    
    if len(layout_names) == 1:
        layout_name = layout_names[0]
        layout_results = comparison_results[layout_name]
        
        lines.append(f"\n{layout_name.upper()} RESULTS:")
        lines.append("=" * 50)
        
        first_scorer_results = list(layout_results.values())[0]
        pair_count = int(first_scorer_results['pair_count'])
        coverage = float(first_scorer_results['coverage'])
        
        lines.append(f"Pair count: {pair_count}")
        lines.append(f"Coverage: {coverage:.1%}")
        
        if 'frequency_coverage' in first_scorer_results:
            freq_coverage = float(first_scorer_results['frequency_coverage'])
            lines.append(f"Frequency coverage: {freq_coverage:.1%}")
        
        lines.append(f"\n{'Scorer':<20} {'Score':<10}")
        lines.append("-" * 35)
        
        for scorer in all_scorers:
            if scorer in layout_results:
                score = float(layout_results[scorer]['average_score'])
                lines.append(f"{scorer:<20} {score:<10.6f}")
    
    else:
        lines.append(f"\nLAYOUT COMPARISON:")
        lines.append("=" * 60)
        
        header = f"{'Scorer':<20}"
        for layout_name in layout_names:
            header += f"{layout_name:<12}"
        lines.append(header)
        lines.append("-" * (20 + 12 * len(layout_names)))
        
        for scorer in all_scorers:
            row = f"{scorer:<20}"
            for layout_name in layout_names:
                if scorer in comparison_results[layout_name]:
                    score = float(comparison_results[layout_name][scorer]['average_score'])
                    row += f"{score:<12.6f}"
                else:
                    row += f"{'N/A':<12}"
            lines.append(row)
        
        if not quiet:
            lines.append(f"\nSummary:")
            pair_counts = []
            coverages = []
            
            for layout_name, layout_results in comparison_results.items():
                if layout_results:
                    first_result = list(layout_results.values())[0]
                    pair_counts.append(int(first_result['pair_count']))
                    coverages.append(float(first_result['coverage']))
            
            if pair_counts:
                lines.append(f"Average pair count: {sum(pair_counts)/len(pair_counts):.0f}")
                lines.append(f"Average coverage: {sum(coverages)/len(coverages):.1%}")
    
    return '\n'.join(lines)

def save_detailed_comparison_csv(comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
                               filename: str, layout_mappings: Dict[str, Dict[str, str]] = None, 
                               use_raw: bool = False, layout_strings: Dict[str, str] = None):
    """Save detailed comparison results to CSV."""
    
    rows = []
    for layout_name, layout_results in comparison_results.items():
        for display_scorer, results in layout_results.items():
            row = {
                'layout_name': layout_name,
                'scorer': display_scorer,
                'average_score': results['average_score'],
                'total_score': results['total_score'],
                'pair_count': results['pair_count'],
                'coverage': results['coverage']
            }
            
            if not use_raw and 'raw_average_score' in results:
                row.update({
                    'raw_average_score': results['raw_average_score'],
                    'raw_total_score': results['raw_total_score'],
                    'total_frequency': results.get('total_frequency', 0),
                    'frequency_coverage': results.get('frequency_coverage', 0.0)
                })
            
            if layout_strings and layout_name in layout_strings:
                row['layout_string'] = layout_strings[layout_name]
            
            if layout_mappings and layout_name in layout_mappings:
                mapping = layout_mappings[layout_name]
                row['layout_letters'] = ''.join(sorted(mapping.keys()))
                row['layout_positions'] = ''.join(mapping[c] for c in sorted(mapping.keys()))
            
            rows.append(row)
    
    if rows:
        fieldnames = list(rows[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
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

  # Basic usage (uses default files: tables/keypair_scores_detailed.csv and input/english-letter-pair-frequencies-google-ngrams.csv)
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
- Uses tables/keypair_scores_detailed.csv for key-pair scoring data (created by prep_scoring_tables.py)
- Uses tables/key_scores.csv for individual key comfort scores (created by prep_scoring_tables.py)
- Uses input/english-letter-pair-frequencies-google-ngrams.csv for frequency weighting (if it exists)
- Uses input/english-letter-frequencies-google-ngrams.csv for letter frequencies (if it exists)
- Uses input/dvorak7_speed_weights.csv for empirical Dvorak-7 speed scoring (if it exists)
- Falls back to raw scoring if frequency file is not found
- With --raw: Ignores frequencies and uses raw (unweighted) scoring
- With --verbose: Shows both weighted and raw scores for comparison
- With --csv-output: Minimal CSV format for programmatic use (layout,scorer,weighted_score,raw_score)

Available scoring methods depend on the score table contents (e.g., distance, comfort, dvorak7, time).
Distance and time scores are automatically inverted (1-score) and renamed to efficiency and speed respectively.
Engram and comfort-key scores are computed dynamically and require letter frequencies and key comfort scores.
dvorak7-speed provides both pure and empirically-weighted Dvorak-7 scores based on 19.4M typing correlations.
        """
    )
    
    parser.add_argument(
        '--score-table',
        default="tables/keypair_scores_detailed.csv",
        help="Path to unified score table CSV file (default: tables/keypair_scores_detailed.csv)"
    )
    
    parser.add_argument(
        '--frequency-file',
        default="input/english-letter-pair-frequencies-google-ngrams.csv",
        help="Path to bigram frequency CSV file (default: input/english-letter-pair-frequencies-google-ngrams.csv)"
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Use raw (unweighted) scoring only, ignore frequency weighting"
    )
    
    parser.add_argument(
        '--csv-output',
        action='store_true',
        help="Output minimal CSV format to stdout (scores only, no headers, for programmatic use)"
    )
    
    scorer_group = parser.add_mutually_exclusive_group()
    scorer_group.add_argument(
        '--scorer',
        help="Run a single scorer (available scorers depend on table contents)"
    )
    scorer_group.add_argument(
        '--scorers',
        help="Run multiple scorers (comma-separated or 'all')"
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help="Compare layouts (e.g., qwerty:qwertyuiop dvorak:',.pyfgcrl)"
    )
    
    layout_group = parser.add_argument_group('Layout Definition')
    layout_group.add_argument(
        '--letters',
        help="String of characters in the layout"
    )
    layout_group.add_argument(
        '--positions', 
        help="String of corresponding QWERTY positions"
    )
    
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
    
    if args.score_only:
        args.format = 'score_only'
    elif args.csv_output:
        args.format = 'csv_output'
    
    try:
        frequency_file = None
        if Path(args.frequency_file).exists():
            frequency_file = args.frequency_file
        elif args.frequency_file != "input/english-letter-pair-frequencies-google-ngrams.csv":
            print(f"Error: Frequency file not found: {args.frequency_file}")
            return 1
        
        suppress_verbose = args.format == 'csv_output'
        scorer = LayoutScorer(args.score_table, frequency_file, args.raw, 
                             verbose=args.verbose and not suppress_verbose)
        
        if args.compare:
            layouts, layout_strings = parse_layout_compare(args.compare)
            
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
                scorers = scorer.available_scorers
            
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
            
            results = scorer.compare_layouts(layouts, scorers)
            
            if args.csv:
                save_detailed_comparison_csv(results, args.csv, layouts, args.raw, layout_strings)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Detailed comparison saved to: {args.csv}")
            else:
                if args.format == 'csv_output':
                    print(format_csv_output(results, layout_strings))
                elif args.format == 'score_only':
                    print(format_score_only_output(results))
                else:
                    if not args.quiet and args.format != 'csv_output':
                        print(f"\n=== RESULTS ===")
                    print(format_comparison_table(results, layout_strings, args.quiet))

        else:
            if not args.letters or not args.positions:
                print("Error: Must specify --letters and --positions for single layout scoring")
                return 1
            
            try:
                layout_mapping = create_layout_mapping(args.letters, args.positions)
            except Exception as e:
                print(f"Error creating layout mapping: {e}")
                return 1
            
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
                scorers = scorer.available_scorers
            
            if not scorers:
                print("Error: No valid scorers specified")
                return 1
            
            layout_name = f"{args.letters} → {args.positions}"
            results = scorer.score_layout(layout_mapping, scorers)
            
            if args.csv:
                comparison_results = {layout_name: results}
                layout_mappings_for_csv = {layout_name: layout_mapping}
                layout_strings_for_csv = {layout_name: args.letters}
                save_detailed_comparison_csv(comparison_results, args.csv, layout_mappings_for_csv, args.raw, layout_strings_for_csv)
                if not args.quiet and args.format != 'csv_output':
                    print(f"Results saved to: {args.csv}")
            else:
                if args.format == 'csv_output':
                    comparison_results = {layout_name: results}
                    layout_strings_for_output = {layout_name: args.letters}
                    print(format_csv_output(comparison_results, layout_strings_for_output))
                elif args.format == 'score_only':
                    comparison_results = {layout_name: results}
                    print(format_score_only_output(comparison_results))
                else:
                    comparison_results = {layout_name: results}
                    print(format_comparison_table(comparison_results, None, args.quiet))
        
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