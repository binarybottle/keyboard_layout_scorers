#!/usr/bin/env python3
"""
Keyboard Layout Scorer using precomputed score table.

A comprehensive tool for evaluating keyboard layouts using frequency-weighted scoring.
Core scoring methods include engram, dvorak7, comfort_combo, comfort, and comfort_key.
Now includes 3-key (trigram) Engram scoring in addition to 2-key (bigram) scoring.

Data Format Clarification
==========================================
Command Line Input:
- --letters: Letters in arbitrary order (e.g., "abc")  
- --positions: QWERTY positions where those letters go (e.g., "XYZ")
- --compare: Layout string in QWERTY key order (e.g., "name:qwertyuiop...")

CSV Table Input/Output:
- letters: Layout string in QWERTY key order (what's at each QWERTY position)
- positions: QWERTY reference string (constant: "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['")  
- layout_qwerty: Same as letters (alternative column name for MOO compatibility)

MOO Format Support:
- items: Letters in assignment order (e.g., "etaoinsrhldcum")
- positions: QWERTY positions where those letters go (e.g., "KJ;ASDVRLFUEIM")  
- layout_qwerty: Layout string in QWERTY key order (primary format for scoring)

Usage:
    # Command line (arbitrary order)
    python score_layouts.py --letters "etaoinsrhldcum" --positions "KJ;ASDVRLFUEIM"
    python score_layouts.py --letters "  cr  du  oinl  teha   s  m     " --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
    
    # CSV comparison (QWERTY order)
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl"
    
    # CSV file input (auto-detects layout_qwerty, letters, or MOO format)
    python score_layouts.py --compare-file layouts.csv --csv output.csv
    
    # Compare multiple layouts - CSV format
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;"
    
    # Save CSV (compatible with display_layouts.py and compare_layouts.py)
    python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv layouts.csv

    # Keyboard layout optimization study command:
    poetry run python3 score_layouts.py --compare \
        Focal:"vlhgkqfoujsrntbycaeizxmdp'w.;,['" \
        Hanster-23:"qyouxkgnfwhiea.dtrscj-'./bplmv;z" \
        Recurva:"frdpvqjuoysntcb.heaizxkgwml;',[/" \
        Graphite:"bldwz'foujnrtsgyhaeiqxmcvkp.-/;," \
        Gallium:"bldcvjfou,nrtsgyhaeixqmwzkp';.[/" \
        Seht-Drai:"fulvbqgnojsehtk'drai;mpwzycx/,[." \
        Octa8:"youkxgwdl,iaenfbstrc/zh'qvpmj.[;" \
        Canary:"wlypkzxou;crstbfneiajvdgqmh/,.['" \
        CTGAP:"qplcjxfou/rntsgyheiazbmwvkd',.['" \
        Nerps:"xldpvzkou;nrtsgyheiajmcwqbf',.['" \
        Whorf:"flhdmvwou,srntkgyaeixjbzqpc';.[/" \
        Sturdy:"vmlcpxfoujstrdy.naeizkqgwbh';,[/" \
        APTv3:"wgdfbqluoyrsthkjneaicmpvxz,.'/[;" \
        Semimak:"flhvzqwuoysrntkcdeaix'bmjpg,./[;" \
        ISRT:"yclmkzfu,'isrtgpneaoqvwdjbh/.x-;" \
        Engram-2021:"byou'\"ldwvciea,.htsngxjk-?rmfpzq" \
        Colemak-Q:";lcmkjfuyqarstgpneiozxwdvbh/.,-'" \
        Boo:",.ucvqfdlyaoesgbntri;x'wzphmkj[/" \
        BEAKL43:"qhou,glrfbjieaydtsnpxk\".'vmcwz/;" \
        Halmak:"wlrbz;qudjshnt,.aeoifmvc/gpxky['" \
        Hieamstrn:"jyou'=kdclhiea,-mtsrxqf.;zwgbvpn" \
        Colemak-DH:"qwfpbjluy;arstgmneiozxcdvkh,./['" \
        Norman:"qwdfkjurl;asetgyniohzxcvbpm,./['" \
        MTGap:"ypou-kdlcwinea,mhtsr()'._bfgvjx\"" \
        Workman:"qdrwbjfup;ashtgyneoizxmcvkl,./['" \
        QGMLWB:"qgmlwbyuv;dstnriaeohzxcfjkp,./['" \
        Colemak:"qwfpgjluy;arstdhneiozxcvbkm,./['" \
        Asset:"qwjfgypul;asetdhniorzxcvbkm,./['" \
        Capewell-Dvorak:"',.pyqfgrkoaeiudhtnszxcvjlmwb;/-" \
        Dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz/-" \
        QWERTY:"qwertyuiopasdfghjkl;zxcvbnm,./['" \
        --csv 31_layouts.csv


Setup:
1. Generate individual score files (keypair_*_scores.csv) using scoring scripts in prep/
2. Generate 3-key score files using: python prep_keytriple_engram_scores.py
3. Run: python prep_scoring_tables.py --input-dir tables/
   This creates: tables/scores_2key_detailed.csv and tables/key_scores.csv
4. Run this script to score layouts using all available methods

Default behavior:
- Score table: tables/scores_2key_detailed.csv (created by prep_scoring_tables.py)
- 3-key scores: tables/engram_3key_scores*.csv (created by prep_keytriple_engram_scores.py)
- Key scores: tables/key_scores.csv (created by prep_scoring_tables.py)  
- Frequency data:
  - Letter frequencies:  input/english-letter-counts-google-ngrams_normalized.csv
  - Bigram frequencies:  input/english-letter-pair-counts-google-ngrams_normalized.csv
  - Trigram frequencies: input/english-letter-triple-counts-google-ngrams_normalized.csv
- Scoring mode: Frequency-weighted (prioritizes common English letter combinations)
- Score mapping: Letter-pair frequencies → Key-pair scores, Letter-triple frequencies → Key-triple scores

Scoring ranges:
- Comfort scores: Normalized 0-1 (higher = more comfortable)
- Engram scores: 0-7 raw (sum of 7 components), normalized 0-1
- Engram 3-key scores, normalized 0-1  
- Dvorak-7 scores: 0-7 raw (sum of 7 components), normalized 0-1  

Core metrics (default):
- engram (bigram scores)
- engram_3key_* (individual 3-key criteria: order)
- dvorak7 (based on Dvorak's 7 typing principles)
- comfort_combo (composite comfort model)
- comfort_key (frequency-weighted key comfort)
- comfort (frequency-weighted key-pair comfort)  

Features:
- Automatic frequency weighting using English bigram and trigram frequencies
- Letter-pair → Key-pair mapping (e.g., "TH" frequency weights T→H key transition)
- Letter-triple → Key-triple mapping (e.g., "THE" frequency weights T→H→E key sequence)
- CSV format compatible with both display scripts and comparison tools
- Fallback to raw scoring if frequency file missing
- Support for all scoring methods in the score table
- Dynamic comfort_combo and comfort_key scoring (requires prep_scoring_tables.py output)
- 3-key Engram scoring with individual criterion breakdown
"""

import sys
import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# QWERTY reference order for positions
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

def load_layouts_from_csv(filepath: str, verbose: bool = False) -> Dict[str, str]:
    """
    Load layouts from CSV file, auto-detecting format.
    
    Supports multiple input formats:
    - layout_qwerty column (preferred)
    - letters column (QWERTY order)  
    - items + positions columns (MOO format - converts to layout_qwerty)
    
    Args:
        filepath: Path to CSV file
        verbose: Print format detection info
        
    Returns:
        Dict mapping layout names to layout strings in QWERTY order
    """
    try:
        df = pd.read_csv(filepath)
        layouts = {}
        
        if 'layout_qwerty' in df.columns:
            # Preferred format: layout_qwerty column
            if verbose:
                print(f"Using layout_qwerty column from {filepath}")
            
            layout_col = df.columns[0] if 'layout' in df.columns[0].lower() else 'layout'
            if layout_col not in df.columns:
                layout_col = 'layout'  # fallback
                
            for i, row in df.iterrows():
                layout_name = row.get(layout_col, f"layout_{i}")
                layout_qwerty = str(row['layout_qwerty'])
                layouts[layout_name] = layout_qwerty
                
        elif 'letters' in df.columns:
            # Standard format: letters in QWERTY order
            if verbose:
                print(f"Using letters column from {filepath}")
                
            layout_col = df.columns[0] if 'layout' in df.columns[0].lower() else 'layout'
            if layout_col not in df.columns:
                layout_col = 'layout'  # fallback
                
            for i, row in df.iterrows():
                layout_name = row.get(layout_col, f"layout_{i}")
                letters = str(row['letters'])
                layouts[layout_name] = letters
                
        elif 'items' in df.columns and 'positions' in df.columns:
            # MOO format: convert items + positions to layout_qwerty
            if verbose:
                print(f"Converting MOO format (items + positions) from {filepath}")
                
            layout_col = df.columns[0] if 'layout' in df.columns[0].lower() else 'config_id'
            if layout_col not in df.columns:
                layout_col = 'layout'  # fallback
                
            QWERTY_ORDER = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"
            
            for i, row in df.iterrows():
                layout_name = row.get(layout_col, f"layout_{i}")
                items = str(row['items'])
                positions = str(row['positions'])
                
                # Convert to layout_qwerty format
                pos_to_letter = dict(zip(positions, items))
                layout_chars = []
                for qwerty_pos in QWERTY_ORDER:
                    if qwerty_pos in pos_to_letter:
                        layout_chars.append(pos_to_letter[qwerty_pos])
                    else:
                        layout_chars.append(' ')
                
                layout_qwerty = ''.join(layout_chars)
                layouts[layout_name] = layout_qwerty
        else:
            raise ValueError("CSV file must contain either 'layout_qwerty', 'letters', or 'items'+'positions' columns")
            
        if verbose:
            print(f"Loaded {len(layouts)} layouts from {filepath}")
            
        return layouts
        
    except Exception as e:
        raise ValueError(f"Error loading layouts from {filepath}: {e}")
    
class LayoutScorer:
    """Layout scorer using pre-computed score table."""
    
    def __init__(self, score_table_path: str, frequency_file: Optional[str] = None, use_raw: bool = False, verbose: bool = False, experimental_metrics: bool = False):
        """Initialize scorer with score table and optional frequency data."""
        self.verbose = verbose
        self.use_raw = use_raw
        self.experimental_metrics = experimental_metrics
        self.score_table = self._load_score_table(score_table_path)
        
        # Load 3-key score tables
        self.score_3key_tables = self._load_3key_score_tables()
        
        self.available_scorers = self._detect_available_scorers()
        
        self.bigram_frequencies = None
        if frequency_file:
            self.bigram_frequencies = self._load_frequency_data(frequency_file)
        
        # Try to load trigram frequencies
        self.trigram_frequencies = self._load_trigram_frequencies()
        
        self.letter_frequencies = self._load_letter_frequencies()
        self.key_comfort_scores = self._load_key_comfort_scores()
        
        self._update_available_scorers()
        
        if self.verbose:
            self._print_initialization_info(score_table_path, frequency_file)
    
    def _load_score_table(self, filepath: str) -> pd.DataFrame:
        """Load the score table with careful NA handling."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Score table not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath, 
                        keep_default_na=False,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan'])
        except Exception as e:
            raise ValueError(f"Error reading score table: {e}")
        
        # Determine which pair column to use
        position_column_name = None
        if 'key_pair' in df.columns:
            position_column_name = 'key_pair'
        elif 'position_pair' in df.columns:
            position_column_name = 'position_pair'
        elif 'key_triple' in df.columns:
            position_column_name = 'key_triple'
        elif 'position_triple' in df.columns:
            position_column_name = 'position_triple'
        else:
            raise ValueError("Score table must have either 'key_pair/position_pair/key_triple/position_triple' column")
        
        # Apply string dtype to the pair column
        df[position_column_name] = df[position_column_name].astype('str')
        
        missing_count = df[position_column_name].isna().sum()
        if missing_count > 0:
            df = df.dropna(subset=[position_column_name])
            if self.verbose:
                print(f"Removed {missing_count} rows with missing {position_column_name} values")
        
        empty_count = (df[position_column_name].astype(str).str.strip() == '').sum()
        if empty_count > 0:
            df = df[df[position_column_name].astype(str).str.strip() != '']
            if self.verbose:
                print(f"Removed {empty_count} rows with empty {position_column_name} values")
        
        if self.verbose and 'NA' in df[position_column_name].values:
            print(f"'NA' {position_column_name} preserved in score table")
        
        return df.set_index(position_column_name)

    def _load_3key_score_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all 3-key score tables."""
        tables = {}
        base_path = Path("tables")
        
        # Define 3-key score files
        score_files = {
            'engram_order': 'engram_3key_scores_order.csv',
        }
        
        for score_name, filename in score_files.items():
            filepath = base_path / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath,
                                   dtype={'key_triple': 'str'},
                                   keep_default_na=False,
                                   na_values=['', 'NULL', 'null', 'NaN', 'nan'])
                    
                    if 'key_triple' in df.columns:
                        # Clean data
                        missing_count = df['key_triple'].isna().sum()
                        if missing_count > 0:
                            df = df.dropna(subset=['key_triple'])
                        
                        empty_count = (df['key_triple'].astype(str).str.strip() == '').sum()
                        if empty_count > 0:
                            df = df[df['key_triple'].astype(str).str.strip() != '']
                        
                        # Set index and normalize scores to 0-1 range
                        df_indexed = df.set_index('key_triple')
                        
                        # Find the score column - check direct match first
                        potential_columns = [score_name]  # Try direct match first
                        if score_name.startswith('engram_'):
                            # Also try without the engram_ prefix
                            criterion = score_name.replace('engram_', '', 1)
                            potential_columns.append(f'engram_{criterion}')
                        else:
                            potential_columns.append(f'engram_{score_name}')

                        score_col = None
                        for col in potential_columns:
                            if col in df_indexed.columns:
                                score_col = col
                                break

                        if score_col in df_indexed.columns:
                            # Normalize scores (Engram ranges from 0-6, individual criteria 0-1)
                            scores = df_indexed[score_col].values
                            if score_name == 'engram_3key':
                                # Overall score: normalize 0-6 to 0-1
                                normalized_scores = scores / 6.0
                            else:
                                # Individual criteria: already 0-1, but clip to be safe
                                normalized_scores = np.clip(scores, 0.0, 1.0)
                            
                            df_indexed[f'{score_col}_normalized'] = normalized_scores
                            tables[score_name] = df_indexed
                            
                            if self.verbose:
                                print(f"Loaded 3-key table: {filename} ({len(df_indexed)} triples)")
                        else:
                            if self.verbose:
                                print(f"Warning: Score column '{score_col}' not found in {filename}")
                    else:
                        if self.verbose:
                            print(f"Warning: 'key_triple' column not found in {filename}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading 3-key table {filename}: {e}")
            else:
                if self.verbose:
                    print(f"3-key table not found: {filename}")
        
        return tables

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
        
        bigram_col = self._find_column(df, ['bigram', 'pair', 'key_pair', 'letter_pair', 'item', 'item_pair'])
        freq_col = self._find_column(df, ['normalized_frequency', 'frequency', 'count', 'score'])
        
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

    def _load_trigram_frequencies(self) -> Optional[Dict[str, float]]:
        """Load trigram frequency data from CSV file."""
        filepath = "input/english-letter-triple-counts-google-ngrams_normalized.csv"
        
        if not Path(filepath).exists():
            if self.verbose:
                print(f"Trigram frequency file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            if self.verbose:
                print(f"Error reading trigram frequency file: {e}")
            return None
        
        trigram_col = self._find_column(df, ['trigram', 'triple', 'key_triple', 'letter_triple', 'item', 'item_triple'])
        freq_col = self._find_column(df, ['normalized_frequency', 'frequency', 'count', 'score'])
        
        if not trigram_col or not freq_col:
            if self.verbose:
                print("Could not find required columns in trigram frequency file")
            return None
        
        frequencies = {}
        for _, row in df.iterrows():
            trigram = str(row[trigram_col]).strip().upper()
            freq = float(row[freq_col])
            
            if len(trigram) == 3:
                frequencies[trigram] = freq
        
        if self.verbose:
            total_freq = sum(frequencies.values())
            print(f"Loaded {len(frequencies)} trigram frequencies (total: {total_freq:,})")
        
        return frequencies
    
    def _load_letter_frequencies(self) -> Optional[Dict[str, float]]:
        """Load individual letter frequency data."""
        filepath = "input/english-letter-counts-google-ngrams_normalized.csv"
        
        if not Path(filepath).exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            letter_col = self._find_column(df, ['letter', 'char', 'character', 'item'])
            freq_col = self._find_column(df, ['normalized_frequency', 'frequency', 'count', 'score'])
            
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
        
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _detect_available_scorers(self) -> List[str]:
        """Detect available scoring methods from table columns, ordered logically."""
        available_scorers = set()
        
        # Add 2-key scorers from main table
        for col in self.score_table.columns:
            if col.endswith('_normalized'):
                scorer_name = col.replace('_normalized', '').replace('_score', '')
                available_scorers.add(scorer_name)
        
        # Add 3-key scorers from 3-key tables
        for score_name in self.score_3key_tables.keys():
            available_scorers.add(score_name)
        
        # Filter out experimental metrics unless experimental_metrics is enabled
        if not self.experimental_metrics:
            filtered_time = [s for s in available_scorers if self._is_time_speed_metric(s)]
            filtered_distance = [s for s in available_scorers if self._is_distance_efficiency_metric(s)]
            filtered_out = filtered_time + filtered_distance
            
            available_scorers = {s for s in available_scorers if not self._is_time_speed_metric(s) and not self._is_distance_efficiency_metric(s)}
            
            if self.verbose and filtered_out:
                print(f"Filtered out experimental metrics (use --experimental-metrics to enable): {filtered_out}")
        
        # Order scorers logically: composite scores first, then components by category
        ordered_scorers = []
        
        # 1. Dynamic composite scores (computed on-the-fly)
        dynamic_composites = ['comfort_combo', 'comfort_key']
        for scorer in dynamic_composites:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 2. Core composite scores (2-key)
        core_2key_composites = ['comfort', 'engram', 'dvorak7']
        for scorer in core_2key_composites:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 3. 3-key composite scores
        key_3key_composites = ['engram_3key']
        for scorer in key_3key_composites:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 4. 2-key Engram component scores (in logical order)
        engram_2key_components = ['engram_key_preference', 'engram_row_separation', 'engram_same_row', 'engram_same_finger', 
                                  'engram_outside']
        for scorer in engram_2key_components:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 5. 3-key Engram component scores (in logical order)
        engram_3key_components = ['engram_3key_order']
        for scorer in engram_3key_components:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 6. Dvorak-7 component scores (in logical order)
        dvorak7_components = [
            'dvorak7_distribution', 'dvorak7_strength', 'dvorak7_home', 'dvorak7_vspan',
            'dvorak7_columns', 'dvorak7_remote', 'dvorak7_inward'
        ]
        for scorer in dvorak7_components:
            if scorer in available_scorers:
                ordered_scorers.append(scorer)
        
        # 7. Experimental metrics (if enabled)
        if self.experimental_metrics:
            # Distance/efficiency metrics
            distance_metrics = sorted([s for s in available_scorers if self._is_distance_efficiency_metric(s)])
            ordered_scorers.extend(distance_metrics)
            
            # Time/speed metrics  
            time_metrics = sorted([s for s in available_scorers if self._is_time_speed_metric(s)])
            ordered_scorers.extend(time_metrics)
        
        # 8. Any remaining scorers (fallback for unexpected scorers)
        remaining = sorted([s for s in available_scorers if s not in ordered_scorers])
        ordered_scorers.extend(remaining)
        
        return ordered_scorers
    
    def _update_available_scorers(self):
        """Add dynamic scorers to available list."""
        if self.letter_frequencies and self.key_comfort_scores:
            if 'comfort_key' not in self.available_scorers:
                self.available_scorers.append('comfort_key')
            if 'comfort' in self.available_scorers and 'comfort_combo' not in self.available_scorers:
                self.available_scorers.append('comfort_combo')
            
    def _print_initialization_info(self, score_table_path: str, frequency_file: Optional[str]):
        """Print initialization information if verbose."""
        print(f"Score table: {score_table_path}")
        if frequency_file:
            print(f"Frequency file: {frequency_file}")
        print(f"Loaded score table with {len(self.score_table)} key pairs")
        print(f"Loaded {len(self.score_3key_tables)} 3-key score tables")
        print(f"Available scorers: {', '.join(self.available_scorers)}")
        
        if self.use_raw:
            print("Using raw (unweighted) scoring only")
        elif self.bigram_frequencies:
            total_freq = sum(self.bigram_frequencies.values())
            print(f"Loaded frequency data for {len(self.bigram_frequencies)} bigrams")
            print(f"Total frequency count: {total_freq:,}")
            if self.trigram_frequencies:
                trigram_total = sum(self.trigram_frequencies.values())
                print(f"Loaded frequency data for {len(self.trigram_frequencies)} trigrams")
                print(f"Total trigram frequency count: {trigram_total:,}")
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
    
    def _is_time_speed_metric(self, scorer: str) -> bool:
        """Determine if a scorer is a time/speed-related metric that should be filtered by default."""
        scorer_lower = scorer.lower()
        return (
            scorer_lower.startswith('time_') or
            scorer_lower.startswith('speed_') or
            scorer_lower == 'time' or
            scorer_lower == 'speed'
        )
    
    def _is_distance_efficiency_metric(self, scorer: str) -> bool:
        """Determine if a scorer is a distance/efficiency-related metric that should be filtered by default."""
        scorer_lower = scorer.lower()
        return (
            scorer_lower.startswith('distance_') or
            scorer_lower.startswith('efficiency_') or
            scorer_lower == 'distance' or
            scorer_lower == 'efficiency' or
            scorer_lower.endswith('_dist')
        )
    
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

    def _get_valid_letter_triples(self, layout_mapping: Dict[str, str]) -> List[str]:
        """Get letter triples that can be scored with the current layout."""
        if not self.trigram_frequencies:
            return []
        
        valid_triples = []
        for letter_triple in self.trigram_frequencies.keys():
            if len(letter_triple) == 3:
                letter1, letter2, letter3 = letter_triple[0], letter_triple[1], letter_triple[2]
                if letter1 in layout_mapping and letter2 in layout_mapping and letter3 in layout_mapping:
                    valid_triples.append(letter_triple)
        
        return valid_triples
    
    def _compute_comfort_key_score(self, letter_pair: str, layout_mapping: Dict[str, str]) -> Optional[float]:
        """Compute comfort_key score for a letter pair."""
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

    def _score_layout_3key(self, layout_mapping: Dict[str, str], scorer: str) -> Dict[str, float]:
        """Score a layout using 3-key scoring method."""
        if scorer not in self.score_3key_tables:
            raise ValueError(f"3-key scorer '{scorer}' not available")
        
        score_table = self.score_3key_tables[scorer]
        
        # Determine score column - try multiple possibilities
        score_col = None
        potential_columns = [
            scorer,  # Direct match (e.g., 'engram_order')
            f'{scorer}_normalized',  # With _normalized suffix
            scorer.replace('engram_', ''),  # Without engram_ prefix
            f'{scorer.replace("engram_", "")}_normalized'  # Without prefix, with suffix
        ]

        for col in potential_columns:
            if col in score_table.columns:
                score_col = col
                break

        if score_col is None:
            raise ValueError(f"Score column for '{scorer}' not found. Available columns: {list(score_table.columns)}")
        
        if score_col not in score_table.columns:
            raise ValueError(f"Score column '{score_col}' not found in 3-key table")
        
        # Use trigram frequencies if available, otherwise fall back to raw scoring
        use_frequency = self.trigram_frequencies is not None and not self.use_raw
        
        if use_frequency:
            valid_letter_triples = self._get_valid_letter_triples(layout_mapping)
        else:
            # For raw scoring, generate all possible triples from the layout
            letters = list(layout_mapping.keys())
            valid_letter_triples = []
            for l1 in letters:
                for l2 in letters:
                    for l3 in letters:
                        valid_letter_triples.append(l1 + l2 + l3)
        
        if self.verbose:
            if use_frequency:
                print(f"3-key {scorer}: Processing {len(valid_letter_triples)} valid triples out of {len(self.trigram_frequencies)} total")
            else:
                print(f"3-key {scorer}: Processing {len(valid_letter_triples)} raw triples")
        
        raw_total_score = 0.0
        raw_count = 0
        weighted_total_score = 0.0
        total_frequency = 0.0
        frequency_coverage = 0.0
        
        for letter_triple in valid_letter_triples:
            if len(letter_triple) != 3:
                continue
                
            letter1, letter2, letter3 = letter_triple[0], letter_triple[1], letter_triple[2]
            
            if letter1 not in layout_mapping or letter2 not in layout_mapping or letter3 not in layout_mapping:
                continue
            
            key1 = layout_mapping[letter1]
            key2 = layout_mapping[letter2]
            key3 = layout_mapping[letter3]
            key_triple = str(key1) + str(key2) + str(key3)
            
            if len(key_triple) != 3:
                continue
            
            if key_triple not in score_table.index:
                continue
            
            score = score_table.loc[key_triple, score_col]
            
            raw_total_score += score
            raw_count += 1
            
            if use_frequency:
                frequency = self.trigram_frequencies.get(letter_triple, 0.0)
                weighted_total_score += score * frequency
                total_frequency += frequency
                if frequency > 0:
                    frequency_coverage += frequency
        
        # Calculate averages
        raw_average = raw_total_score / raw_count if raw_count > 0 else 0.0
        frequency_weighted_average = weighted_total_score / total_frequency if total_frequency > 0 else 0.0
        
        results = {
            'average_score': frequency_weighted_average if use_frequency else raw_average,
            'raw_average_score': raw_average,
            'total_score': weighted_total_score if use_frequency else raw_total_score,
            'raw_total_score': raw_total_score,
            'pair_count': raw_count,  # Actually triple_count, but keeping consistent naming
        }
        
        if use_frequency:
            results.update({
                'coverage': raw_count / len(self.trigram_frequencies) if self.trigram_frequencies else 0.0,
                'total_frequency': total_frequency,
                'frequency_coverage': frequency_coverage / sum(self.trigram_frequencies.values()) if self.trigram_frequencies else 0.0
            })
        else:
            # For raw scoring, coverage is against all possible triples in the layout
            total_possible = len(layout_mapping) ** 3
            results['coverage'] = raw_count / total_possible if total_possible > 0 else 0.0
        
        return results
    
    def _score_layout_with_method(self, layout_mapping: Dict[str, str], scorer: str) -> Dict[str, float]:
        """Score a layout using a specific scoring method."""
        self._validate_layout_mapping(layout_mapping)
        
        # Check if this is a 3-key scorer
        if scorer in self.score_3key_tables:
            return self._score_layout_3key(layout_mapping, scorer)
        
        if scorer == 'comfort_key':
            return self._score_layout_comfort_key(layout_mapping)
        elif scorer == 'comfort_combo':
            return self._score_layout_comfort_combo(layout_mapping)
        
        # Standard 2-key scoring
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
        """Score a layout using comfort_key method."""
        if not self.letter_frequencies or not self.key_comfort_scores:
            missing = []
            if not self.letter_frequencies:
                missing.append("input/english-letter-counts-google-ngrams_normalized.csv")
            if not self.key_comfort_scores:
                missing.append("tables/key_scores.csv")
            raise ValueError(f"Required files missing for comfort_key scoring: {missing}")
        
        if not self.bigram_frequencies:
            raise ValueError("Bigram frequencies required for comfort_key scoring")
        
        valid_letter_pairs = self._get_valid_letter_pairs(layout_mapping)
        
        if self.verbose:
            print(f"comfort_key: Processing {len(valid_letter_pairs)} valid pairs")
                
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
    
    def _score_layout_comfort_combo(self, layout_mapping: Dict[str, str]) -> Dict[str, float]:
        """Score a layout using comfort_combo method (comfort * comfort_key)."""
        if not self.letter_frequencies or not self.key_comfort_scores:
            missing = []
            if not self.letter_frequencies:
                missing.append("input/english-letter-counts-google-ngrams_normalized.csv")
            if not self.key_comfort_scores:
                missing.append("tables/key_scores.csv")
            raise ValueError(f"Required files missing for comfort_combo scoring: {missing}")
        
        comfort_col = "comfort_score_normalized"
        if comfort_col not in self.score_table.columns:
            comfort_col = "comfort_normalized"
            if comfort_col not in self.score_table.columns:
                raise ValueError("Comfort scores not found in score table")
        
        if not self.bigram_frequencies:
            raise ValueError("Bigram frequencies required for comfort_combo scoring")
        
        valid_letter_pairs = self._get_valid_letter_pairs(layout_mapping)
        
        if self.verbose:
            print(f"comfort_combo: Processing {len(valid_letter_pairs)} valid pairs")
        
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
                    comfort_combo_score = comfort_score * comfort_key_score
                    
                    raw_total_score += comfort_combo_score
                    raw_count += 1
                    
                    if use_frequency:
                        frequency = self.bigram_frequencies.get(letter_pair, 0.0)
                        weighted_total_score += comfort_combo_score * frequency
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

def safe_string_conversion(value, preserve_spaces: bool = False) -> str:
    """Safely convert value to string, preserving apostrophes and avoiding NaN issues."""
    if value == "'":
        return "'"
    
    str_value = str(value)
    if not preserve_spaces:
        str_value = str_value.strip()
    
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
        layout_str = safe_string_conversion(layout_str, preserve_spaces=True)
        
        layout_strings[name] = layout_str
        
        qwerty_positions = QWERTY_POSITIONS
        
        if len(layout_str) > len(qwerty_positions):
            raise ValueError(f"Layout '{name}' too long. Max {len(qwerty_positions)} characters.")
        
        # Pad layout string if it's shorter than QWERTY
        if len(layout_str) < len(qwerty_positions):
            layout_str = layout_str + ' ' * (len(qwerty_positions) - len(layout_str))
        
        mapping = {}
        for i, char in enumerate(layout_str):
            if char != ' ':  # Skip spaces
                char_upper = safe_string_conversion(char).upper()
                qwerty_pos = qwerty_positions[i]
                mapping[char_upper] = qwerty_pos
    
        layouts[name] = mapping
    
    return layouts, layout_strings

def layout_string_to_letters_positions(layout_string: str) -> Tuple[str, str]:
    """Convert layout string to letters and positions for display compatibility."""
    # Pad or truncate to match QWERTY length
    padded_layout = layout_string
    if len(padded_layout) < len(QWERTY_POSITIONS):
        padded_layout = padded_layout + ' ' * (len(QWERTY_POSITIONS) - len(padded_layout))
    elif len(padded_layout) > len(QWERTY_POSITIONS):
        padded_layout = padded_layout[:len(QWERTY_POSITIONS)]
    
    letters = padded_layout
    positions = QWERTY_POSITIONS
    
    return letters, positions

def format_unified_csv_output(comparison_results: Dict[str, Dict[str, Dict[str, float]]], 
                             layout_strings: Dict[str, str], 
                             scorers: List[str]) -> str:
    """Format CSV output compatible with display scripts and compare_layouts.py."""
    lines = []
    
    # Header: layout,letters,positions,layout_qwerty,scorer1,scorer2,...
    header = ['layout', 'letters', 'positions', 'layout_qwerty']
    header.extend(scorers)
    lines.append(','.join(header))
    
    # Data rows
    for layout_name, layout_results in comparison_results.items():
        row_data = [f'"{layout_name}"']
        
        # Get letters and positions from layout string
        layout_string = layout_strings.get(layout_name, "")
        letters, positions = layout_string_to_letters_positions(layout_string)
        row_data.append(f'"{letters}"')
        row_data.append(f'"{positions}"')
        
        # Add layout_qwerty (same as letters for this format)
        row_data.append(f'"{letters}"')
        
        # Add scorer values
        for scorer in scorers:
            if scorer in layout_results:
                score = float(layout_results[scorer]['average_score'])
                row_data.append(f"{score:.6f}")
            else:
                row_data.append("")  # Missing score
        
        lines.append(','.join(row_data))
    
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

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Keyboard layout scorer with CSV output format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Data Format Clarification:
=========================

Command Line Input:
  --letters: Letters in arbitrary order (e.g., "abc")  
  --positions: QWERTY positions where those letters go (e.g., "XYZ")

CSV Input/Output Formats:
  Standard: letters (QWERTY order), positions (QWERTY reference)
  MOO: items (arbitrary order), positions (where they go), layout_qwerty (QWERTY order)
  Preferred: layout_qwerty (QWERTY order), positions (QWERTY reference)

Examples:
  # Command line scoring (arbitrary order)
  python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
  
  # Compare layouts (QWERTY order) 
  python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl"
  
  # CSV file input (auto-detects format)
  python score_layouts.py --compare-file layouts.csv --csv output.csv
  
  # Save CSV output
  python score_layouts.py --compare qwerty:"qwertyuiop" --csv layouts.csv
        """
    )
    
    parser.add_argument(
        '--score-table',
        default="tables/scores_2key_detailed.csv",
        help="Path to score table CSV file (default: tables/scores_2key_detailed.csv)"
    )
    
    parser.add_argument(
        '--frequency-file',
        default="input/english-letter-pair-counts-google-ngrams_normalized.csv",
        help="Path to bigram frequency CSV file (default: input/english-letter-pair-counts-google-ngrams_normalized.csv)"
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Use raw (unweighted) scoring only, ignore frequency weighting"
    )
    
    parser.add_argument(
        '--experimental-metrics',
        action='store_true',
        help="Enable distance/efficiency AND time/speed metrics (disabled by default due to limitations described above)"
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
    parser.add_argument(
        '--compare-file',
        help="Compare layouts from CSV file (auto-detects layout_qwerty, letters, or items+positions format)"
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
        choices=['detailed', 'score_only', 'table'],
        default='table',
        help="Output format (default: table)"
    )
    output_group.add_argument(
        '--csv',
        help="Save CSV format to file (layout,letters,positions,scorer1,scorer2,...)"
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
    
    try:
        frequency_file = None
        if Path(args.frequency_file).exists():
            frequency_file = args.frequency_file
        elif args.frequency_file != "input/english-letter-pair-counts-google-ngrams_normalized.csv":
            print(f"Error: Frequency file not found: {args.frequency_file}")
            return 1
        
        scorer = LayoutScorer(args.score_table, frequency_file, args.raw, 
                             verbose=args.verbose, experimental_metrics=args.experimental_metrics)
        
        if args.compare or args.compare_file:
            # Handle CSV file input
            if args.compare_file:
                # Load from CSV file
                layouts_from_file = load_layouts_from_csv(args.compare_file, args.verbose)
                # Convert to the format expected by parse_layout_compare
                compare_args = [f"{name}:{layout}" for name, layout in layouts_from_file.items()]
                layouts, layout_strings = parse_layout_compare(compare_args)
            else:
                # Use command line compare arguments
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
            
            if not args.quiet:
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
                # Save CSV format
                unified_csv = format_unified_csv_output(results, layout_strings, scorers)
                with open(args.csv, 'w', encoding='utf-8') as f:
                    f.write(unified_csv)
                if not args.quiet:
                    print(f"CSV saved to: {args.csv}")
            else:
                if args.format == 'score_only':
                    print(format_score_only_output(results))
                else:
                    if not args.quiet:
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
                # Save CSV format for single layout
                comparison_results = {layout_name: results}
                layout_strings_for_csv = {layout_name: args.letters}
                unified_csv = format_unified_csv_output(comparison_results, layout_strings_for_csv, scorers)
                with open(args.csv, 'w', encoding='utf-8') as f:
                    f.write(unified_csv)
                if not args.quiet:
                    print(f"CSV saved to: {args.csv}")
            else:
                if args.format == 'score_only':
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