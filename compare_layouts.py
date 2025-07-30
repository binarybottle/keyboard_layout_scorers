#!/usr/bin/env python3
"""
Comprehensive keyboard layout comparison with full scoring matrix:
- Engram scores (total, item, item_pair)  
- Dvorak-9 scores: unweighted, speed-weighted, comfort-weighted (text-independent)
- Distance scores: total distance, average distance, normalized score (text-dependent)
- All 9 individual Dvorak-9 criteria
- Tested on Google Ngrams bigram data + multiple text corpora

Usage:
    python compare_layouts.py --text-dir ../text_data/samples --output output/
    python compare_layouts.py --sample-mode --output-prefix quick_test
"""

import os
import sys
import subprocess
import pandas as pd
import argparse
import tempfile
import re
import csv
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Layout definitions
LAYOUTS = {
    'Engram': ['B','Y','O','U', 'C','I','E','A', 'G','X','J','K', 'L','D','W','V', 'H','T','S','N', 'R','M','F','P', "'",',','-', '"','.','?', 'Z','Q'],
    'Halmak': ['W','L','R','B', 'S','H','N','T', 'F','M','V','C', 'Q','U','D','J', 'A','E','O','I', 'P','X','K','Y', 'Z',',','/', ';','.','G', '[',"'"],
    'Hieamtsrn': ['B','Y','O','U', 'H','I','E','A', 'X','-','"','.', 'D','C','L','P', 'T','S','R','N', 'G','F','J','Z', "'",',','?', 'K','M','W', 'Q','V'],
    'Norman': ['Q','W','D','F', 'A','S','E','T', 'Z','X','C','V', 'U','R','L',';', 'N','I','O','H', 'M',',','.','/', 'K','G','B', 'J','Y','P', '[',"'"],
    'Workman': ['Q','D','R','W', 'A','S','H','T', 'Z','X','M','C', 'F','U','P',';', 'N','E','O','I', 'L',',','.','/', 'B','G','V', 'J','Y','K', '[',"'"],
    'MTGap': [',','F','H','D', 'O','A','N','T', 'Q','X','B','P', 'C','U','L','.', 'S','E','R','I', 'W',"'",'V',';', "K",'G','Z', 'J','M','Y', '[','/'],
    'QGMLWB': ['Q','G','M','L', 'D','S','T','N', 'Z','X','C','F', 'Y','U','V',';', 'A','E','O','H', 'P',',','.','/', 'W','R','J', 'B','I','K', '[',"'"],
    'ColemakMod': ['Q','W','F','P', 'A','R','S','T', 'Z','X','C','D', 'L','U','Y',';', 'N','E','I','O', 'H',',','.','/', 'B','J','G', 'K','V','M', '[',"'"],
    'Colemak': ['Q','W','F','P', 'A','R','S','T', 'Z','X','C','V', 'L','U','Y',';', 'N','E','I','O', 'M',',','.','/', 'G','D','B', 'J','H','K', '[',"'"],
    'Asset': ['Q','W','J','F', 'A','S','E','T', 'Z','X','C','V', 'P','U','L',';', 'N','I','O','R', 'M',',','.','/', 'G','Y','D', 'H','B','K', '[',"'"],
    'Dvorak': ["'",',','.','P', 'A','O','E','U', ';','Q','J','K', 'G','C','R','L', 'H','T','N','S', 'M','W','V','Z', 'Y','F','I', 'D','X','B', '/','-'],
    'QWERTY': ['Q','W','E','R', 'A','S','D','F', 'Z','X','C','V', 'U','I','O','P', 'J','K','L',';', 'M',',','.','/', 'T','G','B', 'Y','H','N', '[',"'"],
}

# Text corpus definitions
TEXT_SOURCES = {
    "Alice": "AliceInWonderland_Ch1.txt",
    "Memento": "Memento_screenplay.txt", 
    "Tweets_100K": "training.1600000.processed.noemoticon_1st100000tweets.txt",
    "Tweets_20K": "gender-classifier-20000tweets.txt",
    "Tweets_MASC": "MASC_tweets_cleaned.txt",
    "Spoken_MASC": "MASC_spoken_transcripts_of_phone_face2face.txt",
    "COCA_blogs": "COCA_corpusdata.org_sample_text_blog_cleaned.txt",
    "iweb": "iweb-corpus-cleaned-150000-lines.txt",
    "Monkey": "monkey0-7_IanDouglas.txt",
    "Coder": "coder0-7_IanDouglas.txt",
    "Rosetta": "rosettacode.org_TowersOfHanoi_AtoZ.txt"
}

# Standard QWERTY positions
QWERTY_POSITIONS = [
    'Q', 'W', 'E', 'R',  # Top row left
    'A', 'S', 'D', 'F',  # Home row left  
    'Z', 'X', 'C', 'V',  # Bottom row left
    'U', 'I', 'O', 'P',  # Top row right
    'J', 'K', 'L', ';',  # Home row right
    'M', ',', '.', '/',  # Bottom row right
    'T', 'G', 'B',       # Center columns
    'Y', 'H', 'N',       # Center columns
    '[', "'"             # Additional keys
]

class ComprehensiveScorer:
    """Comprehensive layout scoring with full matrix of scoring approaches."""
    
    def __init__(self, 
                 dvorak9_working_dir: str = "../dvorak9-scorer",
                 dvorak9_speed_weights: str = "weights/combinations_weights_from_speed_significant.csv",
                 dvorak9_comfort_weights: str = "weights/combinations_weights_from_comfort_significant.csv",
                 engram_working_dir: str = "../optimize_layouts", 
                 engram_config: str = "config.yaml",
                 engram_bigrams: str = "input/letter_pair_frequencies_english.csv",
                 distance_scorer_path: str = "./distance_scorer.py",
                 timeout: int = 180):
        """
        Initialize the comprehensive scorer.
        
        Args:
            dvorak9_working_dir: Working directory for dvorak9_scorer.py
            dvorak9_speed_weights: Path to speed weights (relative to dvorak9_working_dir)
            dvorak9_comfort_weights: Path to comfort weights (relative to dvorak9_working_dir)  
            engram_working_dir: Working directory for score_complete_layout.py
            engram_config: Config file for score_complete_layout.py (relative to engram_working_dir)
            engram_bigrams: Path to bigram frequency CSV (relative to engram_working_dir)
            distance_scorer_path: Path to distance_scorer.py
            timeout: Timeout for subprocess calls in seconds
        """
        self.dvorak9_working_dir = dvorak9_working_dir
        self.dvorak9_path = "dvorak9_scorer.py"
        self.dvorak9_speed_weights = dvorak9_speed_weights
        self.dvorak9_comfort_weights = dvorak9_comfort_weights
        
        self.engram_working_dir = engram_working_dir
        self.engram_path = "score_complete_layout.py"
        self.engram_config = engram_config
        self.engram_bigrams = engram_bigrams
        
        self.distance_scorer_path = distance_scorer_path
        
        self.timeout = timeout
        
        # Check availability
        self.has_dvorak9 = os.path.exists(os.path.join(dvorak9_working_dir, self.dvorak9_path))
        self.has_dvorak9_speed_weights = os.path.exists(os.path.join(dvorak9_working_dir, dvorak9_speed_weights))
        self.has_dvorak9_comfort_weights = os.path.exists(os.path.join(dvorak9_working_dir, dvorak9_comfort_weights))
        self.has_engram = os.path.exists(os.path.join(engram_working_dir, self.engram_path))
        self.has_engram_config = os.path.exists(os.path.join(engram_working_dir, engram_config))
        self.has_engram_bigrams = os.path.exists(os.path.join(engram_working_dir, engram_bigrams))
        self.has_distance_scorer = os.path.exists(distance_scorer_path)
        
        print(f"Scorer availability:")
        print(f"  Dvorak-9: {'✓' if self.has_dvorak9 else '✗'} ({os.path.join(dvorak9_working_dir, self.dvorak9_path)})")
        print(f"  Dvorak-9 speed weights:   {'✓' if self.has_dvorak9_speed_weights else '✗'} ({os.path.join(dvorak9_working_dir, dvorak9_speed_weights)})")
        print(f"  Dvorak-9 comfort weights: {'✓' if self.has_dvorak9_comfort_weights else '✗'} ({os.path.join(dvorak9_working_dir, dvorak9_comfort_weights)})")
        print(f"  Engram:   {'✓' if self.has_engram else '✗'} ({os.path.join(engram_working_dir, self.engram_path)})")
        print(f"  Engram config:   {'✓' if self.has_engram_config else '✗'} ({os.path.join(engram_working_dir, engram_config)})")
        print(f"  Engram bigrams:      {'✓' if self.has_engram_bigrams else '✗'} ({os.path.join(engram_working_dir, engram_bigrams)})")
        print(f"  Distance scorer: {'✓' if self.has_distance_scorer else '✗'} ({distance_scorer_path})")

    def create_layout_mapping(self, layout_chars: List[str]) -> Dict[str, str]:
        """Create mapping from characters to QWERTY positions."""
        char_to_pos = {}
        for i, char in enumerate(layout_chars):
            if i < len(QWERTY_POSITIONS):
                char_to_pos[char.upper()] = QWERTY_POSITIONS[i]
        return char_to_pos
    
    def get_common_characters(self, layouts: Dict[str, List[str]]) -> str:
        """Get characters that appear in all specified layouts."""
        all_chars = None
        for layout_chars in layouts.values():
            layout_set = set(char.upper() for char in layout_chars if char.isalpha())
            if all_chars is None:
                all_chars = layout_set
            else:
                all_chars &= layout_set
        
        # Sort by frequency (convert to uppercase)
        char_priority = "ETAOINSRHLDCUMFPGWYBVKXJQZ"
        common_chars = []
        for char in char_priority:
            if char in all_chars:
                common_chars.append(char)
        
        # Add remaining
        for char in sorted(all_chars):
            if char not in common_chars:
                common_chars.append(char)
        
        return ''.join(common_chars)
    
    def clean_text(self, text: str) -> str:
        """Clean text for scoring by replacing non-ASCII with spaces."""
        # Replace non-ASCII characters with spaces, then normalize whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_bigram_text(self, max_number_of_bigrams: int = 1000000, output_file: str = None) -> str:
        """Generate synthetic text from bigram frequencies."""

        engram_bigrams_path = os.path.join(self.engram_working_dir, self.engram_bigrams)
        
        if not self.has_engram_bigrams:
            print("Warning: Bigram CSV not found, using sample text")
            return "the quick brown fox jumps over the lazy dog"
        
        print(f"Generating bigram text from {engram_bigrams_path}...")
        
        try:
            bigram_freq = pd.read_csv(engram_bigrams_path)
            print(f"Loaded {len(bigram_freq)} bigrams")
        except Exception as e:
            print(f"Error loading bigram frequencies: {e}")
            return "the quick brown fox jumps over the lazy dog"
        
        # Generate text weighted by frequency
        bigram_text_parts = []
        total_bigrams = 0
        
        for _, row in bigram_freq.iterrows():
            bigram = str(row['item_pair']).lower()
            frequency = int(row['score'])
            
            # Filter to only alphabetic bigrams
            if len(bigram) == 2 and all(char.isalpha() for char in bigram):
                repetitions = max(1, frequency // 100000)  # Scale down
                bigram_with_spaces = f" {bigram} "
                bigram_text_parts.extend([bigram_with_spaces] * repetitions)
                total_bigrams += repetitions
                
                if total_bigrams > max_number_of_bigrams:  # Limit size
                    break
        
        bigram_text = "".join(bigram_text_parts)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(bigram_text)
            print(f"Saved bigram text: {output_file}")
        
        print(f"Generated {total_bigrams:,} bigram instances")
        return bigram_text
    
    def load_text_corpus(self, text_dir: str) -> Dict[str, str]:
        """Load all text files from directory."""
        texts = {}
        
        # Generate bigram text
        bigram_text = self.generate_bigram_text()
        texts['Bigrams'] = bigram_text
        
        # Load corpus files
        for text_name, filename in TEXT_SOURCES.items():
            filepath = os.path.join(text_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                texts[text_name] = content
                print(f"Loaded {text_name}: {len(content):,} characters")
            except Exception as e:
                print(f"Warning: Could not load {text_name} from {filename}: {e}")
                texts[text_name] = "sample text for " + text_name  # Fallback
        
        return texts
    
    def run_dvorak9_scorer(self, items: str, positions: str, 
                          weights_csv: Optional[str] = None, 
                          score_type: str = "unweighted") -> Optional[Dict[str, float]]:
        """Run dvorak9_scorer.py from its working directory (text-independent)."""
        if not self.has_dvorak9:
            return None
            
        try:
            # Build command (no text file needed for updated dvorak9_scorer.py)
            cmd = [
                'python3', self.dvorak9_path,
                '--letters', items,
                '--qwerty_keys', positions,  # Note: updated parameter name
                '--ten-scores'
            ]
            
            if weights_csv and os.path.exists(os.path.join(self.dvorak9_working_dir, weights_csv)):
                cmd.extend(['--weights', weights_csv])
            
            print(f"    dvorak9_scorer.py ({score_type}): {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=self.dvorak9_working_dir)
            
            if result.returncode != 0:
                print(f"Error running dvorak9_scorer.py ({score_type}): {result.stderr}")
                return None
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            scores_line = None
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line[0] == '-'):
                    scores_line = line
                    break
            
            if not scores_line:
                return None
            
            try:
                score_values = [float(x) for x in scores_line.split()]
            except Exception as e:
                print(f"Failed to parse dvorak9 scores ({score_type}): {e}")
                return None
            
            # Map to metric names with score type prefix
            metric_names = [
                f'dvorak9_{score_type}_total',
                f'dvorak9_{score_type}_hands', 
                f'dvorak9_{score_type}_fingers', 
                f'dvorak9_{score_type}_skip_fingers', 
                f'dvorak9_{score_type}_dont_cross_home',
                f'dvorak9_{score_type}_same_row', 
                f'dvorak9_{score_type}_home_row', 
                f'dvorak9_{score_type}_columns', 
                f'dvorak9_{score_type}_strum', 
                f'dvorak9_{score_type}_strong_fingers'
            ]
            
            scores = {}
            for i, metric in enumerate(metric_names):
                if i < len(score_values):
                    scores[metric] = score_values[i]
            
            return scores
            
        except Exception as e:
            print(f"Error running dvorak9_scorer.py ({score_type}): {e}")
            return None
    
    def run_engram_scorer(self, items: str, positions: str) -> Optional[Dict[str, float]]:
        """Run score_complete_layout.py from optimize_layouts directory (text-independent)."""
        if not self.has_engram:
            return None
            
        try:
            cmd = [
                'python3', self.engram_path,
                '--items', items,
                '--positions', positions,
                '--config', self.engram_config
            ]

            print(f"    {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=self.engram_working_dir)
            
            if result.returncode != 0:
                print(f"Error running score_complete_layout.py: {result.stderr}")
                return None
            
            # Parse CSV output (look for the CSV section)
            lines = result.stdout.strip().split('\n')
            csv_start = -1
            for i, line in enumerate(lines):
                if line.startswith('total_score,item_score,item_pair_score'):
                    csv_start = i
                    break
            
            if csv_start == -1 or csv_start + 1 >= len(lines):
                print("Could not find CSV output")
                return None
            
            data_line = lines[csv_start + 1]
            try:
                total_score, item_score, item_pair_score = map(float, data_line.split(','))
                
                is_debug = True
                if is_debug:
                    print(f"        Engram total: {total_score}, item: {item_score}, item_pair: {item_pair_score}")
                
                return {
                    'engram_total': total_score,
                    'engram_item': item_score,
                    'engram_item_pair': item_pair_score
                }

            except Exception as e:
                print(f"Failed to parse engram scores: {e}")
                return None
            
        except Exception as e:
            print(f"Error running score_complete_layout.py: {e}")
            return None
    
    def run_distance_scorer(self, items: str, positions: str, text: str, text_name: str) -> Optional[Dict[str, float]]:
        """Run distance_scorer.py with text input (text-dependent)."""
        if not self.has_distance_scorer:
            return None
            
        try:
            # Create temp file for text
            os.makedirs("./output/tmp", exist_ok=True)
            temp_text_path = os.path.join("./output/tmp", f'temp_text_distance_{text_name}.txt')
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            cmd = [
                'python3', self.distance_scorer_path,
                '--letters', items,
                '--qwerty-keys', positions,
                '--text-file', temp_text_path,
                '--csv'
            ]
            
            print(f"    distance_scorer.py: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            # Clean up temp file
            os.unlink(temp_text_path)
            
            if result.returncode != 0:
                print(f"Error running distance_scorer.py: {result.stderr}")
                return None
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2 or not lines[0].startswith('metric,value'):
                print("Could not find CSV output from distance_scorer.py")
                return None
            
            scores = {}
            for line in lines[1:]:
                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        metric, value = parts
                        try:
                            scores[f'distance_{metric}'] = float(value)
                        except ValueError:
                            scores[f'distance_{metric}'] = value  # Keep as string for non-numeric
            
            return scores
            
        except Exception as e:
            print(f"Error running distance_scorer.py: {e}")
            return None
    
    def score_layout_comprehensive(self, items: str, positions: str, text: str, text_name: str) -> Dict[str, float]:
        """Score layout using all available scoring configurations."""
        all_scores = {}
        
        print(f"    Scoring with all configurations...")
        
        # Text-independent scoring (run once per layout)
        if text_name == list(TEXT_SOURCES.keys())[0] or text_name == "Bigrams":  # Only run once per layout
            
            # Engram scoring
            if self.has_engram:
                engram_scores = self.run_engram_scorer(items, positions)
                if engram_scores:
                    all_scores.update(engram_scores)
            
            # Dvorak-9 unweighted
            if self.has_dvorak9:
                unweighted_scores = self.run_dvorak9_scorer(items, positions, None, "unweighted")
                if unweighted_scores:
                    all_scores.update(unweighted_scores)
            
            # Dvorak-9 speed weights
            if self.has_dvorak9 and self.has_dvorak9_speed_weights:
                speed_scores = self.run_dvorak9_scorer(items, positions, self.dvorak9_speed_weights, "speed")
                if speed_scores:
                    all_scores.update(speed_scores)
            
            # Dvorak-9 comfort weights  
            if self.has_dvorak9 and self.has_dvorak9_comfort_weights:
                comfort_scores = self.run_dvorak9_scorer(items, positions, self.dvorak9_comfort_weights, "comfort")
                if comfort_scores:
                    all_scores.update(comfort_scores)
        
        # Text-dependent scoring (run for each text)
        if self.has_distance_scorer:
            distance_scores = self.run_distance_scorer(items, positions, text, text_name)
            if distance_scores:
                all_scores.update(distance_scores)
        
        return all_scores
    
    def compare_all_layouts(self, texts: Dict[str, str], layouts: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Run comprehensive comparison across all layouts and texts."""
        if layouts is None:
            layouts = LAYOUTS
        
        print(f"Starting comprehensive comparison:")
        print(f"  Layouts: {len(layouts)}")
        print(f"  Text sources: {len(texts)}")
        print(f"  Total combinations: {len(layouts) * len(texts)}")
        
        # Get common characters
        common_chars = self.get_common_characters(layouts)
        print(f"  Common characters: {len(common_chars)} ({common_chars})")
        
        results = []
        text_independent_scores = {}  # Cache text-independent scores
        
        for layout_idx, (layout_name, layout_chars) in enumerate(layouts.items()):
            print(f"\n[{layout_idx+1}/{len(layouts)}] Processing {layout_name}...")
            
            # Create character mapping
            char_to_pos = self.create_layout_mapping(layout_chars)
            
            # Build filtered items and positions
            items_filtered = ""
            positions_filtered = ""
            for char in common_chars:
                if char in char_to_pos:
                    items_filtered += char
                    positions_filtered += char_to_pos[char]
            
            print(f"  Using {len(items_filtered)} characters")
            
            # Score on each text
            for text_idx, (text_name, text_content) in enumerate(texts.items()):
                print(f"  [{text_idx+1}/{len(texts)}] {text_name}...")
                
                # Clean text
                clean_text = self.clean_text(text_content)
                if not clean_text:
                    print(f"    Skipping - no valid text")
                    continue
                
                # Get comprehensive scores
                scores = self.score_layout_comprehensive(items_filtered, positions_filtered, clean_text, text_name)
                
                # For the first text, cache text-independent scores
                if text_idx == 0:
                    text_independent_scores[layout_name] = {
                        k: v for k, v in scores.items() 
                        if not k.startswith('distance_')
                    }
                else:
                    # For subsequent texts, add cached text-independent scores
                    scores.update(text_independent_scores[layout_name])
                
                if scores:
                    result = {
                        'layout': layout_name,
                        'text': text_name,
                        **scores
                    }
                    results.append(result)
                    print(f"    Scored: {len(scores)} metrics")
                else:
                    print(f"    Failed to get scores")
        
        if results:
            df = pd.DataFrame(results)
            print(f"\nGenerated {len(df)} total score records")
            return df
        else:
            print("No results obtained!")
            return pd.DataFrame()

def create_analysis_tables(df: pd.DataFrame, output_dir: str):
    """Create comprehensive analysis tables."""
    if df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get score columns grouped by type
    engram_cols = [col for col in df.columns if col.startswith('engram_')]
    dvorak9_unweighted_cols = [col for col in df.columns if 'unweighted' in col]
    dvorak9_speed_cols = [col for col in df.columns if 'speed' in col]
    dvorak9_comfort_cols = [col for col in df.columns if 'comfort' in col]
    distance_cols = [col for col in df.columns if col.startswith('distance_')]
    
    score_groups = {
        'engram': engram_cols,
        'dvorak9_unweighted': dvorak9_unweighted_cols,
        'dvorak9_speed': dvorak9_speed_cols,
        'dvorak9_comfort': dvorak9_comfort_cols,
        'distance': distance_cols
    }
    
    print(f"\nGenerating analysis tables...")
    
    # 1. Summary tables by scoring system
    for group_name, cols in score_groups.items():
        if not cols:
            continue
            
        print(f"  Creating {group_name} summary tables...")
        
        for col in cols:
            if col in df.columns:
                # Pivot table: layouts as rows, texts as columns
                summary = df.pivot(index='layout', columns='text', values=col)
                summary['average'] = summary.mean(axis=1, skipna=True)
                summary = summary.sort_values('average', ascending=False)
                
                # Save
                filename = f"{output_dir}/{group_name}_{col.replace(group_name+'_', '')}.csv"
                summary.to_csv(filename)
                print(f"    Saved: {filename}")
    
    # 2. Cross-correlations between scoring systems
    print(f"  Creating correlation analysis...")
    
    # Primary metrics correlation
    primary_metrics = []
    if 'engram_total' in df.columns:
        primary_metrics.append('engram_total')
    if 'dvorak9_unweighted_total' in df.columns:
        primary_metrics.append('dvorak9_unweighted_total')
    if 'dvorak9_speed_total' in df.columns:
        primary_metrics.append('dvorak9_speed_total')
    if 'dvorak9_comfort_total' in df.columns:
        primary_metrics.append('dvorak9_comfort_total')
    if 'distance_normalized_score' in df.columns:
        primary_metrics.append('distance_normalized_score')
    
    if len(primary_metrics) > 1:
        corr_matrix = df[primary_metrics].corr()
        corr_matrix.to_csv(f"{output_dir}/primary_metrics_correlations.csv")
        print(f"    Saved: {output_dir}/primary_metrics_correlations.csv")
    
    # 3. Full detailed results
    df.to_csv(f"{output_dir}/comprehensive_results_detailed.csv", index=False)
    print(f"    Saved: {output_dir}/comprehensive_results_detailed.csv")
    
    # 4. Layout rankings by different metrics
    rankings = {}
    for col in primary_metrics:
        if col in df.columns:
            # Average across all texts for each layout
            layout_averages = df.groupby('layout')[col].mean().sort_values(ascending=False)
            rankings[col] = layout_averages
    
    if rankings:
        rankings_df = pd.DataFrame(rankings)
        rankings_df.to_csv(f"{output_dir}/layout_rankings_by_metric.csv")
        print(f"    Saved: {output_dir}/layout_rankings_by_metric.csv")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive keyboard layout comparison with full scoring matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Full comprehensive analysis
  python compare_layouts.py --text-dir ../text_data/samples --output output/
  
  # Quick test with sample texts
  python compare_layouts.py --sample-mode --output-prefix quick_test
  
  # Specific layouts only
  python compare_layouts.py --text-dir ../text_data/samples --layouts "QWERTY,Dvorak,Colemak" --output small_test/
        """
    )
    
    # Input options
    parser.add_argument('--text-dir', default='../text_data/samples',
                       help='Directory containing text corpus files')
    parser.add_argument('--sample-mode', action='store_true',
                       help='Use sample texts instead of full corpus (for testing)')
    
    # Layout selection
    parser.add_argument('--layouts',
                       help='Comma-separated list of layout names (default: all)')
    
    parser.add_argument('--dvorak9-working-dir', default='../dvorak9-scorer')
    parser.add_argument('--dvorak9-speed-weights', default='weights/combinations_weights_from_speed_significant.csv')
    parser.add_argument('--dvorak9-comfort-weights', default='weights/combinations_weights_from_comfort_significant.csv')
    parser.add_argument('--engram-working-dir', default='../optimize_layouts')
    parser.add_argument('--engram-config', default='config.yaml')
    parser.add_argument('--engram-bigrams', default='input/letter_pair_frequencies_english.csv')
    parser.add_argument('--distance-scorer-path', default='./distance_scorer.py')
    
    # Output options
    parser.add_argument('--output', default='./output/',
                       help='Output directory for analysis tables')
    parser.add_argument('--output-prefix', 
                       help='Prefix for output files (when not using --output)')
    
    args = parser.parse_args()
    
    try:
        # Determine output directory
        if args.output_prefix:
            output_dir = f"{args.output_prefix}_results"
        else:
            output_dir = args.output
        
        # Initialize scorer
        scorer = ComprehensiveScorer(
            dvorak9_working_dir=args.dvorak9_working_dir,
            dvorak9_speed_weights=args.dvorak9_speed_weights,
            dvorak9_comfort_weights=args.dvorak9_comfort_weights,
            engram_working_dir=args.engram_working_dir,
            engram_config=args.engram_config,
            engram_bigrams=args.engram_bigrams,
            distance_scorer_path=args.distance_scorer_path
        )
        
        # Load texts
        if args.sample_mode:
            print("Using sample mode...")
            texts = {
                'Sample': "the quick brown fox jumps over the lazy dog",
                'Bigrams': scorer.generate_bigram_text()
            }
        elif args.text_dir:
            texts = scorer.load_text_corpus(args.text_dir)
        else:
            print("Error: Must specify --text-dir or --sample-mode")
            return 1
        
        # Select layouts
        layouts = LAYOUTS
        if args.layouts:
            layout_names = [name.strip() for name in args.layouts.split(',')]
            layouts = {name: LAYOUTS[name] for name in layout_names if name in LAYOUTS}
            
            missing = [name for name in layout_names if name not in LAYOUTS]
            if missing:
                print(f"Warning: Unknown layouts: {missing}")
        
        print(f"\nStarting comprehensive comparison...")
        print(f"Selected layouts: {list(layouts.keys())}")
        print(f"Text sources: {list(texts.keys())}")
        
        # Run comprehensive comparison
        results_df = scorer.compare_all_layouts(texts, layouts)
        
        if results_df.empty:
            print("Error: No results obtained")
            return 1
        
        # Generate analysis tables
        create_analysis_tables(results_df, output_dir)
        
        # Print summary
        score_cols = [col for col in results_df.columns if col not in ['layout', 'text']]
        print(f"\nComparison Complete!")
        print(f"=" * 50)
        print(f"Layouts compared: {len(layouts)}")
        print(f"Text sources: {len(texts)}")
        print(f"Scoring metrics: {len(score_cols)}")
        print(f"Total score records: {len(results_df)}")
        print(f"Output directory: {output_dir}")
        
        # Show available metrics
        print(f"\nScoring metrics calculated:")
        metric_groups = {
            'Engram': [col for col in score_cols if col.startswith('engram_')],
            'Dvorak-9 Unweighted': [col for col in score_cols if 'unweighted' in col],
            'Dvorak-9 Speed': [col for col in score_cols if 'speed' in col and 'unweighted' not in col],
            'Dvorak-9 Comfort': [col for col in score_cols if 'comfort' in col],
            'Distance': [col for col in score_cols if col.startswith('distance_')],
        }
        
        for group_name, group_cols in metric_groups.items():
            if group_cols:
                print(f"  {group_name}: {len(group_cols)} metrics")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())