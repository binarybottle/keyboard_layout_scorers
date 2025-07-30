#!/usr/bin/env python3
"""
Dvorak-9 empirical scoring model for keyboard layout evaluation.

This script implements the 9 evaluation criteria derived from Dvorak's "Typing Behavior" 
book and patent (1936) with empirical weights based on analysis of real typing performance data.

The 9 scoring criteria for typing bigrams are (0-1, higher = better performance):
1. Hands - favor alternating hands over same hand
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
4. Don't cross home - avoid crossing over the home row (hurdling)
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls (same hand)
9. Strong fingers - favor stronger fingers over weaker ones

Three scoring approaches are provided:
- Pure Dvorak: Unweighted average of 9 criteria (theoretical baseline)
- Frequency-Weighted: English bigram frequency-weighted average of 9 criteria
- Empirically-Weighted: Frequency-weighted with empirical combination weights (speed/comfort optimized)

Layout scoring uses English bigram frequencies and precomputed Dvorak-9 scores for all 
32×32 QWERTY key-pair combinations, enabling fast frequency-weighted evaluation.

Requires precomputed key-pair scores from generate_key_pair_scores.py.
Optionally uses empirical combination weights for speed or comfort optimization.

Usage:
qwerty_layout = "qwertyuiopasdfghjkl;zxcvbnm,./"
dvorak_layout = "',.pyfgcrlaoeuidhtns;qjkxbmwvz"

# Basic scoring (shows all three approaches)
python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ"

# Speed-weighted scoring
python dvorak9_scorer.py --letters qwerty_layout --qwerty_keys qwerty_layout \
  --weights "input/dvorak9/speed_weights.csv"

# Comfort-weighted scoring
python dvorak9_scorer.py --letters qwerty_layout --qwerty_keys qwerty_layout \
  --weights "input/dvorak9/comfort_weights.csv"

# Custom frequency file
python dvorak9_scorer.py --letters qwerty_layout --qwerty_keys qwerty_layout \
  --frequency_csv "input/engram/normalized_letter_pair_frequencies_en.csv"

# CSV output (clean CSV data only)
python dvorak9_scorer.py --letters qwerty_layout --qwerty_keys qwerty_layout --csv

# Just ten scores output (average score + 9 individual scores)
python dvorak9_scorer.py --letters qwerty_layout --qwerty_keys qwerty_layout --ten_scores

"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# QWERTY keyboard layout with (row, finger, hand) mapping
# Rows: 0=number, 1=top, 2=home, 3=bottom
# Fingers: 1=index, 2=middle, 3=ring, 4=pinky  
# Hands: L=left, R=right
QWERTY_LAYOUT = {
    # Number row (row 0)
    '1': (0, 4, 'L'), '2': (0, 3, 'L'), '3': (0, 2, 'L'), '4': (0, 1, 'L'), '5': (0, 1, 'L'),
    '6': (0, 1, 'R'), '7': (0, 1, 'R'), '8': (0, 2, 'R'), '9': (0, 3, 'R'), '0': (0, 4, 'R'),
    
    # Top row (row 1)
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    
    # Home row (row 2) 
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    
    # Bottom row (row 3)
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    
    # Additional common keys
    "'": (2, 4, 'R'), '[': (1, 4, 'R'), #, ']': (1, 4, 'R'), '\\': (1, 4, 'R'),
    #'-': (0, 4, 'R'), '=': (0, 4, 'R'),
}

# Define finger strength (1=index, 2=middle are strong; 3=ring, 4=pinky are weak)
STRONG_FINGERS = {1, 2}
WEAK_FINGERS = {3, 4}

# Define home row
HOME_ROW = 2

# Define finger column assignments for detecting lateral movement
FINGER_COLUMNS = {
    # Left hand columns (4=leftmost to 1=rightmost)
    'L': {
        4: ['Q', 'A', 'Z'],                                 # Pinky column
        3: ['W', 'S', 'X'],                                 # Ring column  
        2: ['E', 'D', 'C'],                                 # Middle column
        1: ['R', 'F', 'V']                                  # Index column
    },
    # Right hand columns (1=leftmost to 4=rightmost)  
    'R': {
        1: ['U', 'J', 'M'],                                 # Index column
        2: ['I', 'K', ','],                                 # Middle column
        3: ['O', 'L', '.'],                                 # Ring column
        4: ['P', ';', '/']                                  # Pinky column
    }
}

def load_bigram_frequencies(csv_path: str = "input/engram/normalized_letter_pair_frequencies_en.csv", quiet: bool = False) -> Tuple[List[str], List[float]]:
    """
    Load normalized bigram frequencies from CSV file.
    
    Args:
        csv_path: Path to CSV file with bigram and frequency columns
        quiet: If True, suppress informational output
        
    Returns:
        Tuple of (bigrams, frequencies) lists in the same order
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Bigram frequencies file not found: {csv_path}")
    
    bigrams = []
    frequencies = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Get the available columns
        if not reader.fieldnames:
            raise ValueError(f"No columns found in CSV file: {csv_path}")
        
        columns = reader.fieldnames
        #if not quiet:
        #    print(f"Available columns in {csv_path}: {columns}")
        
        # Try to find bigram column (common names)
        bigram_col = None
        for col_name in ['bigram', 'letter_pair', 'pair', 'sequence', 'letters']:
            if col_name in columns:
                bigram_col = col_name
                break
        
        if bigram_col is None:
            raise ValueError(f"Could not find bigram column. Available columns: {columns}. "
                           f"Expected one of: bigram, letter_pair, pair, sequence, letters")
        
        # Try to find frequency column (common names)
        freq_col = None
        for col_name in ['frequency', 'freq', 'probability', 'prob', 'weight', 'count']:
            if col_name in columns:
                freq_col = col_name
                break
        
        if freq_col is None:
            raise ValueError(f"Could not find frequency column. Available columns: {columns}. "
                           f"Expected one of: frequency, freq, probability, prob, weight, count")

        #if not quiet:
        #    print(f"Using columns: bigram='{bigram_col}', frequency='{freq_col}'")

        for row in reader:
            bigram_str = row[bigram_col].strip().lower()
            freq_str = row[freq_col].strip()
            
            # Skip empty rows
            if not bigram_str or not freq_str:
                continue
                
            try:
                frequency = float(freq_str)
                bigrams.append(bigram_str)
                frequencies.append(frequency)
            except ValueError:
                if not quiet:
                    print(f"Warning: Could not parse frequency '{freq_str}' for bigram '{bigram_str}', skipping")
                continue
    
    if not bigrams:
        raise ValueError(f"No valid bigram-frequency pairs found in {csv_path}")
    
    if not quiet:
        print(f"Loaded {len(bigrams)} bigram frequencies from {csv_path}")
    return bigrams, frequencies

def load_key_pair_scores(csv_path: str = "input/dvorak9/key_pair_scores.csv", quiet: bool = False) -> Dict[str, float]:
    """
    Load precomputed Dvorak-9 scores for key-pairs.
    
    Args:
        csv_path: Path to CSV file with columns 'key_pair' and 'dvorak9_score'
        quiet: If True, suppress informational output
        
    Returns:
        Dict mapping key-pairs (e.g., "QW") to their Dvorak-9 scores
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"key-pair scores file not found: {csv_path}. "
                               f"Please run generate_key_pair_scores.py first.")
    
    key_pair_scores = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key_pair_scores[row['key_pair']] = float(row['dvorak9_score'])
    
    if not quiet:
        print(f"Loaded {len(key_pair_scores)} precomputed key-pair scores from {csv_path}")
    return key_pair_scores

def load_combination_weights(csv_path: Optional[str] = None, quiet: bool = False) -> Optional[Dict[Tuple[str, ...], float]]:
    """
    Load empirical correlation weights for different feature combinations from CSV file.
    
    Args:
        csv_path: Path to CSV file containing combination correlations (optional)
        quiet: If True, suppress informational output
        
    Returns:
        Dict mapping combination tuples to correlation values, or None if no file provided
        
    The CSV should have 'combination' and 'correlation' columns.
    These correlations are derived from analysis of 136M+ keystroke dataset
    with FDR correction applied to 529 statistical tests.
    """
    if csv_path is None:
        return None
        
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Combination weights file not found: {csv_path}")
    
    combination_weights: Dict[Tuple[str, ...], float] = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse combination name
                combination_str = row.get('combination', '').strip()
                
                # Get correlation value
                correlation_str = row.get('correlation', '').strip()
                if not correlation_str:
                    continue  # Skip rows without correlation values
                
                try:
                    correlation = float(correlation_str)
                except ValueError:
                    continue  # Skip rows with invalid correlation values
                
                # Convert combination string to tuple
                if combination_str.lower() in ['none', 'empty', '']:
                    combination = ()
                else:
                    # For individual features, just use the feature name
                    if '+' in combination_str:
                        # Multi-feature combination like "fingers+same_row+strum"
                        features = [f.strip() for f in combination_str.split('+')]
                        combination = tuple(sorted(features))
                    else:
                        # Single feature like "fingers"
                        combination = (combination_str,)
                
                combination_weights[combination] = correlation
                
    except Exception as e:
        raise ValueError(f"Error parsing combination weights CSV: {e}")
    
    # Ensure we have at least an empty combination
    if () not in combination_weights:
        combination_weights[()] = 0.0
    
    return combination_weights

def get_key_info(key: str):
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key]
    else:
        return None

def is_finger_in_column(key: str, finger: int, hand: str) -> bool:
    """Check if a key is in the designated column for a finger."""
    key = key.upper()
    if hand in FINGER_COLUMNS and finger in FINGER_COLUMNS[hand]:
        return key in FINGER_COLUMNS[hand][finger]
    return False

def score_bigram_dvorak9(bigram: str) -> Dict[str, float]:
    """
    Calculate all 9 Dvorak criteria scores for a bigram.
        
    Args:
        bigram: Two-character string (e.g., "th", "er")
        
    Returns:
        Dict with keys: hands, fingers, skip_fingers, dont_cross_home, 
                        same_row, home_row, columns, strum, strong_fingers
        Values are 0-1 where higher = better for typing speed according to Dvorak principles
    """
    if len(bigram) != 2:
        raise ValueError("Bigram must be exactly 2 characters long")    
    
    char1, char2 = bigram[0].upper(), bigram[1].upper()
    
    # Get key information
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    
    if key_info1 is None or key_info2 is None:
        return {criterion: 0.5 for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                                                'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']}
    
    row1, finger1, hand1 = key_info1
    row2, finger2, hand2 = key_info2
    
    scores = {}
    
    # 1. Hands - favor alternating hands
    scores['hands'] = 1.0 if hand1 != hand2 else 0.0
    
    # 2. Fingers - avoid same finger repetition
    if hand1 != hand2:
        scores['fingers'] = 1.0  # Different hands = different fingers
    else:
        scores['fingers'] = 0.0 if finger1 == finger2 else 1.0
    
    # 3. Skip fingers - favor skipping more fingers (same hand only)
    if hand1 != hand2:
        scores['skip_fingers'] = 1.0      # Different hands is good
    elif finger1 == finger2:
        scores['skip_fingers'] = 0.0      # Same finger is bad
    else:
        finger_gap = abs(finger1 - finger2)
        if finger_gap == 1:
            scores['skip_fingers'] = 0    # Adjacent fingers is bad
        elif finger_gap == 2:
            scores['skip_fingers'] = 0.5  # Skipping 1 finger is good
        elif finger_gap == 3:
            scores['skip_fingers'] = 1.0  # Skipping 2 fingers is better
    
    # 4. Don't cross home - avoid hurdling over home row
    if hand1 != hand2:
        scores['dont_cross_home'] = 1.0  # Different hands always score well
    else:
        # Check for hurdling (top to bottom or bottom to top, skipping home)
        if (row1 == 1 and row2 == 3) or (row1 == 3 and row2 == 1):
            scores['dont_cross_home'] = 0.0  # Hurdling over home row
        else:
            scores['dont_cross_home'] = 1.0  # No hurdling
    
    # 5. Same row - favor staying in same row
    scores['same_row'] = 1.0 if row1 == row2 else 0.0
    
    # 6. Home row - favor using home row
    home_count = sum(1 for row in [row1, row2] if row == HOME_ROW)
    if home_count == 2:
        scores['home_row'] = 1.0      # Both in home row
    elif home_count == 1:
        scores['home_row'] = 0.5      # One in home row
    else:
        scores['home_row'] = 0.0      # Neither in home row
    
    # 7. Columns - favor fingers staying in their designated columns
    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    
    if in_column1 and in_column2:
        scores['columns'] = 1.0       # Both in correct columns
    elif in_column1 or in_column2:
        scores['columns'] = 0.5       # One in correct column
    else:
        scores['columns'] = 0.0       # Neither in correct column
    
    # 8. Strum - favor inward rolls (outer to inner fingers)
    if hand1 != hand2:
        scores['strum'] = 1.0         # Different hands get full score
    elif finger1 == finger2:
        scores['strum'] = 0.0         # Same finger gets zero
    else:
        # Inward roll: from higher finger number to lower (4→3→2→1)
        # This represents rolling from pinky toward index finger
        if finger1 > finger2:
            scores['strum'] = 1.0     # Inward roll (e.g., pinky to ring, ring to middle)
        else:
            scores['strum'] = 0.0     # Outward roll (e.g., index to middle, middle to ring)
    
    # 9. Strong fingers - favor index and middle fingers
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['strong_fingers'] = 1.0    # Both strong fingers
    elif strong_count == 1:
        scores['strong_fingers'] = 0.5    # One strong finger
    else:
        scores['strong_fingers'] = 0.0    # Both weak fingers
    
    return scores

def identify_bigram_combination(bigram_scores: Dict[str, float], threshold: float = 0) -> Tuple[str, ...]:
    """
    Identify which feature combination a bigram exhibits.
    
    Args:
        bigram_scores: Dict of feature scores for a single bigram
        threshold: Minimum score to consider a feature "active"
    
    Returns:
        Tuple of active feature names, sorted for consistent lookup
    """
    active_features = []
    
    for feature, score in bigram_scores.items():
        if score >= threshold:
            active_features.append(feature)
    
    return tuple(sorted(active_features))

def score_bigram_weighted(bigram_scores: Dict[str, float], 
                         combination_weights: Dict[Tuple[str, ...], float]) -> float:
    """
    Score a single bigram using empirical combination-specific weights.
    
    Args:
        bigram_scores: Dict of 9 feature scores for the bigram
        combination_weights: Dict mapping combinations to empirical weights
    
    Returns:
        Weighted score for this bigram (negative = good for typing speed)
    """
    # Identify which combination this bigram exhibits
    combination = identify_bigram_combination(bigram_scores)
    
    # Try to find exact match first
    if combination in combination_weights:
        weight = combination_weights[combination]
        # Calculate combination strength (how well does bigram exhibit this combination)
        combination_strength = sum(bigram_scores[feature] for feature in combination) / len(combination) if combination else 0
        return weight * combination_strength
    
    # Fall back to best partial match
    best_score = 0.0
    
    for combo, weight in combination_weights.items():
        if not combo:  # Skip empty combination
            continue
            
        # Check if this bigram exhibits this combination (partial match allowed)
        if all(feature in bigram_scores for feature in combo):
            combo_strength = sum(bigram_scores[feature] for feature in combo) / len(combo)
            
            # Penalize partial matches
            match_completeness = len(set(combo) & set(identify_bigram_combination(bigram_scores))) / len(combo)
            adjusted_score = weight * combo_strength * match_completeness
            
            if abs(adjusted_score) > abs(best_score):
                best_score = adjusted_score
    
    return best_score

class Dvorak9Scorer:
    """
    Dvorak-9 scorer using frequency-weighted scoring with optional empirical combination weights.
    
    Implements the 9 evaluation criteria derived from Dvorak's work with
    frequency-weighted averaging based on English bigram frequencies and
    optional empirical weights derived from analysis of real typing performance data.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], 
             frequency_csv: str = "input/engram/normalized_letter_pair_frequencies_en.csv",
             key_pair_scores_csv: str = "input/dvorak9/key_pair_scores.csv",
             weights: Optional[str] = None,
             quiet: bool = False):
        """
        Initialize scorer with layout mapping and data files.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions (e.g., {'a': 'F', 'b': 'D'})
            frequency_csv: Path to CSV file containing English bigram frequencies
            key_pair_scores_csv: Path to CSV file containing precomputed key-pair Dvorak-9 scores
            weights: Path to CSV file containing empirical combination weights (optional)
            quiet: If True, suppress informational output
        """
        self.layout_mapping = layout_mapping
        
        # Load bigram frequencies
        self.bigrams, self.frequencies = load_bigram_frequencies(frequency_csv, quiet=quiet)
        
        # Load precomputed key-pair scores
        self.key_pair_scores = load_key_pair_scores(key_pair_scores_csv, quiet=quiet)
        
        # Load weights if provided, otherwise use None for unweighted scoring
        if weights:
            self.combination_weights = load_combination_weights(weights, quiet=quiet)
            # Determine weights type based on filename for proper sign handling
            self.weights_type = self._determine_weights_type(weights)
        else:
            self.combination_weights = None
            self.weights_type = None

    def _determine_weights_type(self, weights: str) -> str:
        """Determine if weights are speed-based or comfort-based from filename."""
        filename = weights.lower()
        if 'speed' in filename:
            return 'speed'
        elif 'comfort' in filename:
            return 'comfort'
        else:
            # Try to determine from the actual correlations in the file
            try:
                with open(weights, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    correlations = []
                    for row in reader:
                        try:
                            corr = float(row.get('correlation', '0'))
                            correlations.append(corr)
                        except ValueError:
                            continue
                    
                    if correlations:
                        avg_corr = sum(correlations) / len(correlations)
                        # If most correlations are negative, assume speed weights
                        return 'speed' if avg_corr < 0 else 'comfort'
            except:
                pass
            
            # Default fallback
            return 'speed'

    def calculate_scores(self) -> Dict:
        """Calculate layout score using frequency-weighted scoring with optional empirical combination weights."""
        
        if not self.bigrams:
            return {
                'pure_dvorak_score': 0.0,
                'frequency_weighted_score': 0.0,
                'layout_score': 0.0,
                'normalized_score': 0.0,
                'theoretical_maximum': 1.0,
                'bigram_count': 0,
                'individual_scores': {},
                'pure_individual_scores': {},  # Add this
                'scoring_mode': 'unweighted' if self.combination_weights is None else 'weighted',
                'bigram_coverage': 0.0
            }
        
        # Initialize accumulators
        pure_dvorak_sum = 0.0
        frequency_weighted_score = 0.0
        empirically_weighted_score = 0.0
        individual_criterion_sums = defaultdict(float)
        pure_individual_criterion_sums = defaultdict(float)  # Add this
        total_frequency = 0.0
        covered_bigrams = 0
        total_bigrams_for_pure = 0
        
        # Process each bigram in the frequency list
        for bigram, frequency in zip(self.bigrams, self.frequencies):
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            # Check if both characters are in our layout
            if char1 not in self.layout_mapping or char2 not in self.layout_mapping:
                continue
            
            # Map characters to QWERTY positions
            pos1 = self.layout_mapping[char1]
            pos2 = self.layout_mapping[char2]
            key_pair = pos1 + pos2
            
            # Get individual criterion scores for this bigram (always needed)
            bigram_scores = score_bigram_dvorak9(key_pair)
            
            # 1. Pure Dvorak score (unweighted average, no frequency weighting)
            pure_bigram_score = sum(bigram_scores.values()) / len(bigram_scores)
            pure_dvorak_sum += pure_bigram_score
            total_bigrams_for_pure += 1
            
            # Track pure individual criterion scores (unweighted)
            for criterion, score in bigram_scores.items():
                pure_individual_criterion_sums[criterion] += score
            
            # 2. Frequency-weighted Dvorak score (frequency-weighted average of 9 criteria)
            frequency_weighted_score += frequency * pure_bigram_score
            
            # 3. Empirically-weighted score (if weights provided)
            if self.combination_weights is not None:
                weighted_score = score_bigram_weighted(bigram_scores, self.combination_weights)
                
                # Apply correct sign based on weights type
                if self.weights_type == 'speed':
                    final_empirical_score = -weighted_score
                else:  # comfort weights
                    final_empirical_score = weighted_score
                
                empirically_weighted_score += frequency * final_empirical_score
            
            # Accumulate frequency-weighted individual criterion scores
            for criterion, score in bigram_scores.items():
                individual_criterion_sums[criterion] += frequency * score
            
            total_frequency += frequency
            covered_bigrams += 1
        
        # Calculate final scores
        pure_dvorak_score = pure_dvorak_sum / total_bigrams_for_pure if total_bigrams_for_pure > 0 else 0.0
        freq_weighted_dvorak_score = frequency_weighted_score / total_frequency if total_frequency > 0 else 0.0
        
        if self.combination_weights is not None:
            empirical_score = empirically_weighted_score / total_frequency if total_frequency > 0 else 0.0
            primary_score = empirical_score
        else:
            empirical_score = None
            primary_score = freq_weighted_dvorak_score
        
        # Calculate individual criterion averages
        criterion_names = ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                        'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']
        
        # Pure individual scores (unweighted averages)
        pure_individual_scores = {}
        for criterion in criterion_names:
            if total_bigrams_for_pure > 0:
                pure_individual_scores[criterion] = pure_individual_criterion_sums[criterion] / total_bigrams_for_pure
            else:
                pure_individual_scores[criterion] = 0.0
        
        # Frequency-weighted individual scores
        individual_scores = {}
        for criterion in criterion_names:
            if total_frequency > 0:
                individual_scores[criterion] = individual_criterion_sums[criterion] / total_frequency
            else:
                individual_scores[criterion] = 0.0
        
        # Calculate theoretical maximum and normalized score
        if self.combination_weights is not None:
            theoretical_max = self.calculate_theoretical_maximum()
            if theoretical_max > 0:
                raw_ratio = primary_score / theoretical_max
                normalized_score = max(0.0, min(1.0, raw_ratio))
            else:
                normalized_score = max(0.0, min(1.0, primary_score))
        else:
            theoretical_max = 1.0
            normalized_score = max(0.0, min(1.0, primary_score))
        
        # Calculate coverage
        bigram_coverage = covered_bigrams / len(self.bigrams) if self.bigrams else 0.0
        
        result = {
            'pure_dvorak_score': pure_dvorak_score,
            'frequency_weighted_score': freq_weighted_dvorak_score,
            'layout_score': primary_score,
            'normalized_score': normalized_score,
            'theoretical_maximum': theoretical_max,
            'bigram_count': covered_bigrams,
            'total_bigrams': len(self.bigrams),
            'bigram_coverage': bigram_coverage,
            'individual_scores': individual_scores,
            'pure_individual_scores': pure_individual_scores,  # Add this
            'scoring_mode': 'unweighted' if self.combination_weights is None else 'weighted',
            'weights_type': self.weights_type
        }
        
        # Add empirical score if available
        if empirical_score is not None:
            result['empirically_weighted_score'] = empirical_score
        
        return result

    def calculate_theoretical_maximum(self) -> float:
        """Calculate theoretical maximum possible score for current weights type."""
        if self.combination_weights is None:
            return 1.0  # Unweighted maximum is 1.0 (average of all 1.0 scores)
        
        # For weighted scoring, find the highest magnitude correlation
        max_magnitude = 0.0
        
        for combination, weight in self.combination_weights.items():
            if combination:  # Skip empty combination
                # Calculate what this combination would contribute with perfect scores
                weighted_score = weight * 1.0  # All criteria in combination = 1.0
                
                # Apply same sign logic as actual scoring
                if self.weights_type == 'speed':
                    final_score = -weighted_score  # Speed: negative correlation = good, so flip
                else:  # comfort
                    final_score = weighted_score   # Comfort: positive correlation = good, so keep
                
                # Track the highest positive contribution possible
                if final_score > max_magnitude:
                    max_magnitude = final_score
        
        # Return the theoretical maximum
        return max_magnitude if max_magnitude > 0 else 1.0

def print_results(results: Dict) -> None:
    """Print formatted results from scoring."""
    scoring_mode = results.get('scoring_mode', 'unknown')
    weights_type = results.get('weights_type', '')
    
    if scoring_mode == 'weighted':
        weights_desc = f" {weights_type}-based" if weights_type else ""
        print(f"\nDvorak-9 scoring results (with{weights_desc} empirical weights)")
    else:
        print("\nDvorak-9 scoring results")

    print("=" * 70)
    
    # Show all three scoring approaches
    print(f"Pure Dvorak-9 score:        {results['pure_dvorak_score']:8.3f}  (unweighted average)")
    print(f"Frequency-weighted score:   {results['frequency_weighted_score']:8.3f}  (frequency-weighted)")
    
    if 'empirically_weighted_score' in results:
        print(f"Empirically-weighted score: {results['empirically_weighted_score']:8.3f}  ({weights_type}-weighted)")
    
    print(f"Normalized score:           {results.get('normalized_score', 0.0):8.3f}  (0-1 scale)")
    print(f"Theoretical maximum:        {results.get('theoretical_maximum', 1.0):8.3f}")
    
    # Coverage and data info
    print(f"\nCoverage information:")
    print(f"Bigrams covered: {results['bigram_count']:8d} / {results.get('total_bigrams', 0):d}")
    print(f"Coverage:          {results.get('bigram_coverage', 0.0)*100:8.1f}%")
    print(f"Scoring mode:           {scoring_mode}")
    
    # Individual criteria breakdown with all score types
    print(f"\nIndividual criterion scores (0-1 scale):")
    print("-" * 80)
    print(f"{'Criterion':<25} {'Pure':<8} {'Freq-Wt':<8} {'Note'}")
    print("-" * 80)
    
    criteria_info = [
        ('hands', '1. Hands (alternating)'),
        ('fingers', '2. Fingers (avoid same)'),
        ('skip_fingers', '3. Skip fingers'),
        ('dont_cross_home', '4. Don\'t cross home'),
        ('same_row', '5. Same row'),
        ('home_row', '6. Home row'),
        ('columns', '7. Columns'),
        ('strum', '8. Strum (inward rolls)'),
        ('strong_fingers', '9. Strong fingers')
    ]
    
    individual_scores = results.get('individual_scores', {})
    pure_individual_scores = results.get('pure_individual_scores', {})
    
    for key, name in criteria_info:
        pure_score = pure_individual_scores.get(key, 0.0)
        freq_score = individual_scores.get(key, 0.0)
        
        # Show difference between pure and frequency-weighted
        diff = freq_score - pure_score
        if abs(diff) < 0.001:
            note = "same"
        elif diff > 0:
            note = f"+{diff:.3f}"
        else:
            note = f"{diff:.3f}"
        
        print(f"  {name:<23}: {pure_score:6.3f}   {freq_score:6.3f}   {note}")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Dvorak-9 layout scores using frequency-weighted scoring with optional empirical combination weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Unweighted scoring (pure Dvorak theory)
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ"
  
  # Weighted scoring with speed-based weights
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ" \
    --weights "input/dvorak9/speed_weights.csv"
  
  # Weighted scoring with comfort-based weights
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ" \
    --weights "input/dvorak9/comfort_weights.csv"
  
  # Custom frequency file
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ" \
    --frequency_csv "custom_bigram_frequencies.csv"

  # CSV output (clean CSV data only)
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ" --csv

  # Ten scores output (layout score + 9 individual scores)
  python dvorak9_scorer.py --letters "etaoinshrlcu" --qwerty_keys "FDESGJWXRTYZ" --ten_scores
  
"""
    )
    
    parser.add_argument("--letters", required=True,
                       help="String of characters in the layout (e.g., 'etaoinshrlcu')")
    parser.add_argument("--qwerty_keys", required=True,
                       help="String of corresponding QWERTY positions (e.g., 'FDESGJWXRTYZ')")
    parser.add_argument("--frequency-csv", default="input/engram/normalized_letter_pair_frequencies_en.csv",
                       help="Path to CSV file containing bigram frequencies (default: input/engram/normalized_letter_pair_frequencies_en.csv)")
    parser.add_argument("--key-pair-scores-csv", default="input/dvorak9/key_pair_scores.csv",
                       help="Path to CSV file containing precomputed key-pair scores (default: input/dvorak9/key_pair_scores.csv)")
    parser.add_argument("--weights",
                       help="Path to CSV file containing empirical combination weights (optional)")
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format (clean CSV data only)")
    parser.add_argument("--ten-scores", action="store_true",
                       help="Output only 10 scores: layout score followed by 9 individual scores")
    parser.add_argument("--all-scores", action="store_true",
                       help="Output all scores: pure dvorak, frequency-weighted, empirically-weighted (if available), plus 9 individual scores")
    args = parser.parse_args()
    
    try:
        # Determine if we should suppress informational output
        quiet = args.csv
        
        # Validate inputs
        if len(args.letters) != len(args.qwerty_keys):
            if not quiet:
                print(f"Error: Character count ({len(args.letters)}) != Position count ({len(args.qwerty_keys)})")
            return
        
        # Filter to only letters, keeping corresponding positions
        letter_pairs = [(char, pos) for char, pos in zip(args.letters, args.qwerty_keys) if char.isalpha()]
        
        if not letter_pairs:
            if not quiet:
                print("Error: No letters found in --letters")
            return
        
        # Reconstruct filtered strings
        filtered_letters = ''.join(pair[0] for pair in letter_pairs)
        filtered_positions = ''.join(pair[1] for pair in letter_pairs)
        
        # Only show layout information in non-CSV mode
        if not quiet:
            print(f"\nLayout: {filtered_letters} → {filtered_positions}\n")

        # Create layout mapping
        layout_mapping = dict(zip(filtered_letters.lower(), filtered_positions.upper()))

        # Calculate scores
        scorer = Dvorak9Scorer(
            layout_mapping=layout_mapping,
            frequency_csv=args.frequency_csv,
            key_pair_scores_csv=args.key_pair_scores_csv,
            weights=args.weights,
            quiet=quiet
        )
        results = scorer.calculate_scores()

        if args.all_scores:
            # Output all score types + 9 individual scores
            individual_scores = results.get('individual_scores', {})
            
            scores = [
                results['pure_dvorak_score'],
                results['frequency_weighted_score']
            ]
            
            # Add empirical score if available
            if 'empirically_weighted_score' in results:
                scores.append(results['empirically_weighted_score'])
            
            # Add individual scores in consistent order
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                            'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                scores.append(individual_scores.get(criterion, 0.0))
            
            print(' '.join(f"{score:.6f}" for score in scores))

        elif args.ten_scores:
            # Output scores: primary score + 9 individual scores
            individual_scores = results.get('individual_scores', {})
            
            # Use layout_score as the primary score (maintains backward compatibility)
            scores = [results['layout_score']]
            
            # Add individual scores in consistent order
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                            'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                scores.append(individual_scores.get(criterion, 0.0))
            
            print(' '.join(f"{score:.6f}" for score in scores))

        elif args.csv:
            # CSV output with all score types - clean CSV data only
            print("metric,value")
            
            # All score types
            print(f"pure_dvorak_score,{results['pure_dvorak_score']:.6f}")
            print(f"frequency_weighted_score,{results['frequency_weighted_score']:.6f}")
            if 'empirically_weighted_score' in results:
                print(f"empirically_weighted_score,{results['empirically_weighted_score']:.6f}")
            print(f"layout_score,{results['layout_score']:.6f}")
            print(f"normalized_score,{results.get('normalized_score', results['layout_score']):.6f}")
            print(f"theoretical_maximum,{results.get('theoretical_maximum', 1.0):.6f}")
            
            # Metadata
            scoring_mode = results.get('scoring_mode', 'unknown')
            weights_type = results.get('weights_type', '')
            print(f"scoring_mode,{scoring_mode}")
            if weights_type:
                print(f"weights_type,{weights_type}")
            print(f"bigram_count,{results['bigram_count']}")
            print(f"total_bigrams,{results.get('total_bigrams', 0)}")
            print(f"bigram_coverage,{results.get('bigram_coverage', 0.0):.6f}")
            
            # Individual frequency-weighted scores
            individual_scores = results.get('individual_scores', {})
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                            'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                score = individual_scores.get(criterion, 0.0)
                print(f"individual_{criterion},{score:.6f}")

        else:
            # Human-readable output
            print_results(results)
            
        
    except Exception as e:
        if not args.csv:  # Only show error details in non-CSV mode
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()