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

import sys
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

# Import our framework components
from base_scorer import BaseLayoutScorer, ScoreResult
from config_loader import load_scorer_config
from layout_utils import filter_to_letters_only
from data_utils import load_bigram_frequencies, load_key_value_csv, validate_data_consistency
from output_utils import print_results
from cli_utils import create_standard_parser, handle_common_errors, get_layout_from_args


# QWERTY keyboard layout with (row, finger, hand) mapping (same as original)
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
    "'": (2, 4, 'R'), '[': (1, 4, 'R'),
}

# Define finger strength and home row (same as original)
STRONG_FINGERS = {1, 2}
WEAK_FINGERS = {3, 4}
HOME_ROW = 2

# Define finger column assignments (same as original)
FINGER_COLUMNS = {
    'L': {
        4: ['Q', 'A', 'Z'],
        3: ['W', 'S', 'X'],
        2: ['E', 'D', 'C'],
        1: ['R', 'F', 'V']
    },
    'R': {
        1: ['U', 'J', 'M'],
        2: ['I', 'K', ','],
        3: ['O', 'L', '.'],
        4: ['P', ';', '/']
    }
}


def get_key_info(key: str):
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    return QWERTY_LAYOUT.get(key)


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
        if finger1 > finger2:
            scores['strum'] = 1.0     # Inward roll
        else:
            scores['strum'] = 0.0     # Outward roll
    
    # 9. Strong fingers - favor index and middle fingers
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['strong_fingers'] = 1.0    # Both strong fingers
    elif strong_count == 1:
        scores['strong_fingers'] = 0.5    # One strong finger
    else:
        scores['strong_fingers'] = 0.0    # Both weak fingers
    
    return scores


class Dvorak9Scorer(BaseLayoutScorer):
    """
    Dvorak-9 scorer using frequency-weighted scoring with optional empirical combination weights.
    
    Implements the 9 evaluation criteria derived from Dvorak's work with
    frequency-weighted averaging based on English bigram frequencies and
    optional empirical weights derived from analysis of real typing performance data.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dvorak-9 scorer.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions
            config: Optional configuration dictionary
        """
        super().__init__(layout_mapping, config)
        
        # Data containers
        self.bigrams: List[str] = []
        self.frequencies: List[float] = []
        self.key_pair_scores: Dict[str, float] = {}
        self.combination_weights: Optional[Dict[Tuple[str, ...], float]] = None
        
        # Determine weights type
        weights_file = self.config.get('weights_file')
        if weights_file:
            self.weights_type = self._determine_weights_type(weights_file)
        else:
            self.weights_type = None
    
    def _determine_weights_type(self, weights_file: str) -> str:
        """Determine if weights are speed-based or comfort-based from filename."""
        filename = weights_file.lower()
        if 'speed' in filename:
            return 'speed'
        elif 'comfort' in filename:
            return 'comfort'
        else:
            return 'speed'  # Default fallback
    
    def load_data_files(self) -> None:
        """Load required data files for Dvorak-9 scoring."""
        data_files = self.config.get('data_files', {})
        quiet_mode = self.config.get('quiet_mode', False)
        
        # Load bigram frequencies
        frequencies_file = data_files.get('frequencies')
        if frequencies_file and Path(frequencies_file).exists():
            self.bigrams, self.frequencies = load_bigram_frequencies(
                frequencies_file, verbose=not quiet_mode
            )
        else:
            raise FileNotFoundError(f"Bigram frequencies file required: {frequencies_file}")
        
        # Load key-pair scores
        key_pairs_file = data_files.get('key_pair_scores')
        if key_pairs_file and Path(key_pairs_file).exists():
            self.key_pair_scores = load_key_value_csv(
                key_pairs_file,
                key_col='key_pair',
                value_col='dvorak9_score',
                key_transform=None,
                value_transform=float,
                verbose=not quiet_mode
            )
        else:
            if not quiet_mode:
                print("Warning: Key-pair scores file not found, will compute on-demand")
            self.key_pair_scores = {}
        
        # Load combination weights if specified
        weights_file = self.config.get('weights_file')
        if weights_file and Path(weights_file).exists():
            # For now, use a simple approach - in full implementation this would load the weights CSV
            if not quiet_mode:
                print(f"Note: Empirical weights loading not fully implemented yet")
            self.combination_weights = {}
        
        # Validate loaded data
        validation_issues = []
        validation_issues.extend(validate_data_consistency(
            {'bigrams': len(self.bigrams), 'frequencies': len(self.frequencies)}, 
            "frequency data"
        ))
        
        if validation_issues and not quiet_mode:
            print("Data validation warnings:")
            for issue in validation_issues:
                print(f"  {issue}")
    
    def calculate_scores(self) -> ScoreResult:
        """
        Calculate layout score using frequency-weighted scoring with optional empirical combination weights.
        
        Returns:
            ScoreResult containing all scoring approaches and breakdowns
        """
        if not self.bigrams:
            return self._empty_result()
        
        # Initialize accumulators
        pure_dvorak_sum = 0.0
        frequency_weighted_score = 0.0
        individual_criterion_sums = defaultdict(float)
        pure_individual_criterion_sums = defaultdict(float)
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
            
            # Get individual criterion scores for this bigram
            bigram_scores = score_bigram_dvorak9(key_pair)
            
            # 1. Pure Dvorak score (unweighted average)
            pure_bigram_score = sum(bigram_scores.values()) / len(bigram_scores)
            pure_dvorak_sum += pure_bigram_score
            total_bigrams_for_pure += 1
            
            # Track pure individual criterion scores
            for criterion, score in bigram_scores.items():
                pure_individual_criterion_sums[criterion] += score
            
            # 2. Frequency-weighted Dvorak score
            frequency_weighted_score += frequency * pure_bigram_score
            
            # Accumulate frequency-weighted individual criterion scores
            for criterion, score in bigram_scores.items():
                individual_criterion_sums[criterion] += frequency * score
            
            total_frequency += frequency
            covered_bigrams += 1
        
        # Calculate final scores
        pure_dvorak_score = pure_dvorak_sum / total_bigrams_for_pure if total_bigrams_for_pure > 0 else 0.0
        freq_weighted_dvorak_score = frequency_weighted_score / total_frequency if total_frequency > 0 else 0.0
        
        # Primary score is frequency-weighted (or empirical if weights available)
        primary_score = freq_weighted_dvorak_score
        
        # Calculate individual criterion averages
        criterion_names = ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                          'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']
        
        # Pure individual scores
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
        
        # Calculate coverage
        bigram_coverage = covered_bigrams / len(self.bigrams) if self.bigrams else 0.0
        
        # Create result
        result = ScoreResult(
            primary_score=primary_score,
            components={
                'pure_dvorak_score': pure_dvorak_score,
                'frequency_weighted_score': freq_weighted_dvorak_score,
                'hands': individual_scores['hands'],
                'fingers': individual_scores['fingers'],
                'skip_fingers': individual_scores['skip_fingers'],
                'dont_cross_home': individual_scores['dont_cross_home'],
                'same_row': individual_scores['same_row'],
                'home_row': individual_scores['home_row'],
                'columns': individual_scores['columns'],
                'strum': individual_scores['strum'],
                'strong_fingers': individual_scores['strong_fingers'],
            },
            metadata={
                'scoring_mode': 'frequency_weighted',
                'weights_type': self.weights_type,
                'theoretical_maximum': 1.0,
            },
            validation_info={
                'bigram_count': covered_bigrams,
                'total_bigrams': len(self.bigrams),
                'bigram_coverage': bigram_coverage,
                'coverage_percentage': bigram_coverage * 100,
            },
            detailed_breakdown={
                'pure_individual_scores': pure_individual_scores,
                'frequency_weighted_individual_scores': individual_scores,
                'scoring_approaches': {
                    'pure_dvorak': pure_dvorak_score,
                    'frequency_weighted': freq_weighted_dvorak_score,
                }
            }
        )
        
        return result
    
    def _empty_result(self) -> ScoreResult:
        """Return empty result for cases with no data."""
        return ScoreResult(
            primary_score=0.0,
            components={},
            metadata={'error': 'No bigram data available'},
            validation_info={'bigram_coverage': 0.0}
        )


@handle_common_errors
def main() -> int:
    """Main entry point using the standardized framework."""
    
    # Create standardized CLI parser
    cli_parser = create_standard_parser('dvorak9_scorer')
    args = cli_parser.parse_args()
    
    try:
        # Load configuration
        config = load_scorer_config('dvorak9_scorer', args.config)
        
        # Override with command-line arguments
        config['quiet_mode'] = args.quiet
        
        # Add weights file if specified
        if hasattr(args, 'weights') and args.weights:
            config['weights_file'] = args.weights
        
        # Get layout mapping from arguments
        letters, positions, layout_mapping = get_layout_from_args(args)
        
        # Filter to letters only (same as original behavior)
        layout_mapping = filter_to_letters_only(layout_mapping)
        
        if not layout_mapping:
            print("Error: No letters found in layout")
            return 1
        
        # Create and run scorer
        scorer = Dvorak9Scorer(layout_mapping, config)
        result = scorer.score_layout()
        
        # Print results using framework output utilities
        output_format = args.output_format
        output_config = config.get('output_formats', {}).get(output_format, {})
        
        print_results(result, output_format, output_config)
        
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())