#!/usr/bin/env python3
"""
Dvorak-9 Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

This script implements the 9 evaluation criteria derived from Dvorak's "Typing Behavior" 
book and patent (1936) with multiple scoring approaches including empirical weights.

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

Four scoring approaches are provided:
  - Pure Dvorak score: unweighted average of all 9 individual criteria
  - Frequency-weighted score: English bigram frequency-weighted average of 9 criteria
  - Speed-weighted score: frequency-weighted with empirical speed combination weights
  - Comfort-weighted score: frequency-weighted with empirical comfort combination weights

Usage:
    # Basic scoring (shows all four approaches)
    python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

    # With cross-hand filtering
    python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --ignore-cross-hand

    # CSV output
    python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --csv

    # Score only
    python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-only

Required input files:
  - normalized_letter_pair_frequencies_en.csv - English bigram frequencies
  - key_pair_scores.csv - Precomputed Dvorak-9 scores  
  - speed_weights.csv - Speed-based empirical weights (optional)
  - comfort_weights.csv - Comfort-based empirical weights (optional)

Each weights file contains correlations between each bigram's (speed or comfort) score 
and the combination of Dvorak criteria characterizing that bigram. 
These correlations can serve as weights in a layout scoring or optimization 
algorithm, where a correlation is used as a weight to emphasize the 
contribution of a bigram on the layout's score.

"""

import sys
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

# Import our framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config
from framework.layout_utils import filter_to_letters_only, is_same_hand_pair
from framework.data_utils import load_bigram_frequencies, load_key_value_csv, validate_data_consistency
from framework.output_utils import print_results
from framework.cli_utils import create_standard_parser, handle_common_errors, get_layout_from_args


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

# Define finger strength and home row
STRONG_FINGERS = {1, 2}
WEAK_FINGERS   = {3, 4}
HOME_ROW = 2

# Define finger column assignments
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
    """Calculate all 9 Dvorak criteria scores for a bigram."""
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


def load_combination_weights(csv_path: str, quiet: bool = False) -> Dict[Tuple[str, ...], float]:
    """Load empirical correlation weights for different feature combinations from CSV file."""
    if not Path(csv_path).exists():
        if not quiet:
            print(f"Warning: Combination weights file not found: {csv_path}")
        return {}
    
    combination_weights: Dict[Tuple[str, ...], float] = {}
    
    try:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                combination_str = row.get('combination', '').strip()
                correlation_str = row.get('correlation', '').strip()
                
                if not correlation_str:
                    continue
                
                try:
                    correlation = float(correlation_str)
                except ValueError:
                    continue
                
                # Convert combination string to tuple
                if combination_str.lower() in ['none', 'empty', '']:
                    combination = ()
                else:
                    if '+' in combination_str:
                        features = [f.strip() for f in combination_str.split('+')]
                        combination = tuple(sorted(features))
                    else:
                        combination = (combination_str,)
                
                combination_weights[combination] = correlation
                
    except Exception as e:
        if not quiet:
            print(f"Warning: Error parsing combination weights CSV: {e}")
        return {}
    
    # Ensure we have at least an empty combination
    if () not in combination_weights:
        combination_weights[()] = 0.0
    
    if not quiet:
        print(f"Loaded {len(combination_weights)} combination weights from {csv_path}")
    
    return combination_weights


def identify_bigram_combination(bigram_scores: Dict[str, float], threshold: float = 0.0) -> Tuple[str, ...]:
    """Identify which feature combination a bigram exhibits."""
    active_features = []
    
    for feature, score in bigram_scores.items():
        if score >= threshold:
            active_features.append(feature)
    
    return tuple(sorted(active_features))


def score_bigram_weighted(bigram_scores: Dict[str, float], 
                         combination_weights: Dict[Tuple[str, ...], float]) -> Optional[float]:
    """Score a single bigram using exact empirical combination matching only."""
    # Identify which combination this bigram exhibits
    combination = identify_bigram_combination(bigram_scores)
    
    # Only use exact matches
    if combination in combination_weights:
        weight = combination_weights[combination]
        combination_strength = sum(bigram_scores[feature] for feature in combination) / len(combination) if combination else 0
        return weight * combination_strength
    
    # No exact match found
    return None


class Dvorak9Scorer(BaseLayoutScorer):
    """
    Dvorak-9 scorer with four scoring approaches.
    
    Implements the 9 evaluation criteria derived from Dvorak's work with
    four different scoring approaches: pure, frequency-weighted, speed-weighted, and comfort-weighted.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """Initialize the Dvorak-9 scorer."""
        super().__init__(layout_mapping, config)
        
        # Data containers
        self.bigrams: List[str] = []
        self.frequencies: List[float] = []
        self.key_pair_scores: Dict[str, float] = {}
        self.speed_weights: Optional[Dict[Tuple[str, ...], float]] = None
        self.comfort_weights: Optional[Dict[Tuple[str, ...], float]] = None
        
        # Cross-hand filtering option
        scoring_options = self.config.get('scoring_options', {})
        self.ignore_cross_hand = scoring_options.get('ignore_cross_hand', False)
    
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
        
        # Load speed weights (optional)
        speed_weights_file = data_files.get('speed_weights')
        if speed_weights_file:
            self.speed_weights = load_combination_weights(speed_weights_file, quiet=quiet_mode)
        
        # Load comfort weights (optional)
        comfort_weights_file = data_files.get('comfort_weights')
        if comfort_weights_file:
            self.comfort_weights = load_combination_weights(comfort_weights_file, quiet=quiet_mode)

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
        """Calculate layout scores using all four scoring approaches."""
        if not self.bigrams:
            return self._empty_result()
        
        # Initialize accumulators for all scoring approaches
        pure_dvorak_sum = 0.0
        frequency_weighted_score = 0.0
        individual_criterion_sums = defaultdict(float)
        pure_individual_criterion_sums = defaultdict(float)
        total_frequency = 0.0
        covered_bigrams = 0
        total_bigrams_for_pure = 0
        
        # Empirical scoring accumulators
        speed_weighted_score = 0.0
        comfort_weighted_score = 0.0
        speed_exact_matches_count = 0
        speed_exact_matches_frequency = 0.0
        comfort_exact_matches_count = 0
        comfort_exact_matches_frequency = 0.0

        # Process each bigram in the frequency list
        for bigram, frequency in zip(self.bigrams, self.frequencies):
            char1, char2 = bigram[0].lower(), bigram[1].lower()
            
            # Check if both characters are in our layout
            if char1 not in self.layout_mapping or char2 not in self.layout_mapping:
                continue
            
            # Map characters to QWERTY positions
            pos1 = self.layout_mapping[char1]
            pos2 = self.layout_mapping[char2]
            
            # Apply cross-hand filtering if requested
            if self.ignore_cross_hand and not is_same_hand_pair(pos1, pos2):
                continue
            
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
            
            # 3. Speed-weighted score (if weights provided)
            if self.speed_weights:
                weighted_score = score_bigram_weighted(bigram_scores, self.speed_weights)
                
                if weighted_score is not None:
                    speed_exact_matches_count += 1
                    speed_exact_matches_frequency += frequency
                    # Apply correct sign (speed weights are negative correlations)
                    final_speed_score = -weighted_score
                    speed_weighted_score += frequency * final_speed_score
            
            # 4. Comfort-weighted score (if weights provided)
            if self.comfort_weights:
                weighted_score = score_bigram_weighted(bigram_scores, self.comfort_weights)
                
                if weighted_score is not None:
                    comfort_exact_matches_count += 1
                    comfort_exact_matches_frequency += frequency
                    # Comfort weights are positive correlations
                    comfort_weighted_score += frequency * weighted_score

            # Accumulate frequency-weighted individual criterion scores
            for criterion, score in bigram_scores.items():
                individual_criterion_sums[criterion] += frequency * score
            
            total_frequency += frequency
            covered_bigrams += 1
        
        # Calculate final scores
        pure_dvorak_score = pure_dvorak_sum / total_bigrams_for_pure if total_bigrams_for_pure > 0 else 0.0
        freq_weighted_dvorak_score = frequency_weighted_score / total_frequency if total_frequency > 0 else 0.0
        
        # Calculate empirically-weighted scores
        final_speed_score = None
        final_comfort_score = None
        
        if self.speed_weights and speed_exact_matches_frequency > 0:
            final_speed_score = speed_weighted_score / speed_exact_matches_frequency
            
        if self.comfort_weights and comfort_exact_matches_frequency > 0:
            final_comfort_score = comfort_weighted_score / comfort_exact_matches_frequency

        # Determine primary score (prefer empirical, fallback to frequency-weighted)
        if final_speed_score is not None:
            primary_score = final_speed_score
        elif final_comfort_score is not None:
            primary_score = final_comfort_score
        else:
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
        
        # Create result with all four scoring approaches
        components = {
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
        }
        
        # Add empirical scores if available
        if final_speed_score is not None:
            components['speed_weighted_score'] = final_speed_score
        if final_comfort_score is not None:
            components['comfort_weighted_score'] = final_comfort_score
        
        # Create empirical coverage information
        empirical_coverage = {}
        if self.speed_weights:
            empirical_coverage['speed_exact_matches_count'] = speed_exact_matches_count
            empirical_coverage['speed_exact_matches_percentage'] = (speed_exact_matches_count / covered_bigrams * 100) if covered_bigrams > 0 else 0.0
            empirical_coverage['speed_exact_matches_frequency_weight'] = (speed_exact_matches_frequency / total_frequency * 100) if total_frequency > 0 else 0.0
        
        if self.comfort_weights:
            empirical_coverage['comfort_exact_matches_count'] = comfort_exact_matches_count
            empirical_coverage['comfort_exact_matches_percentage'] = (comfort_exact_matches_count / covered_bigrams * 100) if covered_bigrams > 0 else 0.0
            empirical_coverage['comfort_exact_matches_frequency_weight'] = (comfort_exact_matches_frequency / total_frequency * 100) if total_frequency > 0 else 0.0
        
        result = ScoreResult(
            primary_score=primary_score,
            components=components,
            metadata={
                'scoring_approaches': {
                    'pure_available': True,
                    'frequency_weighted_available': True,
                    'speed_weighted_available': final_speed_score is not None,
                    'comfort_weighted_available': final_comfort_score is not None,
                },
                'ignore_cross_hand': self.ignore_cross_hand,
                'theoretical_maximum': 1.0,
                'available_speed_combinations': len(self.speed_weights) if self.speed_weights else 0,
                'available_comfort_combinations': len(self.comfort_weights) if self.comfort_weights else 0,
            },
            validation_info={
                'bigram_count': covered_bigrams,
                'total_bigrams': len(self.bigrams),
                'bigram_coverage': bigram_coverage,
                'coverage_percentage': bigram_coverage * 100,
                'cross_hand_filtering': self.ignore_cross_hand,
            },
            detailed_breakdown={
                'pure_individual_scores': pure_individual_scores,
                'frequency_weighted_individual_scores': individual_scores,
                'scoring_approaches': {
                    'pure_dvorak': pure_dvorak_score,
                    'frequency_weighted': freq_weighted_dvorak_score,
                    'speed_weighted': final_speed_score,
                    'comfort_weighted': final_comfort_score,
                },
                'empirical_weight_coverage': empirical_coverage,
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
        
        # Handle cross-hand filtering
        if hasattr(args, 'ignore_cross_hand') and args.ignore_cross_hand:
            if 'scoring_options' not in config:
                config['scoring_options'] = {}
            config['scoring_options']['ignore_cross_hand'] = True
        
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